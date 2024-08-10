import os
import torch
import einops
import torch.nn.functional as F

from omegaconf import OmegaConf
from typing import Any, Dict, Tuple, List
from transformers import PreTrainedModel
from sklearn.covariance import EmpiricalCovariance
from llm_unlearning.evals.utils import sequence_nll
from llm_unlearning.utils.schedules import get_schedule


def check_inputs(required_inputs: List[str], **kwargs):
    missing_inputs = [input_name for input_name in required_inputs if input_name not in kwargs]

    if missing_inputs:
        raise ValueError(f"Missing required input(s): {', '.join(missing_inputs)}")

class Method:
    def __init__(self, **kwargs):
        self.setup(**kwargs)
        self.input_keys = ["input_ids", "inputs_embeds", "attention_mask", "labels"]
        self.schedules = {}
        self.original_values = {}
        self.register_scheduled_values(kwargs)

    def setup(self, **kwargs):
        pass

    def register_scheduled_values(self, config):
        for key, value in config.items():
            if not key.startswith("schedule_"): continue
            param_name = key[len("schedule_"):]
            if not hasattr(self, param_name):
                raise AttributeError(f"Method has no attribute '{param_name}' to schedule")
            original_value = getattr(self, param_name)
            try:
                self.schedules[param_name] = get_schedule(value)
                self.original_values[param_name] = original_value
            except ValueError as e:
                raise ValueError(f"Error creating schedule for '{param_name}': {str(e)}")

    def update_scheduled_values(self, step_ratio: float):
        for name, schedule in self.schedules.items():
            original_value = self.original_values[name]
            setattr(self, name, schedule(step_ratio) * original_value)

    def compute_loss(self, model: PreTrainedModel, **kwargs) -> Tuple[torch.Tensor, Dict[str, float], Any]:
        raise NotImplementedError("Subclasses must implement this method")

    def prehook(self, trainer, model, inputs):
        pass

    def posthook(self, trainer, model, inputs, loss):
        pass

class GradientDescent(Method):
    def compute_loss(self, model: PreTrainedModel, **kwargs) -> Tuple[torch.Tensor, Dict[str, float], Any]:
        check_inputs(["forget_inputs"], **kwargs)
        self.update_scheduled_values(kwargs.get("step_ratio", 1.0))

        ft_inputs = {k: v for k, v in kwargs['forget_inputs'].items() if k in self.input_keys}
        outputs = model(**ft_inputs)
        loss_ft = outputs.loss

        loss_dict = {
            "loss_ft": loss_ft.item(),
        }

        return loss_ft, loss_dict, outputs

class GradientAscent(Method):
    def compute_loss(self, model: PreTrainedModel, **kwargs) -> Tuple[torch.Tensor, Dict[str, float], Any]:
        check_inputs(["forget_inputs"], **kwargs)
        self.update_scheduled_values(kwargs.get("step_ratio", 1.0))

        forget_inputs = {k: v for k, v in kwargs['forget_inputs'].items() if k in self.input_keys}
        outputs = model(**forget_inputs)
        forget_loss = outputs.loss * -1

        loss_dict = {
            "loss_forget": forget_loss.item(),
        }

        return forget_loss, loss_dict, outputs

class GradientDifference(Method):
    def compute_loss(self, model: PreTrainedModel, **kwargs) -> Tuple[torch.Tensor, Dict[str, float], Any]:
        check_inputs(["forget_inputs", "retain_inputs"], **kwargs)
        self.update_scheduled_values(kwargs.get("step_ratio", 1.0))

        forget_inputs = {k: v for k, v in kwargs['forget_inputs'].items() if k in self.input_keys}
        retain_inputs = {k: v for k, v in kwargs['retain_inputs'].items() if k in self.input_keys}

        forget_outputs = model(**forget_inputs)
        forget_loss = forget_outputs.loss * -1

        retain_outputs = model(**retain_inputs)
        retain_loss = retain_outputs.loss

        total_loss = forget_loss + retain_loss

        loss_dict = {
            "loss_forget": forget_loss.item(),
            "loss_retain": retain_loss.item(),
        }

        return total_loss, loss_dict, (forget_outputs, retain_outputs)

class NPO(Method):
    # https://arxiv.org/abs/2404.05868
    def __init__(self, **kwargs):
        self.beta = kwargs.get("beta", 0.1)
        self.retain_coeff = kwargs.get("retain_coeff", 1.0)
        super().__init__(**kwargs)

    def compute_loss(self, model: PreTrainedModel, **kwargs) -> Tuple[torch.Tensor, Dict[str, float], Any]:
        check_inputs(["forget_inputs", "retain_inputs"], **kwargs)
        self.update_scheduled_values(kwargs.get("step_ratio", 1.0))
        reference_model = kwargs.get("reference_model") if "reference_model" in kwargs else None
        if not reference_model: raise ValueError("NPO requires a config.reference_model to be set")

        forget_inputs = {k: v for k, v in kwargs['forget_inputs'].items() if k in self.input_keys}

        forget_outputs = model(**forget_inputs)
        forget_loss = sequence_nll(forget_outputs.logits, forget_inputs["labels"])
        with torch.no_grad():
            reference_outputs = reference_model(**forget_inputs)
            reference_loss = sequence_nll(reference_outputs.logits, forget_inputs["labels"])

        neg_logloss_ratio = (forget_loss - reference_loss)
        npo_loss = (F.logsigmoid(self.beta * neg_logloss_ratio) * -2 / self.beta).mean()
        loss = npo_loss

        if self.retain_coeff:
            retain_inputs = {k: v for k, v in kwargs['retain_inputs'].items() if k in self.input_keys}
            retain_outputs = model(**retain_inputs)
            retain_loss = retain_outputs.loss
            loss += self.retain_coeff * retain_loss

        loss_dict = {
            "npo_loss": npo_loss.item(),
            "forget_loss": forget_loss.mean().item(),
            "reference_loss": reference_loss.mean().item(),
            "retain_loss": retain_loss.item() if self.retain_coeff else 0,
        }

        return loss, loss_dict, (forget_outputs, reference_outputs)

class RMU(Method):
    # https://arxiv.org/abs/2403.03218
    def __init__(self, **kwargs):
        self.steering_coeff = kwargs.get("steering_coeff", 10.0)
        self.alpha = kwargs.get("alpha", 100.0)
        self.layer_id = kwargs.get("layer_id", 5)
        self.module_str = kwargs.get("module_str", "{model_name}.model.layers[{layer_id}]")
        self.control_vec = None
        self.frozen_layers = []
        super().__init__(**kwargs)

    def freeze_layers(self, model: PreTrainedModel):
        target_layer = self.layer_id
        layers_to_update = [target_layer - 2, target_layer - 1, target_layer]
        for name, param in model.named_parameters():
            if not any(f"layers.{layer}.mlp" in name for layer in layers_to_update):
                param.requires_grad = False
                self.frozen_layers.append(name)
            else:
                param.requires_grad = True

    def unfreeze_layers(self, model: PreTrainedModel):
        for name, param in model.named_parameters():
            if name in self.frozen_layers:
                param.requires_grad = True
        self.frozen_layers = []

    def prehook(self, trainer, model, inputs):
        self.freeze_layers(model)

    def posthook(self, trainer, model, inputs, loss):
        self.unfreeze_layers(model)

    def get_module_activations(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        activations = []

        def hook_fn(module, input, output):
            activations.append(output)

        module = eval(self.module_str.format(model_name="model", layer_id=self.layer_id))
        handle = module.register_forward_hook(hook_fn)

        with torch.set_grad_enabled(model.training):
            model(**inputs)

        handle.remove()
        return activations[0][0]

    def compute_loss(self, model: PreTrainedModel, **kwargs) -> Tuple[torch.Tensor, Dict[str, float], Any]:
        check_inputs(["forget_inputs", "retain_inputs"], **kwargs)
        self.update_scheduled_values(kwargs.get("step_ratio", 1.0))
        reference_model = kwargs.get("reference_model") if "reference_model" in kwargs else None
        if not reference_model: raise ValueError("RMU requires a config.reference_model to be set")

        forget_inputs = {k: v for k, v in kwargs['forget_inputs'].items() if k in self.input_keys}
        retain_inputs = {k: v for k, v in kwargs['retain_inputs'].items() if k in self.input_keys}

        # Control vector is kept the same throughout unlearning
        if self.control_vec is None:
            self.control_vec = torch.rand(1, 1, model.config.hidden_size, dtype=model.dtype, device=model.device)
            self.control_vec = self.control_vec / torch.norm(self.control_vec) * self.steering_coeff

        forget_activations = self.get_module_activations(model, forget_inputs)
        control_vec_repeat = einops.repeat(self.control_vec, "1 1 d -> b n d", b=forget_activations.size(0), n=forget_activations.size(1))
        unlearn_loss = F.mse_loss(forget_activations, control_vec_repeat)

        retain_activations = self.get_module_activations(model, retain_inputs)
        with torch.no_grad():
            reference_retain_activations = self.get_module_activations(reference_model, retain_inputs)
        retain_loss = F.mse_loss(retain_activations, reference_retain_activations)
        retain_loss *= self.alpha

        total_loss = unlearn_loss + retain_loss

        loss_dict = {
            "unlearn_loss": unlearn_loss.item(),
            "retain_loss": retain_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict, (forget_activations, retain_activations)

class EmbeddingRemapping(Method):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = kwargs.get("layer_id", 5)
        self.num_at_supports = kwargs.get("num_at_supports", 10)
        self.num_inner_at_iterations = kwargs.get("num_inner_at_iterations", 2)
        self.epsilon = kwargs.get("epsilon", 0.01)
        self.alpha = kwargs.get('alpha', 0.001)
        self.boundary_type = kwargs.get("boundary_type", "epsilon_ball")
        self.boundaries = []

    def get_layer_embedding(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor], with_grad: bool = False) -> torch.Tensor:
        embeddings = []

        def hook_fn(module, input, output):
            embeddings.append(output)

        layer = model.model.layers[self.layer_id]
        handle = layer.register_forward_hook(hook_fn)


        valid_inputs = {k: v for k, v in inputs.items() if k in self.input_keys}
        with torch.set_grad_enabled(with_grad):
            model(**valid_inputs)

        handle.remove()
        return embeddings[0] if type(embeddings[0]) == torch.Tensor else embeddings[0][0]

    def get_target_token_embedding(self, inputs: Dict[str, torch.Tensor], layer_embedding: torch.Tensor) -> torch.Tensor:
        # Maybe we want to do something else here? Currently it's the first non-tag answer token.
        if 'question_length' not in inputs:
            question_end = (inputs['input_ids'] == 33706).nonzero(as_tuple=True)[1]
        else:
            question_end = inputs['question_length'] # [..., 33706|, 25, first_token, ...] where 33706,25 is "Answer:"
        return layer_embedding[torch.arange(layer_embedding.size(0)), question_end + 2]

    def adversarial_search(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        original_embeddings = model.get_input_embeddings()(input_ids)
        layer_embeddings = self.get_layer_embedding(model, inputs)
        target_token_embeddings = [self.get_target_token_embedding(inputs, layer_embeddings)]

        for _outer in range(self.num_at_supports - 1):
            random_direction = torch.rand_like(original_embeddings)
            random_direction = F.normalize(random_direction, dim=-1)
            perturbed_embeddings = original_embeddings.clone().detach() + self.epsilon * random_direction

            for _inner in range(self.num_inner_at_iterations):
                perturbed_embeddings.requires_grad = True
                model.zero_grad()

                layer_embedding = self.get_layer_embedding(model, {'inputs_embeds': perturbed_embeddings, 'attention_mask': attention_mask}, with_grad=True)
                target_token_embedding = self.get_target_token_embedding(inputs, layer_embedding)

                distances = torch.cdist(target_token_embedding.unsqueeze(0), torch.stack(target_token_embeddings, dim=1))
                distances = einops.rearrange(distances, "b p r -> b (p r)")
                loss = torch.min(distances, dim=1).values.sum()
                loss.backward()

                with torch.no_grad():
                    # TODO: Do we want to perturb the full sequence embedding or just the target token?
                    grad = perturbed_embeddings.grad
                    perturbed_embeddings = perturbed_embeddings + self.alpha * grad.sign()
                    delta = self.project_l2(perturbed_embeddings - original_embeddings)
                    perturbed_embeddings = (original_embeddings + delta).detach()

            layer_embedding = self.get_layer_embedding(model, {'inputs_embeds': perturbed_embeddings, 'attention_mask': attention_mask})
            target_token_embedding = self.get_target_token_embedding(inputs, layer_embedding)
            target_token_embeddings.append(target_token_embedding.detach())

        return target_token_embeddings

    def project_l2(self, delta):
        norm = delta.norm(p=2, dim=-1, keepdim=True)
        mask = norm > self.epsilon
        return torch.where(mask, delta * self.epsilon / norm, delta)

    def fit_boundary(self, embeddings: List[torch.Tensor]) -> Any:
        if self.boundary_type == "epsilon_ball":
            center = torch.mean(torch.stack(embeddings), dim=0)
            radius = torch.max(torch.cdist(center.unsqueeze(0), torch.stack(embeddings)))
            return center, radius
        elif self.boundary_type == "ellipsoid":
            embeddings_np = torch.stack(embeddings).flatten(1).cpu().numpy()
            cov = EmpiricalCovariance().fit(embeddings_np)
            return torch.tensor(cov.location_, device=embeddings[0].device), torch.tensor(cov.covariance_, device=embeddings[0].device)
        elif self.boundary_type == "axis_aligned":
            stacked_embeddings = torch.stack(embeddings)
            center = torch.mean(stacked_embeddings, dim=0)
            variances = torch.var(stacked_embeddings, dim=0)
            return center, variances
        else:
            raise ValueError(f"Unsupported boundary type: {self.boundary_type}")

    def project_embedding(self, embedding: torch.Tensor, boundary: Any) -> torch.Tensor:
        if self.boundary_type == "epsilon_ball":
            center, radius = boundary
            direction = embedding - center
            return center + (radius + self.epsilon) * F.normalize(direction, dim=-1)
        elif self.boundary_type == "ellipsoid":
            center, cov = boundary
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            scaled_direction = torch.matmul(embedding - center, eigenvectors) / torch.sqrt(eigenvalues)
            scaled_direction = F.normalize(scaled_direction, dim=-1)
            return center + torch.matmul(scaled_direction * (torch.max(eigenvalues) + self.epsilon), eigenvectors.T)
        elif self.boundary_type == "axis_aligned":
            center, variances = boundary
            direction = embedding - center
            scaled_direction = direction / torch.sqrt(variances)
            max_variance = torch.max(variances)
            return center + (max_variance + self.epsilon) * F.normalize(scaled_direction, dim=-1)
        else:
            raise ValueError(f"Unsupported boundary type: {self.boundary_type}")

    def is_inside_boundary(self, embedding: torch.Tensor, boundary: Any) -> bool:
        if self.boundary_type == "epsilon_ball":
            center, radius = boundary
            distance = torch.norm(embedding - center)
            return distance <= radius
        elif self.boundary_type == "ellipsoid":
            center, cov = boundary
            diff = embedding - center
            mahalanobis_distance = torch.sqrt(torch.einsum('i,ij,j->', diff, torch.inverse(cov), diff))
            return mahalanobis_distance <= 1
        elif self.boundary_type == "axis_aligned":
            center, variances = boundary
            normalized_distances = ((embedding - center) ** 2) / variances
            return torch.all(normalized_distances <= 1)
        else:
            raise ValueError(f"Unsupported boundary type: {self.boundary_type}")

    def compute_retain_loss(self, retain_embeddings: torch.Tensor, boundary: Any) -> torch.Tensor:
        if self.boundary_type == "epsilon_ball":
            center, radius = boundary
            retain_distances = torch.norm(retain_embeddings - center.unsqueeze(0), dim=-1)
            return F.relu(radius - retain_distances).mean()
        elif self.boundary_type == "ellipsoid":
            center, cov = boundary
            retain_distances = torch.sum(torch.matmul(retain_embeddings - center.unsqueeze(0), torch.inverse(cov)) * (retain_embeddings - center.unsqueeze(0)), dim=-1)
            return F.relu(1 - retain_distances).mean()
        elif self.boundary_type == "axis_aligned":
            center, variances = boundary
            normalized_distances = (retain_embeddings - center.unsqueeze(0)) ** 2 / variances.unsqueeze(0)
            retain_distances = torch.sum(normalized_distances, dim=-1)
            return F.relu(1 - retain_distances).mean()

    def compute_loss(self, model: PreTrainedModel, **kwargs) -> Tuple[torch.Tensor, Dict[str, float], Any]:
        check_inputs(["forget_inputs", "retain_inputs"], **kwargs)
        self.update_scheduled_values(kwargs.get("step_ratio", 1.0))

        forget_inputs = kwargs['forget_inputs']
        retain_inputs = kwargs['retain_inputs']

        # Perform adversarial search for each item in the batch TODO: Parallelize
        all_forget_embeddings = []
        for i in range(forget_inputs['input_ids'].size(0)):
            single_forget_input = {k: v[i].unsqueeze(0) for k, v in forget_inputs.items()}
            forget_embeddings = self.adversarial_search(model, single_forget_input)
            all_forget_embeddings.append(forget_embeddings)

        # Fit boundaries and save them
        new_boundaries = []
        for forget_embeddings in all_forget_embeddings:
            boundary = self.fit_boundary(forget_embeddings)
            new_boundaries.append(boundary)
        self.boundaries.extend(new_boundaries)

        # TODO: Do we want this?
        # Compute retain loss against all saved boundaries
        # retain_embeddings = self.get_layer_embedding(model, retain_inputs, with_grad=True)
        # retain_loss = 0
        # for saved_boundary in self.boundaries:
        #     retain_loss += self.compute_retain_loss(retain_embeddings, saved_boundary)
        # retain_loss /= len(self.boundaries)

        # Return a dummy loss of 0 to keep the trainer happy
        total_loss = torch.tensor(0.0, requires_grad=True, device=model.device)

        loss_dict = {
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict, None

    def save_config(self, save_directory: str):
        config = {
            "layer_id": self.layer_id,
            "num_at_supports": self.num_at_supports,
            "num_inner_at_iterations": self.num_inner_at_iterations,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "boundary_type": self.boundary_type,
        }
        config_path = os.path.join(save_directory, "embedding_remapping_config.yaml")
        OmegaConf.save(config, config_path)

    def save_boundaries(self, save_directory: str):
        boundaries_path = os.path.join(save_directory, "embedding_boundaries.pt")
        torch.save(self.boundaries, boundaries_path)
        self.save_config(save_directory)

    @classmethod
    def load_config(cls, load_directory: str):
        config_path = os.path.join(load_directory, "embedding_remapping_config.yaml")
        if os.path.exists(config_path):
            return OmegaConf.load(config_path)
        raise FileNotFoundError(f"Config file not found at {config_path}")

    @classmethod
    def load_boundaries(cls, load_directory: str):
        boundaries_path = os.path.join(load_directory, "embedding_boundaries.pt")
        if os.path.exists(boundaries_path):
            return torch.load(boundaries_path)
        else:
            return []


def get_method(method_name: str, **kwargs) -> Method:
    methods = {
        "gradient_ascent": GradientAscent,
        "gradient_descent": GradientDescent,
        "gradient_difference": GradientDifference,
        "npo": NPO,
        "rmu": RMU,
        "embedding_remapping": EmbeddingRemapping,
    }

    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}")

    return methods[method_name](**kwargs)
