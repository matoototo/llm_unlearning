import math
import torch
import einops
import torch.nn.functional as F

from typing import Any, Dict, Tuple, List, Callable
from transformers import PreTrainedModel
from llm_unlearning.evals.utils import sequence_nll

def check_inputs(required_inputs: List[str], **kwargs):
    missing_inputs = [input_name for input_name in required_inputs if input_name not in kwargs]

    if missing_inputs:
        raise ValueError(f"Missing required input(s): {', '.join(missing_inputs)}")

def cosine_annealing(start_factor, time_scale):
    def schedule(t: float) -> float:
        if t < time_scale:
            scale = (math.cos(math.pi * t / time_scale) + 1) / 2
            return 1 + (start_factor - 1) * scale
        else:
            return 1.0
    return schedule

schedule_map = {
    "10x_cosine": cosine_annealing(10, 1),
    "10x_cosine_half": cosine_annealing(10, 0.5),
    "10x_cosine_quarter": cosine_annealing(10, 0.25),
}

class Method:
    def __init__(self, **kwargs):
        self.setup(**kwargs)
        self.input_keys = ["input_ids", "inputs_embeds", "attention_mask", "labels"]
        self.schedules = {}
        self.original_values = {}

    def setup(self, **kwargs):
        pass

    def register_scheduled_value(self, name: str, schedule_name: str):
        schedule = schedule_map[schedule_name]
        self.schedules[name] = schedule
        self.original_values[name] = getattr(self, name)

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
        super().__init__(**kwargs)
        self.beta = kwargs.get("beta", 0.1)
        self.retain_coeff = kwargs.get("retain_coeff", 1.0)
        self.hat_coeff = kwargs.get("hat_coeff", 0)
        self.schedule_beta = kwargs.get("schedule_beta", False)
        if self.schedule_beta:
            self.register_scheduled_value("beta", self.schedule_beta)

    def compute_loss(self, model: PreTrainedModel, **kwargs) -> Tuple[torch.Tensor, Dict[str, float], Any]:
        check_inputs(["forget_inputs", "retain_inputs"], **kwargs)
        check_inputs(["inputs_embeds", "delta"], **kwargs["forget_inputs"])
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

        if self.retain_coeff:
            retain_inputs = {k: v for k, v in kwargs['retain_inputs'].items() if k in self.input_keys}
            retain_outputs = model(**retain_inputs)
            retain_loss = retain_outputs.loss
            npo_loss += self.retain_coeff * retain_loss

        if self.hat_coeff:
            hat_inputs_embeds = kwargs["forget_inputs"]["inputs_embeds"] + kwargs["forget_inputs"]["delta"]
            hat_outputs = model(inputs_embeds=hat_inputs_embeds, attention_mask=forget_inputs["attention_mask"])
            with torch.no_grad():
                hat_reference_outputs = reference_model(inputs_embeds=hat_inputs_embeds, attention_mask=forget_inputs["attention_mask"])
            hat_loss = F.kl_div(F.log_softmax(hat_outputs.logits, dim=-1), F.softmax(hat_reference_outputs.logits, dim=-1), reduction="batchmean")
            npo_loss += self.hat_coeff * hat_loss

        loss_dict = {
            "npo_loss": npo_loss.item(),
            "forget_loss": forget_loss.mean().item(),
            "reference_loss": reference_loss.mean().item(),
            "hat_loss": hat_loss.item() if self.hat_coeff else 0,
            "retain_loss": retain_loss.item() if self.retain_coeff else 0,
        }

        return npo_loss, loss_dict, (forget_outputs, reference_outputs)

class RMU(Method):
    # https://arxiv.org/abs/2403.03218
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steering_coeff = kwargs.get("steering_coeff", 10.0)
        self.alpha = kwargs.get("alpha", 100.0)
        self.layer_id = kwargs.get("layer_id", 5)
        self.module_str = kwargs.get("module_str", "{model_name}.model.layers[{layer_id}]")
        self.control_vec = None
        self.frozen_layers = []

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

def get_method(method_name: str, **kwargs) -> Method:
    methods = {
        "gradient_ascent": GradientAscent,
        "gradient_descent": GradientDescent,
        "gradient_difference": GradientDifference,
        "npo": NPO,
        "rmu": RMU,
    }

    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}")

    return methods[method_name](**kwargs)
