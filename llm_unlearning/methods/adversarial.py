import torch
import torch.nn as nn

from typing import Dict

class AdversarialAttack:
    def __init__(self, **kwargs):
        pass

    def attack(self, model: nn.Module, inputs: Dict[str, torch.Tensor], is_grad_accumulation: bool) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement this method")

class PGDAttack(AdversarialAttack):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.alpha = kwargs.get('alpha', 0.001)
        self.num_iterations = kwargs.get('num_iterations', 3)
        self.target_layer = kwargs.get('target_layer', 'model.embed_tokens')

    @staticmethod
    def get_module(model: nn.Module, target: str) -> nn.Module:
        if isinstance(model, nn.DataParallel):
            model = model.module

        parts = target.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            elif part.startswith('[') and part.endswith(']'):
                module = module[int(part[1:-1])]
            else:
                module = getattr(module, part)
        return module

    def get_layer_output(self, model, inputs):
        activations = []

        def hook_fn(module, input, output):
            activations.append(output)

        module = self.get_module(model, self.target_layer)
        handle = module.register_forward_hook(hook_fn)

        with torch.no_grad():
            input_keys = ['input_ids', 'attention_mask', 'labels']
            model(**{k: inputs[k] for k in input_keys})

        handle.remove()
        return activations[0]

    def attack(self, model: nn.Module, inputs: Dict[str, torch.Tensor], is_grad_accumulation: float) -> Dict[str, torch.Tensor]:
        original_grads = [param.grad.clone() if is_grad_accumulation and param.grad is not None else None for param in model.parameters()]

        forget_inputs = inputs['forget_inputs']
        target_module = self.get_module(model, self.target_layer)

        original_output = self.get_layer_output(model, forget_inputs)
        perturbed_output = original_output.clone().detach()

        for _ in range(self.num_iterations):
            perturbed_output.requires_grad = True

            def forward_hook(module, input, output):
                return perturbed_output

            handle = target_module.register_forward_hook(forward_hook)

            input_keys = ['input_ids', 'attention_mask', 'labels']
            outputs = model(**{k: forget_inputs[k] for k in input_keys})
            loss = outputs.loss
            loss.backward()

            handle.remove()

            with torch.no_grad():
                grad = perturbed_output.grad
                perturbed_output = perturbed_output - self.alpha * grad.sign()
                delta = self.project_l2(perturbed_output - original_output)
                perturbed_output = (original_output + delta).detach()

        def pre_hook(model):
            def hook(module, input, output):
                return perturbed_output

            handle = target_module.register_forward_hook(hook)
            return handle

        def post_hook(outputs, handle):
            handle.remove()
            if isinstance(outputs, torch.Tensor):
                return outputs.detach()
            elif isinstance(outputs, tuple):
                return tuple(x.detach() if isinstance(x, torch.Tensor) else x for x in outputs)
            else:
                return outputs

        forget_inputs['pre_hook'] = pre_hook
        forget_inputs['post_hook'] = post_hook

        inputs['forget_inputs'] = forget_inputs

        model.zero_grad()

        if is_grad_accumulation:
            for param, grad in zip(model.parameters(), original_grads):
                param.grad = grad

        return inputs

    def project_l2(self, delta):
        norm = delta.norm(p=2, dim=-1, keepdim=True)
        mask = norm > self.epsilon
        return torch.where(mask, delta * self.epsilon / norm, delta)

def get_attack(attack_name: str, **kwargs) -> AdversarialAttack:
    attacks = {
        "pgd": PGDAttack,
    }

    if attack_name not in attacks:
        raise ValueError(f"Unknown attack: {attack_name}")

    return attacks[attack_name](**kwargs)
