import torch
import einops

from typing import Dict

class AdversarialAttack:
    def __init__(self, **kwargs):
        pass

    def attack(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], is_grad_accumulation: bool) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement this method")

class PGDAttack(AdversarialAttack):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.alpha = kwargs.get('alpha', 0.001)
        self.num_iterations = kwargs.get('num_iterations', 3)

    def attack(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], is_grad_accumulation: bool) -> Dict[str, torch.Tensor]:
        input_ids = inputs['forget_inputs']['input_ids']
        attention_mask = inputs['forget_inputs']['attention_mask']
        labels = inputs['forget_inputs']['labels']

        original_grads = [param.grad.clone() if is_grad_accumulation and param.grad is not None else None for param in model.parameters()]

        original_embeddings = model.get_input_embeddings()(input_ids)
        perturbed_embeddings = original_embeddings.clone().detach()

        for _ in range(self.num_iterations):
            perturbed_embeddings.requires_grad = True
            model.zero_grad()

            outputs = model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            with torch.no_grad():
                grad = perturbed_embeddings.grad
                perturbed_embeddings = perturbed_embeddings - self.alpha * grad.sign()
                delta = self.project_l2(perturbed_embeddings - original_embeddings)
                perturbed_embeddings = (original_embeddings + delta).detach()

        inputs['forget_inputs']['inputs_embeds'] = perturbed_embeddings
        inputs['forget_inputs'].pop('input_ids', None)

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
