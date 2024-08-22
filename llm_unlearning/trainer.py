from transformers import Trainer
from typing import Callable, List, Dict, Optional
from llm_unlearning.methods import get_method, get_attack, EmbeddingBoundary

class UnlearningTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        method = kwargs.pop("method")
        at_attack = kwargs.pop("adversarial_attack", None)
        unlearning_kwargs = kwargs.pop("unlearning_kwargs", {})
        attack_kwargs = kwargs.pop("attack_kwargs", {})
        self.reference_model = kwargs.pop("reference_model", None)
        super().__init__(*args, **kwargs)
        self.loss_components: Dict[str, float] = {}
        self.loss_component_counts: Dict[str, int] = {}
        self.is_grad_accumulation = self.args.gradient_accumulation_steps > 1
        self.method = get_method(method, **unlearning_kwargs)
        self.attack = get_attack(at_attack, **attack_kwargs) if at_attack else None

        if self.reference_model:
            self.reference_model.to(self.args.device)
            self.reference_model.eval()

        self.prehooks: List[Callable] = []
        self.posthooks: List[Callable] = []

        self.prehooks.append(self.method.prehook)
        self.posthooks.append(self.method.posthook)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.attack:
            inputs = self.attack.attack(model, inputs, self.is_grad_accumulation)

        inputs['step_ratio'] = min(1.0, self.state.global_step / self.state.max_steps)

        loss, loss_dict, outputs = self.method.compute_loss(model, **inputs, reference_model=self.reference_model)

        for loss_name, loss_value in loss_dict.items():
            self.accumulate_loss(loss_name, loss_value)

        return (loss, outputs) if return_outputs else loss

    def accumulate_loss(self, loss_name: str, loss_value: float):
        if loss_name not in self.loss_components:
            self.loss_components[loss_name] = 0.0
            self.loss_component_counts[loss_name] = 0
        self.loss_components[loss_name] += loss_value
        self.loss_component_counts[loss_name] += 1

    def log(self, logs: Dict[str, float]) -> Dict[str, float]:
        for loss_name, loss_sum in self.loss_components.items():
            count = self.loss_component_counts[loss_name]
            if count > 0:
                logs[f"{loss_name}"] = round(loss_sum / count, 4)

        self.loss_components = {}
        self.loss_component_counts = {}

        return super().log(logs)

    def training_step(self, model, inputs):
        for prehook in self.prehooks:
            prehook(self, model, inputs)

        if hasattr(self.train_dataset, 'set_epoch'):
            self.train_dataset.set_model(model)
            self.train_dataset.set_epoch(self.state.epoch)

        out = super().training_step(model, inputs)

        for posthook in self.posthooks:
            posthook(self, model, inputs, out)

        return out

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)
        if isinstance(self.method, EmbeddingBoundary):
            self.method.save_boundaries(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint):
        super()._load_from_checkpoint(resume_from_checkpoint)
        if isinstance(self.method, EmbeddingBoundary):
            self.method.boundaries = EmbeddingBoundary.load_boundaries(resume_from_checkpoint)
