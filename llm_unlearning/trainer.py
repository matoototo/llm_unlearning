from transformers import Trainer
from llm_unlearning.methods import get_method
from typing import Dict

class UnlearningTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        method = kwargs.pop("method")
        unlearning_kwargs = kwargs.pop("unlearning_kwargs", {})
        super().__init__(*args, **kwargs)
        self.method = get_method(method, **unlearning_kwargs)
        self.loss_components: Dict[str, float] = {}
        self.loss_component_counts: Dict[str, int] = {}

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, loss_dict, outputs = self.method.compute_loss(model, **inputs)

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
