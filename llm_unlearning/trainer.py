from transformers import Trainer
from llm_unlearning.methods import get_unlearning_method

class UnlearningTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        unlearning_method = kwargs.pop("unlearning_method")
        unlearning_kwargs = kwargs.pop("unlearning_kwargs", {})
        super().__init__(*args, **kwargs)
        self.unlearning_method = get_unlearning_method(unlearning_method, **unlearning_kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = self.unlearning_method.compute_loss(
            model, **inputs
        )

        return (loss, outputs) if return_outputs else loss
