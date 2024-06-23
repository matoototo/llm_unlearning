import torch

from typing import Any, Tuple, List
from transformers import PreTrainedModel

def check_inputs(required_inputs: List[str], **kwargs):
    missing_inputs = [input_name for input_name in required_inputs if input_name not in kwargs]

    if missing_inputs:
        raise ValueError(f"Missing required input(s): {', '.join(missing_inputs)}")

class UnlearningMethod:
    def __init__(self, **kwargs):
        self.setup(**kwargs)

    def setup(self, **kwargs):
        """Override this method to set up method-specific parameters"""
        pass

    def compute_loss(self, model: PreTrainedModel, **kwargs) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("Subclasses must implement this method")

class GradientAscent(UnlearningMethod):
    def compute_loss(self, model: PreTrainedModel, **kwargs) -> Tuple[torch.Tensor, Any]:
        check_inputs(["forget_inputs"], **kwargs)

        forget_inputs = kwargs['forget_inputs']
        outputs = model(**forget_inputs)
        forget_loss = outputs.loss * -1
        return forget_loss, outputs

class GradientDifference(UnlearningMethod):
    def compute_loss(self, model: PreTrainedModel, **kwargs) -> Tuple[torch.Tensor, Any]:
        check_inputs(["forget_inputs", "retain_inputs"], **kwargs)

        forget_inputs = kwargs['forget_inputs']
        retain_inputs = kwargs['retain_inputs']

        forget_outputs = model(**forget_inputs)
        forget_loss = forget_outputs.loss * -1

        retain_outputs = model(**retain_inputs)
        retain_loss = retain_outputs.loss

        loss = forget_loss + retain_loss
        return loss, (forget_outputs, retain_outputs)

def get_unlearning_method(method_name: str, **kwargs) -> UnlearningMethod:
    methods = {
        "gradient_ascent": GradientAscent,
        "gradient_difference": GradientDifference,
    }

    if method_name not in methods:
        raise ValueError(f"Unknown unlearning method: {method_name}")

    return methods[method_name](**kwargs)
