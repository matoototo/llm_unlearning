import torch

from typing import Any, Dict, Tuple, List
from transformers import PreTrainedModel

def check_inputs(required_inputs: List[str], **kwargs):
    missing_inputs = [input_name for input_name in required_inputs if input_name not in kwargs]

    if missing_inputs:
        raise ValueError(f"Missing required input(s): {', '.join(missing_inputs)}")

class Method:
    def __init__(self, **kwargs):
        self.setup(**kwargs)
        self.input_keys = ["input_ids", "attention_mask", "labels"]

    def setup(self, **kwargs):
        pass

    def compute_loss(self, model: PreTrainedModel, **kwargs) -> Tuple[torch.Tensor, Dict[str, float], Any]:
        raise NotImplementedError("Subclasses must implement this method")

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

def get_method(method_name: str, **kwargs) -> Method:
    methods = {
        "gradient_ascent": GradientAscent,
        "gradient_descent": GradientDescent,
        "gradient_difference": GradientDifference,
    }

    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}")

    return methods[method_name](**kwargs)
