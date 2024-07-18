import torch

from abc import ABC, abstractmethod
from typing import Dict, Any

class Evaluation(ABC):
    @abstractmethod
    def compute(self, model, batch: Dict[str, Any], tokenizer=None, **kwargs) -> torch.Tensor:
        pass

class AggregateEvaluation(ABC):
    @abstractmethod
    def compute(self, retain_results: Dict[str, Any], checkpoint_results: Dict[str, Any]) -> torch.tensor:
        pass
