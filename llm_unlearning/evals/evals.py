import torch

from abc import ABC, abstractmethod
from typing import Dict, Any

from llm_unlearning.evals.utils import generate_and_extract, calculate_rouge_scores, RougeType

class Evaluation(ABC):
    @abstractmethod
    def compute(self, model, batch: Dict[str, Any], tokenizer=None, **kwargs) -> torch.Tensor:
        pass

class AggregateEvaluation(ABC):
    @abstractmethod
    def compute(self, retain_results: Dict[str, Any], checkpoint_results: Dict[str, Any]) -> torch.Tensor:
        pass

class AdversarialRouge(Evaluation):
    def __init__(self, max_length: int, rouge_type: RougeType = 'rougeL', num_samples: int = 10):
        super().__init__()
        self.max_length = max_length
        self.rouge_type = rouge_type
        self.num_samples = num_samples

    def compute(self, model, batch: Dict[str, Any], tokenizer=None, **kwargs) -> torch.Tensor:
        pad_token_id = tokenizer.pad_token_id
        device = model.device

        all_rouge_scores = []

        generation_kwargs = {
            'do_sample': True,
            'top_k': 50,
            'top_p': 0.95,
            'temperature': 1.0,
            'use_cache': True,
        }

        for _ in range(self.num_samples):
            question_texts, decoded_labels, decoded_outputs = generate_and_extract(
                model=model,
                batch=batch,
                tokenizer=tokenizer,
                max_length=self.max_length,
                device=device,
                generation_kwargs=generation_kwargs
            )

            rouge_scores = calculate_rouge_scores(decoded_outputs, decoded_labels, self.rouge_type)
            all_rouge_scores.append(rouge_scores)

        # Find the worst-case score for each example
        all_rouge_scores = torch.tensor(all_rouge_scores, device=device)
        worst_case_scores, _ = torch.max(all_rouge_scores, dim=0)

        return worst_case_scores
