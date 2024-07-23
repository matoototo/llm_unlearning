import torch

from abc import ABC, abstractmethod
from typing import Dict, Any

from llm_unlearning.evals.utils import extract_question_tokens, extract_answer_tokens, calculate_rouge_scores, RougeType

class Evaluation(ABC):
    @abstractmethod
    def compute(self, model, batch: Dict[str, Any], tokenizer=None, **kwargs) -> torch.Tensor:
        pass

class AggregateEvaluation(ABC):
    @abstractmethod
    def compute(self, retain_results: Dict[str, Any], checkpoint_results: Dict[str, Any]) -> torch.tensor:
        pass

class AdversarialRouge(Evaluation):
    def __init__(self, max_length: int, rouge_type: RougeType = 'rougeL', num_samples: int = 10):
        super().__init__()
        self.max_length = max_length
        self.rouge_type = rouge_type
        self.num_samples = num_samples

    def compute(self, model, batch: Dict[str, Any], tokenizer=None, **kwargs) -> torch.Tensor:
        pad_token_id = tokenizer.pad_token_id

        input_ids, attention_mask = extract_question_tokens(batch, pad_token_id)
        labels = batch["input_ids"]
        question_length = batch["question_length"]

        all_rouge_scores = []

        for _ in range(self.num_samples):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
            )

            extracted_outputs = extract_answer_tokens(outputs, question_length, pad_token_id)
            extracted_labels = extract_answer_tokens(labels, question_length, pad_token_id)

            decoded_outputs = tokenizer.batch_decode(extracted_outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(extracted_labels, skip_special_tokens=True)

            # Strip away "Answer: " prefix
            decoded_outputs = [output[8:] for output in decoded_outputs]
            decoded_labels = [label[8:] for label in decoded_labels]
            rouge_scores = calculate_rouge_scores(decoded_outputs, decoded_labels, self.rouge_type)

            all_rouge_scores.append(rouge_scores)

        # Find the worst-case score for each example
        all_rouge_scores = torch.tensor(all_rouge_scores, device=model.device)
        worst_case_scores, _ = torch.max(all_rouge_scores, dim=0)

        return worst_case_scores
