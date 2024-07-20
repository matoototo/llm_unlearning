import torch
import einops
import numpy as np

from typing import Dict, Any
from scipy.stats import ks_2samp

from llm_unlearning.evals.evals import Evaluation, AggregateEvaluation
from llm_unlearning.evals.utils import (
    RougeType,
    probability,
    truth_ratio,
    rouge_score,
    harmonic_mean,
    extract_answer_tokens,
    calculate_rouge_scores,
    extract_question_tokens,
)

class KSTest(AggregateEvaluation):
    def compute(self, retain_results: Dict[str, Any], checkpoint_results: Dict[str, Any]) -> torch.tensor:
        retain_scores = retain_results["metrics"]["tofu_forget"]["truth_ratio_metadata"]
        forget_scores = checkpoint_results["metrics"]["tofu_forget"]["truth_ratio_metadata"]
        ks_statistic, ks_p_value = ks_2samp(retain_scores, forget_scores)
        return ks_p_value

class ModelUtility(AggregateEvaluation):
    def compute(self, _: Dict[str, Any], checkpoint_results: Dict[str, Any]) -> torch.tensor:
        metric_values = [checkpoint_results["metrics"][dataset][metric] for dataset in checkpoint_results["metrics"] for metric in checkpoint_results["metrics"][dataset] if not metric.endswith("_metadata") and metric != "truth_ratio"]

        truth_ratios = [np.array(checkpoint_results["metrics"][dataset]["truth_ratio_metadata"]) for dataset in checkpoint_results["metrics"]]
        metric_values.extend([np.mean(np.maximum(0, 1 - truth_ratios[i])) for i in range(len(truth_ratios))])
        return harmonic_mean(torch.tensor(metric_values))

class TruthRatio(Evaluation):
    def compute(self, model, batch: Dict[str, Any], tokenizer=None, **kwargs) -> torch.Tensor:
        gt_outputs = model(input_ids=batch["paraphrased_input_ids"], attention_mask=batch["paraphrased_attention_mask"])
        gt_logits = gt_outputs.logits
        gt_labels = batch["paraphrased_labels"]

        perturbed_logits = []
        perturbed_labels = []

        for i in range(len(batch["perturbed_input_ids"])):
            perturbed_output = model(
                input_ids=batch["perturbed_input_ids"][i],
                attention_mask=batch["perturbed_attention_mask"][i]
            )
            perturbed_logits.append(perturbed_output.logits)
            perturbed_labels.append(batch["perturbed_labels"][i])

        perturbed_logits = torch.stack(perturbed_logits, dim=1)
        perturbed_labels = torch.stack(perturbed_labels, dim=1)

        return truth_ratio(gt_logits, gt_labels, perturbed_logits, perturbed_labels)

class Probability(Evaluation):
    def compute(self, model, batch: Dict[str, Any], tokenizer=None, **kwargs) -> torch.Tensor:
        perturb_probability = kwargs.get("perturb_probability", False)
        input_keys = ["input_ids", "attention_mask", "labels"]
        outputs = model(**{k: v for k, v in batch.items() if k in input_keys})

        answer_probability = probability(outputs.logits, batch["labels"])
        if not perturb_probability: return answer_probability

        # World Facts and Real Authors look at the probability ratio between the answer and the multiple-choice examples
        perturbed_probabilities = []
        for i in range(len(batch["perturbed_input_ids"])):
            perturbed_output = model(
                input_ids=batch["perturbed_input_ids"][i],
                attention_mask=batch["perturbed_attention_mask"][i]
            )
            perturbed_probabilities.append(probability(perturbed_output.logits, batch["perturbed_labels"][i]))

        full_probability = torch.stack([answer_probability] + perturbed_probabilities, dim=0).sum(dim=0)
        return answer_probability / full_probability

class Rouge(Evaluation):
    def __init__(self, max_length: int, rouge_type: RougeType = 'rougeL'):
        self.max_length = max_length
        self.rouge_type = rouge_type

    def compute(self, model, batch: Dict[str, Any], tokenizer=None, **kwargs) -> torch.Tensor:
        pad_token_id = tokenizer.pad_token_id

        input_ids, attention_mask = extract_question_tokens(batch, pad_token_id)
        labels = batch["input_ids"]
        question_length = batch["question_length"]

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            do_sample=False,
        )

        extracted_outputs = extract_answer_tokens(outputs, question_length, pad_token_id)
        extracted_labels = extract_answer_tokens(labels, question_length, pad_token_id)

        decoded_outputs = tokenizer.batch_decode(extracted_outputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(extracted_labels, skip_special_tokens=True)

        # Strip away "Answer: " prefix
        decoded_outputs = [output[8:] for output in decoded_outputs]
        decoded_labels = [label[8:] for label in decoded_labels]

        rouge_score_value = rouge_score(decoded_outputs, decoded_labels, self.rouge_type)
        return torch.tensor(rouge_score_value, device=model.device)
