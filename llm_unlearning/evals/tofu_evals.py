import torch
import torch.nn.functional as F
import einops
import numpy as np

from rouge_score import rouge_scorer, tokenizers
from scipy.stats import ks_2samp
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Literal

def sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute negative log-likelihood for given logits and labels.

    Args:
        logits (batch_size, seq_len, vocab_size)
        labels (batch_size, seq_len)

    Returns:
        Negative log-likelihood (batch_size,), summed over sequence length
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none', ignore_index=-100)
    loss = einops.rearrange(loss, '(b s) -> b s', b=shift_labels.size(0))

    return loss.sum(dim=-1)

def probability(logits: torch.Tensor, labels: torch.Tensor, logprobs: bool = False) -> torch.Tensor:
    """
    Compute length-normalized probabilities for given logits and labels.

    Args:
        logits (batch_size, seq_len, vocab_size)
        labels (batch_size, seq_len)

    Returns:
        Length-normalized probabilities (batch_size,)
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none', ignore_index=-100)
    loss = einops.rearrange(loss, '(b s) -> b s', b=shift_labels.size(0))

    valid_mask = (shift_labels != -100)

    sequence_lengths = valid_mask.sum(dim=-1).float()
    sequence_losses = loss.sum(dim=-1)
    length_normalized_losses = sequence_losses / sequence_lengths

    if logprobs: return -length_normalized_losses
    return torch.exp(-length_normalized_losses)

def geometric_mean(values: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.mean(torch.log(values), dim=-1))

def harmonic_mean(values: torch.Tensor) -> torch.Tensor:
    return values.size(0) / torch.sum(1.0 / values)

def truth_ratio(
    gt_logits: torch.Tensor,
    gt_labels: torch.Tensor,
    perturb_logits: torch.Tensor,
    perturb_labels: torch.Tensor,
    tofu_code_equivalent: bool = True
) -> torch.Tensor:
    """
    Compute the truth ratio for given ground truth and perturbed logits and labels.

    Args:
        gt_logits (batch_size, seq_len, vocab_size)
        gt_labels (batch_size, seq_len)
        perturb_logits (batch_size, perturbations, seq_len, vocab_size)
        perturb_labels (batch_size, perturbations, seq_len)
        tofu_code_equivalent (bool): if set, uses geometric mean (=codebase) instead of arithmetic mean (=paper).

    Returns:
        Truth ratio (batch_size,)
    """
    b, p, s, v = perturb_logits.shape
    gt_logprobs = probability(gt_logits, gt_labels, logprobs=True)
    batched_perturb_logits = einops.rearrange(perturb_logits, 'b p s v -> (b p) s v')
    batched_perturb_labels = einops.rearrange(perturb_labels, 'b p s -> (b p) s')
    batched_perturb_logprobs = probability(batched_perturb_logits, batched_perturb_labels, logprobs=True)
    perturb_logprobs = einops.rearrange(batched_perturb_logprobs, '(b p) -> b p', p=p)

    if tofu_code_equivalent: # -> geomean of probs
        mean_perturb_logprobs = torch.mean(perturb_logprobs, dim=-1)
    else: # -> mean of probs
        mean_perturb_logprobs = torch.logsumexp(perturb_logprobs, dim=-1) - torch.log(torch.tensor(p))

    log_truth_ratio = mean_perturb_logprobs - gt_logprobs
    return torch.exp(log_truth_ratio)

RougeType = Literal['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
def rouge_score(predictions: List[str], references: List[str], rouge_type: RougeType = 'rougeL') -> List[float]:
    rouge = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True, tokenizer=tokenizers.DefaultTokenizer(True))
    recall = sum(rouge.score(ref, pred)[rouge_type].recall for ref, pred in zip(references, predictions))
    return recall / len(predictions)

class Evaluation(ABC):
    @abstractmethod
    def compute(self, model, batch: Dict[str, Any], tokenizer=None, **kwargs) -> torch.Tensor:
        pass

class AggregateEvaluation(ABC):
    @abstractmethod
    def compute(self, retain_results: Dict[str, Any], checkpoint_results: Dict[str, Any]) -> torch.tensor:
        pass

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

        def extract_question_tokens(batch):
            question_length = batch["question_length"]  # (batch_size,)
            input_ids = batch["input_ids"]  # (batch_size, seq_len)
            attention_mask = batch["attention_mask"]  # (batch_size, seq_len)
            batch_size, seq_len = input_ids.shape

            # Extract question tokens (right-padded)
            mask = einops.repeat(torch.arange(seq_len, device=question_length.device), 's -> b s', b=batch_size) < question_length[:, None]
            max_question_length = question_length.max().item()
            extracted_input_ids = torch.where(mask[:, :max_question_length], input_ids[:, :max_question_length], pad_token_id)
            extracted_attention_mask = torch.where(mask[:, :max_question_length], attention_mask[:, :max_question_length], 0)

            # Rotate to convert right-padded to left-padded
            rotation_amounts = max_question_length - question_length
            rotated_input_ids = torch.stack([torch.roll(seq, shift.item()) for seq, shift in zip(extracted_input_ids, rotation_amounts)])
            rotated_attention_mask = torch.stack([torch.roll(seq, shift.item()) for seq, shift in zip(extracted_attention_mask, rotation_amounts)])

            return rotated_input_ids, rotated_attention_mask

        def extract_answer_tokens(tokens, question_length, pad_token_id):
            batch_size, seq_len = tokens.shape
            mask = tokens != pad_token_id

            max_answer_length = seq_len - question_length.min().item()
            extracted_answers = torch.full((batch_size, max_answer_length), pad_token_id, device=tokens.device)

            for i in range(batch_size):
                answer = tokens[i, mask[i]][question_length[i]:]
                extracted_answers[i, :len(answer)] = answer

            return extracted_answers

        input_ids, attention_mask = extract_question_tokens(batch)
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

        # Returns average but that's fine, we average later anyway
        rouge_score_value = rouge_score(decoded_outputs, decoded_labels, self.rouge_type)
        return torch.tensor(rouge_score_value, device=model.device).unsqueeze(0)

all_metrics = {
    "truth_ratio": lambda config: TruthRatio(),
    "probability": lambda config: Probability(),

    "rouge_1": lambda config: Rouge(max_length=config.max_length, rouge_type='rouge1'),
    "rouge_2": lambda config: Rouge(max_length=config.max_length, rouge_type='rouge2'),
    "rouge_l": lambda config: Rouge(max_length=config.max_length, rouge_type='rougeL'),
    "rouge_lsum": lambda config: Rouge(max_length=config.max_length, rouge_type='rougeLsum'),
}

all_aggregate_metrics = {
    "ks_test": lambda config: KSTest(),
    "model_utility": lambda config: ModelUtility()
}
