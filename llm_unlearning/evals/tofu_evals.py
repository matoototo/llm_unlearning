# tofu_evals.py
import torch
import torch.nn.functional as F
import evaluate
import einops
from typing import List, Dict, Any
from abc import ABC, abstractmethod

def probability(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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

    length_normalized_probs = torch.exp(-length_normalized_losses)

    return length_normalized_probs

def truth_ratio(
    gt_logits: torch.Tensor,
    gt_labels: torch.Tensor,
    perturb_logits: torch.Tensor,
    perturb_labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute the truth ratio for given ground truth and perturbed logits and labels.

    Args:
        gt_logits (batch_size, seq_len, vocab_size)
        gt_labels (batch_size, seq_len)
        perturb_logits (batch_size, perturbations, seq_len, vocab_size)
        perturb_labels (batch_size, perturbations, seq_len)

    Returns:
        Truth ratio (batch_size,)
    """
    b, p, s, v = perturb_logits.shape
    gt_probs = probability(gt_logits, gt_labels)

    batched_perturb_logits = einops.rearrange(perturb_logits, 'b p s v -> (b p) s v')
    batched_perturb_labels = einops.rearrange(perturb_labels, 'b p s -> (b p) s')
    batched_perturb_probs = probability(batched_perturb_logits, batched_perturb_labels)

    perturb_probs = einops.rearrange(batched_perturb_probs, '(b p) -> b p', p=p)

    R_truth = torch.mean(perturb_probs, dim=-1) / gt_probs

    return R_truth

def rouge_l(predictions: List[str], references: List[str]) -> List[float]:
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    return results['rougeL']

class Evaluation(ABC):
    @abstractmethod
    def compute(self, model, batch: Dict[str, Any], tokenizer=None, **kwargs) -> torch.Tensor:
        pass

class TruthRatio(Evaluation):
    def compute(self, model, batch: Dict[str, Any], tokenizer=None, **kwargs) -> torch.Tensor:
        input_keys = ["input_ids", "attention_mask", "labels"]
        gt_outputs = model(**{k: v for k, v in batch.items() if k in input_keys})
        gt_logits = gt_outputs.logits
        gt_labels = batch["labels"]

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

class RougeL(Evaluation):
    def __init__(self, max_length: int):
        self.max_length = max_length

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

        input_ids, attention_mask = extract_question_tokens(batch)
        labels = batch["input_ids"]

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Returns average but that's fine, we average later anyway
        rouge_l_score = rouge_l(decoded_outputs, decoded_labels)
        return torch.tensor(rouge_l_score, device=model.device).unsqueeze(0)

all_metrics = {
    "truth_ratio": lambda config: TruthRatio(),
    "probability": lambda config: Probability(),
    "rouge_l": lambda config: RougeL(config.max_length)
}
