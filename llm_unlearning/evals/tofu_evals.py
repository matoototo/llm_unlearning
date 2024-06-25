# tofu_evals.py
import torch
import torch.nn.functional as F
import evaluate
import einops
from typing import List

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

    probs = F.softmax(shift_logits, dim=-1)

    valid_mask = (shift_labels != -100)
    labels_one_hot = F.one_hot(shift_labels.clamp(min=0), num_classes=probs.size(-1))

    correct_probs = torch.einsum('bsv,bsv->bs', probs, labels_one_hot.float())
    correct_probs = correct_probs.masked_fill(~valid_mask, 1.0)

    sequence_probs = torch.prod(correct_probs, dim=-1)
    sequence_lengths = valid_mask.sum(dim=-1).float()
    length_normalized_probs = sequence_probs ** (1 / sequence_lengths)

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

def rouge(predictions: List[str], references: List[str]) -> dict:
    rouge = evaluate.load('rouge')
    return rouge.compute(predictions=predictions, references=references)
