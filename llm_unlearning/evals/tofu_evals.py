import torch
import torch.nn.functional as F
import evaluate

from typing import List

def probability(logits: torch.Tensor, labels: torch.Tensor):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    probs = F.softmax(shift_logits, dim=-1)
    correct_probs = torch.gather(probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    sequence_probs = torch.prod(correct_probs, dim=-1)

    sequence_lengths = (shift_labels != -100).sum(dim=-1).float()
    length_normalized_probs = sequence_probs ** (1 / sequence_lengths)

    return length_normalized_probs

def truth_ratio(gt_logits: torch.Tensor, gt_labels: torch.Tensor, perturb_logits: torch.Tensor, perturb_labels: torch.Tensor):
    gt_loss = -torch.log(probability(gt_logits, gt_labels))
    perturb_loss = -torch.log(probability(perturb_logits, perturb_labels))

    perturb_probability = torch.exp(-perturb_loss).mean(-1)
    gt_probability = torch.exp(-gt_loss)
    truth_ratio = torch.max(torch.zeros_like(gt_probability), 1 - perturb_probability / gt_probability)
    return truth_ratio

def rouge(predictions: List[str], references: List[str]):
    rouge = evaluate.load('rouge')
    return rouge.compute(predictions=predictions, references=references)
