import torch
import torch.nn.functional as F
import einops

from typing import Dict, Any, List, Literal
from rouge_score import rouge_scorer, tokenizers

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

    length_normalized_losses.clamp_(min=-1e5, max=1e5)

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
def rouge_score(predictions: List[str], references: List[str], rouge_type: RougeType = 'rougeL') -> float:
    rouge = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True, tokenizer=tokenizers.DefaultTokenizer(True))
    recall = [rouge.score(ref, pred)[rouge_type].recall for ref, pred in zip(references, predictions)]
    return recall

def extract_question_tokens(batch: Dict[str, torch.Tensor], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract question tokens from input batch and rotate to left-pad for faster generation.

    Args:
        batch["question_length"] (batch_size,)
        batch["input_ids"] (batch_size, seq_len)
        batch["attention_mask"] (batch_size, seq_len)
        pad_token_id (int): Token ID used for padding.

    Returns:
        rotated_input_ids (batch_size, max_question_length): Left-padded question input IDs.
        rotated_attention_mask (batch_size, max_question_length): Corresponding attention mask.
    """
    question_length = batch["question_length"]  # (batch_size,)
    input_ids = batch["input_ids"]  # (batch_size, seq_len)
    attention_mask = batch["attention_mask"]  # (batch_size, seq_len)
    batch_size, seq_len = input_ids.shape

    # Extract question tokens (right-padded)
    mask = torch.arange(seq_len, device=question_length.device).unsqueeze(0) < question_length.unsqueeze(1)
    max_question_length = question_length.max().item()
    extracted_input_ids = torch.where(mask[:, :max_question_length], input_ids[:, :max_question_length], pad_token_id)
    extracted_attention_mask = torch.where(mask[:, :max_question_length], attention_mask[:, :max_question_length], 0)

    # Rotate to convert right-padded to left-padded
    rotation_amounts = max_question_length - question_length
    rotated_input_ids = torch.stack([torch.roll(seq, shift.item()) for seq, shift in zip(extracted_input_ids, rotation_amounts)])
    rotated_attention_mask = torch.stack([torch.roll(seq, shift.item()) for seq, shift in zip(extracted_attention_mask, rotation_amounts)])

    return rotated_input_ids, rotated_attention_mask

def extract_answer_tokens(tokens: torch.Tensor, question_length: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Extract answer tokens from generated outputs or labels.

    Args:
        tokens (batch_size, seq_len)
        question_length (batch_size,)
        pad_token_id (int)
    Returns:
        torch.Tensor: Extracted answer tokens of shape (batch_size, max_answer_length), right-padded.
    """
    batch_size, seq_len = tokens.shape
    mask = tokens != pad_token_id

    max_answer_length = seq_len - question_length.min().item()
    extracted_answers = torch.full((batch_size, max_answer_length), pad_token_id, device=tokens.device)

    for i in range(batch_size):
        answer = tokens[i, mask[i]][question_length[i]:]
        extracted_answers[i, :len(answer)] = answer

    return extracted_answers

def calculate_rouge_scores(decoded_outputs: List[str], decoded_labels: List[str], rouge_type: RougeType) -> List[float]:
    return [
        rouge_score([output], [label], rouge_type)
        for output, label in zip(decoded_outputs, decoded_labels)
    ]
