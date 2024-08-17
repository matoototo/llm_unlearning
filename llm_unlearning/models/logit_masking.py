import torch
import torch.nn as nn
from typing import Optional
from collections import defaultdict

from llm_unlearning.methods import EmbeddingBoundary

def top_k_masking(logits, masking_percentage, **kwargs):
    vocab_size = logits.shape[-1]
    k = int(vocab_size * masking_percentage / 100)
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    logits.scatter_(-1, top_k_indices, float('-inf'))
    return logits

def top_k_mean_masking(logits, masking_percentage, **kwargs):
    vocab_size = logits.shape[-1]
    k = int(vocab_size * masking_percentage / 100)
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    mean_value = torch.mean(logits, dim=-1, keepdim=True)
    logits.scatter_(-1, top_k_indices, mean_value.expand_as(top_k_indices))
    return logits

def top_k_subtract_mean(logits, masking_percentage, mean_factor = 1.0, **kwargs):
    vocab_size = logits.shape[-1]
    k = int(vocab_size * masking_percentage / 100)
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    mean_value = torch.mean(logits, dim=-1, keepdim=True)
    top_k_adjusted = top_k_values - mean_factor * mean_value
    logits.scatter_(-1, top_k_indices, top_k_adjusted)
    return logits

def gaussian_noise(logits, sigma_scale=1.0, **kwargs):
    max_val, _ = torch.max(logits, dim=-1, keepdim=True)
    min_val, _ = torch.min(logits, dim=-1, keepdim=True)
    sigma = (max_val - min_val) * sigma_scale
    noise = torch.randn_like(logits) * sigma
    return logits + noise

STRATEGIES = {
    "top_k_masking": top_k_masking,
    "top_k_mean_masking": top_k_mean_masking,
    "top_k_subtract_mean": top_k_subtract_mean,
    "gaussian_noise": gaussian_noise,
}

class LogitMaskingHook:
    def __init__(self, embedding_boundary: EmbeddingBoundary, strategy: str, **kwargs):
        self.embedding_boundary = embedding_boundary
        self.strategy = STRATEGIES[strategy]
        self.strategy_kwargs = kwargs
        self.input_ids: Optional[torch.Tensor] = None
        self.layer_embeddings: Optional[torch.Tensor] = None
        self.group = "default"
        self.total_count = defaultdict(int)
        self.inside_boundary_count = defaultdict(int)

    def store_layer_embeddings(self, module, input, output):
        current_output = output[0] if isinstance(output, tuple) else output
        if self.layer_embeddings is not None:
            self.layer_embeddings = torch.cat((self.layer_embeddings, current_output), dim=1)
        else:
            self.layer_embeddings = current_output

    def modify_logits(self, logits):
        return self.strategy(logits, **self.strategy_kwargs)

    def mask_logits(self, module, input, output):
        if self.input_ids is None or self.layer_embeddings is None:
            return output

        logits = output[0] if isinstance(output, tuple) else output
        batch_size, seq_length, vocab_size = logits.shape

        target_embeddings = self.embedding_boundary.get_target_token_embedding({'input_ids': self.input_ids}, self.layer_embeddings)

        if target_embeddings is None:
            return output

        for i in range(batch_size):
            self.total_count[self.group] += 1
            embedding = target_embeddings[i]
            if not any(self.embedding_boundary.is_inside_boundary(embedding, boundary) for boundary in self.embedding_boundary.boundaries):
                continue
            self.inside_boundary_count[self.group] += 1
            logits[i] = self.modify_logits(logits[i])

        return logits if not isinstance(output, tuple) else (logits, *output[1:])

class LogitMaskingModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, embedding_boundary: EmbeddingBoundary, strategy: str = "top_k_masking", **kwargs):
        super().__init__()
        self.model = model
        self.device = model.device
        self.embedding_boundary = embedding_boundary
        self.hook = LogitMaskingHook(embedding_boundary, strategy, **kwargs)
        self.group = None
        target_layer = self.model.model.layers[self.embedding_boundary.layer_id]
        target_layer.register_forward_hook(self.hook.store_layer_embeddings)
        self.model.lm_head.register_forward_hook(self.hook.mask_logits)

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get('input_ids')
        self.hook.layer_embeddings = None
        if input_ids is not None:
            self.hook.input_ids = input_ids
        self.hook.group = self.group
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        input_ids = kwargs.get('input_ids')
        self.hook.layer_embeddings = None
        if input_ids is not None:
            self.hook.input_ids = input_ids
        self.hook.group = self.group
        return self.model.generate(*args, **kwargs)
