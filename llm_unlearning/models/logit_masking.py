import torch
import torch.nn as nn

from typing import Optional
from collections import defaultdict

from llm_unlearning.methods import EmbeddingRemapping

class LogitMaskingHook:
    def __init__(self, embedding_remapping: EmbeddingRemapping, masking_percentage: float):
        self.embedding_remapping = embedding_remapping
        self.masking_percentage = masking_percentage
        self.input_ids: Optional[torch.Tensor] = None
        self.layer_embeddings: Optional[torch.Tensor] = None
        self.group = "default"
        self.total_count = defaultdict(int)
        self.remapped_count = defaultdict(int)

    def store_layer_embeddings(self, module, input, output):
        current_output = output[0] if isinstance(output, tuple) else output
        if self.layer_embeddings is not None:
            self.layer_embeddings = torch.cat((self.layer_embeddings, current_output), dim=1)
        else:
            self.layer_embeddings = current_output

    def mask_logits(self, module, input, output):
        if self.input_ids is None or self.layer_embeddings is None:
            return output

        logits = output[0] if isinstance(output, tuple) else output
        batch_size, seq_length, vocab_size = logits.shape

        target_embeddings = self.embedding_remapping.get_target_token_embedding({'input_ids': self.input_ids}, self.layer_embeddings)

        if target_embeddings is None:
            return output

        for i in range(batch_size):
            self.total_count[self.group] += 1
            embedding = target_embeddings[i]
            if not any(self.embedding_remapping.is_inside_boundary(embedding, boundary) for boundary in self.embedding_remapping.boundaries): continue
            self.remapped_count[self.group] += 1
            # Mask top-N% of logits for all tokens in the sequence
            k = int(vocab_size * self.masking_percentage / 100)
            top_k_values, top_k_indices = torch.topk(logits[i], k, dim=-1)
            logits[i].scatter_(1, top_k_indices, float('-inf'))

        return logits if not isinstance(output, tuple) else (logits, *output[1:])

class LogitMaskingModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, embedding_remapping: EmbeddingRemapping, masking_percentage: float = 0.1):
        super().__init__()
        self.model = model
        self.device = model.device
        self.embedding_remapping = embedding_remapping
        self.hook = LogitMaskingHook(embedding_remapping, masking_percentage)
        self.group = None
        target_layer = self.model.model.layers[self.embedding_remapping.layer_id]
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
