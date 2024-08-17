import torch
import torch.nn as nn

from collections import defaultdict
from llm_unlearning.methods import EmbeddingBoundary

class EmbeddingRemappingHook:
    def __init__(self, embedding_boundary: EmbeddingBoundary):
        self.embedding_boundary = embedding_boundary
        self.input_ids = None
        self.group = "default"
        self.total_count = defaultdict(int)
        self.inside_boundary_count = defaultdict(int)

    def remap_embeddings(self, module, input, output):
        if self.input_ids is None:
            return output
        output, *rest_of_output = output

        target_embeddings = self.embedding_boundary.get_target_token_embedding({'input_ids': self.input_ids}, output)

        if target_embeddings is None: return (output, *rest_of_output)

        projected_embeddings = target_embeddings.clone()
        for i, embedding in enumerate(target_embeddings):
            self.total_count[self.group] += 1
            for boundary in self.embedding_boundary.boundaries:
                if not self.embedding_boundary.is_inside_boundary(embedding, boundary): continue
                projected_embeddings[i] = self.embedding_boundary.project_embedding(embedding, boundary)
                self.inside_boundary_count[self.group] += 1
                break

        output_modified = output.clone()
        batch_size = output.size(0)
        question_end = (self.input_ids == 33706).nonzero(as_tuple=True)[1]
        target_positions = question_end + 2
        output_modified[torch.arange(batch_size), target_positions] = projected_embeddings

        return (output_modified, *rest_of_output)

class EmbeddingRemappingModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, embedding_boundary: EmbeddingBoundary):
        super().__init__()
        self.model = model
        self.embedding_boundary = embedding_boundary
        self.hook = EmbeddingRemappingHook(embedding_boundary)
        self.group = None

        target_layer = self.model.model.layers[self.embedding_boundary.layer_id]
        target_layer.register_forward_hook(self.hook.remap_embeddings)

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get('input_ids')
        if input_ids is not None:
            self.hook.input_ids = input_ids
        self.hook.group = self.group
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        raise NotImplementedError
