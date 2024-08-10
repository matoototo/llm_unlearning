import torch
import torch.nn as nn

from llm_unlearning.methods import EmbeddingRemapping

class EmbeddingRemappingHook:
    def __init__(self, embedding_remapping: EmbeddingRemapping):
        self.embedding_remapping = embedding_remapping
        self.input_ids = None

    def remap_embeddings(self, module, input, output):
        if self.input_ids is None:
            return output
        output = output[0]

        target_embeddings = self.embedding_remapping.get_target_token_embedding({'input_ids': self.input_ids}, output)

        projected_embeddings = target_embeddings.clone()
        for i, embedding in enumerate(target_embeddings):
            for boundary in self.embedding_remapping.boundaries:
                if self.embedding_remapping.is_inside_boundary(embedding, boundary):
                    projected_embeddings[i] = self.embedding_remapping.project_embedding(embedding, boundary)
                    break

        output_modified = output.clone()
        batch_size = output.size(0)
        question_end = (self.input_ids == 33706).nonzero(as_tuple=True)[1]
        target_positions = question_end + 2
        output_modified[torch.arange(batch_size), target_positions] = projected_embeddings

        return (output_modified,)

class EmbeddingRemappingModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, embedding_remapping: EmbeddingRemapping):
        super().__init__()
        self.model = model
        self.embedding_remapping = embedding_remapping
        self.hook = EmbeddingRemappingHook(embedding_remapping)

        target_layer = self.model.model.layers[self.embedding_remapping.layer_id]
        target_layer.register_forward_hook(self.hook.remap_embeddings)

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get('input_ids')
        if input_ids is not None:
            self.hook.input_ids = input_ids
        return self.model(*args, **kwargs)

    # TODO: check if this plays nicely with remapping
    def generate(self, *args, **kwargs):
        input_ids = kwargs.get('input_ids')
        if input_ids is not None:
            self.hook.input_ids = input_ids
        return self.model.generate(*args, **kwargs)
