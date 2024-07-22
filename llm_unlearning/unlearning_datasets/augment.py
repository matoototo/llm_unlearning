import torch
import einops

from torch.utils.data import Dataset
from typing import Dict, Any, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

from llm_unlearning.evals.utils import extract_answer_tokens, extract_question_tokens

class AugmentGenerated(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 200,
        num_generated: int = 4,
        every_n_epochs: int = 4
    ):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_generated = num_generated
        self.every_n_epochs = every_n_epochs
        self.current_epoch = 0
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        forget_inputs = item['forget_inputs']

        if idx not in self.cache or self.current_epoch % self.every_n_epochs == 0:
            generated_outputs = self._generate_continuations(forget_inputs)
            self.cache[idx] = generated_outputs
        else:
            generated_outputs = self.cache[idx]

        padded_outputs, padding_masks = self._pad_outputs(generated_outputs)

        augmented_item = {
            'forget_inputs': {
                **forget_inputs,
                'generated_outputs': padded_outputs,
                'generated_masks': padding_masks,
            }
        }

        if 'retain_inputs' in item:
            augmented_item['retain_inputs'] = item['retain_inputs']

        return augmented_item

    def _generate_continuations(self, item: Dict[str, torch.Tensor]) -> torch.Tensor:
        pad_token_id = self.tokenizer.pad_token_id

        # extract_question_tokens expects a batched item
        batched_item = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in item.items()}

        input_ids, attention_mask = extract_question_tokens(batched_item, pad_token_id)
        question_length = einops.repeat(batched_item["question_length"], 'b -> (b n)', n=self.num_generated)

        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
                num_return_sequences=self.num_generated,
                pad_token_id=pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
            )

        extracted_outputs = extract_answer_tokens(outputs, question_length, pad_token_id)
        return extracted_outputs.to(item['input_ids'].device)

    def _pad_outputs(self, outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        padded_outputs = torch.full((self.num_generated, self.max_length), pad_token_id, device=outputs.device)
        padded_outputs[:, :outputs.shape[1]] = outputs
        padding_masks = (padded_outputs != pad_token_id).long()
        return padded_outputs, padding_masks

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
