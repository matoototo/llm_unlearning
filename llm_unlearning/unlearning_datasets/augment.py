import os
import torch
import einops
import pickle
import hashlib

from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, Optional
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
        every_n_epochs: int = 4,
        cache_dir: Optional[str] = None,
    ):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_generated = num_generated
        self.every_n_epochs = every_n_epochs
        self.current_epoch = 0

        self.cache_dir = cache_dir
        if self.cache_dir: os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_id = self._generate_cache_id()
        self.cache_file = os.path.join(self.cache_dir, f"cache_{self.cache_id}.pkl") if self.cache_dir else None
        self.cache = self._load_cache()

    def _generate_cache_id(self):
        model_name = self.model.config._name_or_path
        dataset_split = self.dataset.forget_dataset.split
        unique_string = f"{model_name}_{dataset_split}_{self.max_length}_{self.num_generated}_{self.every_n_epochs}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def _load_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_cache(self):
        if not self.cache_file: return
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        forget_inputs = item['forget_inputs']
        pseudo_epoch = self.current_epoch // self.every_n_epochs

        if idx not in self.cache: self.cache[idx] = {}
        if pseudo_epoch not in self.cache[idx]:
            generated_outputs = self._generate_continuations(forget_inputs)
            self.cache[idx][pseudo_epoch] = generated_outputs
            self._save_cache()
        else:
            generated_outputs = self.cache[idx][pseudo_epoch]

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
