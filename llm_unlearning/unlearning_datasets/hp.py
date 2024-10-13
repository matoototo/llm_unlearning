import os
import torch
from transformers import PreTrainedTokenizer
from typing import Dict, List
from omegaconf import DictConfig
from torch.utils.data import Dataset

from llm_unlearning.unlearning_datasets.rawtext import RawTextDataset
from llm_unlearning.unlearning_datasets.wikitext import WikiTextDataset

class HPDataset(RawTextDataset):
    def _load_dataset(self):
        assert "file_path" in self.config and os.path.exists(self.config.file_path), "Config must contain valid file_path to the dataset file, you might need to download it first"

        with open(self.config.file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []

        for i in range(0, len(tokens), self.max_length):
            chunk_tokens = tokens[i:i + self.max_length]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append({"text": chunk_text})

        if self.full_context_mode:
            return self._create_full_context_items(chunks)

        if self.num_samples is not None:
            chunks = chunks[:self.num_samples]

        return chunks

def get_hp_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DictConfig
) -> Dataset:
    forget_config = config.forget
    retain_config = config.get("retain", None)
    dynamic_config = config.get("dynamic", None)
    retain_validation_config = config.get("retain_validation", None)

    forget_dataset = HPDataset(tokenizer, forget_config, model=None)
    retain_dataset = WikiTextDataset(tokenizer, retain_config) if retain_config else None
    dynamic_dataset = HPDataset(tokenizer, dynamic_config, model=None) if dynamic_config else None
    retain_validation_dataset = HPDataset(tokenizer, retain_validation_config, model=None) if retain_validation_config else None

    class CombinedDataset(Dataset):
        def __init__(self, forget_dataset, retain_dataset, dynamic_dataset, retain_validation_dataset):
            self.forget_dataset = forget_dataset
            self.retain_dataset = retain_dataset
            self.dynamic_dataset = dynamic_dataset
            self.retain_validation_dataset = retain_validation_dataset

        def __len__(self):
            return len(self.forget_dataset)

        def __getitem__(self, idx):
            result = {"forget_inputs": self.forget_dataset[idx]}

            if self.dynamic_dataset:
                result["dynamic_inputs"] = self.dynamic_dataset[idx]

            if self.retain_dataset:
                retain_idx = torch.randint(0, len(self.retain_dataset), (1,)).item()
                result["retain_inputs"] = self.retain_dataset[retain_idx]

                # some wikitext rows are empty, here we resample that case
                while (result["retain_inputs"]['input_ids'] == 50256).all():
                    retain_idx = torch.randint(0, len(self.retain_dataset), (1,)).item()
                    result["retain_inputs"] = self.retain_dataset[retain_idx]

            if self.retain_validation_dataset:
                retain_val_idx = torch.randint(0, len(self.retain_validation_dataset), (1,)).item()
                result["retain_validation_inputs"] = self.retain_validation_dataset[retain_val_idx]

            return result

        def set_epoch(self, epoch):
            for dataset in [self.forget_dataset, self.retain_dataset, self.dynamic_dataset, self.retain_validation_dataset]:
                if hasattr(dataset, 'set_epoch'):
                    dataset.set_epoch(epoch)

        def set_model(self, model):
            for dataset in [self.forget_dataset, self.dynamic_dataset]:
                if hasattr(dataset, 'set_model') and dataset.use_dynamic_labels:
                    dataset.set_model(model)

        @staticmethod
        def collate_fn(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
            result = {}
            for key in ['forget_inputs', 'dynamic_inputs', 'retain_inputs', 'retain_validation_inputs']:
                if key not in batch[0]:
                    continue
                collected_batch = [item[key] for item in batch]
                if key == 'retain_inputs':
                    result[key] = WikiTextDataset.collate_fn(collected_batch)
                else:
                    result[key] = HPDataset.collate_fn(collected_batch)
            return result

    return CombinedDataset(forget_dataset, retain_dataset, dynamic_dataset, retain_validation_dataset)
