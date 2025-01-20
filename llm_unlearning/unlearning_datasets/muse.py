import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List
from omegaconf import DictConfig
import datasets

from llm_unlearning.unlearning_datasets.rawtext import RawTextDataset

class MuseDataset(RawTextDataset):
    def _load_dataset(self):
        subset = self.config.get("subset", "raw")
        split = self.config.get("split", None)

        if not split:
            raise ValueError("Split must be specified in config")

        download_config = datasets.DownloadConfig(force_download=True)
        texts = []

        if split == "*" or split == "retrain":
            dataset = datasets.load_dataset(
                "muse-bench/MUSE-News",
                subset,
                download_config=download_config
            )
            available_splits = [x for x in list(dataset.keys()) if x != "holdout"]
            if split == "retrain": available_splits = [x for x in available_splits if "forget" not in x]

            if not available_splits: raise ValueError(f"No splits found for subset '{subset}'")

            for current_split in available_splits:
                texts.extend(self._process_split(dataset[current_split], subset))
        else:
            dataset = datasets.load_dataset(
                "muse-bench/MUSE-News",
                subset,
                split=split,
                download_config=download_config
            )
            texts.extend(self._process_split(dataset, subset))

        if self.num_samples is not None:
            texts = texts[:self.num_samples]

        full_text = " ".join(texts)
        return self.tokenizer.encode(full_text, add_special_tokens=False)

    def _process_split(self, dataset, subset):
         return [item["text"] for item in dataset]

def get_muse_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DictConfig
) -> Dataset:
    forget_config = config.forget
    retain_config = config.get("retain", None)
    dynamic_config = config.get("dynamic", None)
    retain_validation_config = config.get("retain_validation", None)

    forget_dataset = MuseDataset(tokenizer, forget_config)
    retain_dataset = MuseDataset(tokenizer, retain_config) if retain_config else None
    dynamic_dataset = MuseDataset(tokenizer, dynamic_config, model=None) if dynamic_config else None
    retain_validation_dataset = MuseDataset(tokenizer, retain_validation_config) if retain_validation_config else None

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

            if self.retain_validation_dataset:
                retain_val_idx = torch.randint(0, len(self.retain_validation_dataset), (1,)).item()
                result["retain_validation_inputs"] = self.retain_validation_dataset[retain_val_idx]

            return result

        def set_epoch(self, epoch):
            for dataset in [self.forget_dataset, self.retain_dataset, self.dynamic_dataset, self.retain_validation_dataset]:
                if dataset and hasattr(dataset, 'set_epoch'):
                    dataset.set_epoch(epoch)

        def set_model(self, model):
            for dataset in [self.forget_dataset, self.dynamic_dataset]:
                if dataset and hasattr(dataset, 'set_model') and dataset.use_dynamic_labels:
                    dataset.set_model(model)

        @staticmethod
        def collate_fn(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
            result = {}
            for key in ['forget_inputs', 'dynamic_inputs', 'retain_inputs', 'retain_validation_inputs']:
                if key not in batch[0]:
                    continue
                collected_batch = [item[key] for item in batch]
                result[key] = MuseDataset.collate_fn(collected_batch)
            return result

    return CombinedDataset(forget_dataset, retain_dataset, dynamic_dataset, retain_validation_dataset)
