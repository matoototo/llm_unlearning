from typing import Dict, List, Any
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig

from .hp import HPDataset, get_hp_dataset
from .tofu import TofuDataset, get_tofu_dataset
from .wmdp import WMDPDataset, get_wmdp_dataset
from .wikitext import WikiTextDataset
from .ultrachat import UltraChatDataset

DATASET_REGISTRY = {
    "hp": HPDataset,
    "tofu": TofuDataset,
    "wmdp": WMDPDataset,
    "wikitext": WikiTextDataset,
    "ultrachat": UltraChatDataset
}

class CombinedDataset(Dataset):
    def __init__(self, datasets: Dict[str, Dataset]):
        self.datasets = datasets
        self.primary_dataset = next(iter(datasets.values()))  # Use the first dataset as primary

    def __len__(self):
        return len(self.primary_dataset)

    def __getitem__(self, idx):
        result = {}
        for name, dataset in self.datasets.items():
            if dataset == self.primary_dataset or name == 'dynamic':
                result[f"{name}_inputs"] = dataset[idx]
            else:
                random_idx = torch.randint(0, len(dataset), (1,)).item()
                result[f"{name}_inputs"] = dataset[random_idx]

                # Handle empty WikiText rows
                if isinstance(dataset, WikiTextDataset):
                    while (result[f"{name}_inputs"]['input_ids'] == 50256).all():
                        random_idx = torch.randint(0, len(dataset), (1,)).item()
                        result[f"{name}_inputs"] = dataset[random_idx]

        return result

    def set_epoch(self, epoch):
        for dataset in self.datasets.values():
            if hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(epoch)

    def set_model(self, model):
        for dataset in self.datasets.values():
            if hasattr(dataset, 'set_model') and getattr(dataset, 'use_dynamic_labels', False):
                dataset.set_model(model)

    def set_reference_model(self, reference_model):
        for dataset in self.datasets.values():
            if hasattr(dataset, 'set_reference_model'):
                dataset.set_reference_model(reference_model)

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        result = {}
        for key in batch[0].keys():
            result[key] = {
                inner_key: torch.stack([item[key][inner_key] for item in batch])
                for inner_key in batch[0][key].keys()
            }
        return result

def load_unlearning_dataset(dataset_cfg: DictConfig, tokenizer: PreTrainedTokenizer) -> Dataset:
    has_type = any("type" in config for config in dataset_cfg.values())

    if has_type:
        datasets = {}
        for dataset_name, config in dataset_cfg.items():
            if dataset_name == "name":
                continue
            dataset_type = config.get("type", dataset_cfg.name)
            if dataset_type not in DATASET_REGISTRY:
                raise ValueError(f"Invalid dataset type: {dataset_type}")
            datasets[dataset_name] = DATASET_REGISTRY[dataset_type](tokenizer, config, model=None)

        combined_dataset = CombinedDataset(datasets)
        return combined_dataset, combined_dataset.collate_fn
    else:
        # Old format
        if dataset_cfg.name == "tofu":
            dataset = get_tofu_dataset(tokenizer, dataset_cfg)
        elif dataset_cfg.name == "wmdp":
            dataset = get_wmdp_dataset(tokenizer, dataset_cfg)
        elif dataset_cfg.name == "hp":
            dataset = get_hp_dataset(tokenizer, dataset_cfg)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_cfg.name}")
        return dataset, dataset.collate_fn
