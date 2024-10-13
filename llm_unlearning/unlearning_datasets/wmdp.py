import torch
import datasets
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Dict, List, Optional
from omegaconf import DictConfig
from torch.utils.data import Dataset

from llm_unlearning.unlearning_datasets.rawtext import RawTextDataset
from llm_unlearning.unlearning_datasets.wikitext import WikiTextDataset

class WMDPDataset(RawTextDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        model: Optional[PreTrainedModel] = None,
        full_context_mode: bool = False,
        num_samples: Optional[int] = None
    ):
        super().__init__(tokenizer, config, model, full_context_mode, num_samples)
        self.max_offset = config.get("max_offset", 999999)

    def _load_dataset(self):
        dataset = datasets.load_dataset(self.config.path, self.config.split)["train"]
        if self.full_context_mode:
            return self._create_full_context_items(dataset)
        return dataset

def get_wmdp_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DictConfig
) -> Dataset:
    forget_config = config.forget
    retain_config = config.retain
    dynamic_config = config.get("dynamic", None)
    retain_validation_config = config.get("retain_validation", None)

    forget_dataset = WMDPDataset(tokenizer, forget_config, model=None)
    # retain_dataset = WMDPDataset(tokenizer, retain_config, model=None) if retain_config.get("path") and "wmdp" in retain_config.get("path") else WikiTextDataset(tokenizer, retain_config)
    retain_dataset = WikiTextDataset(tokenizer, retain_config)
    dynamic_dataset = WMDPDataset(tokenizer, dynamic_config, model=None) if dynamic_config else None
    retain_validation_dataset = WMDPDataset(tokenizer, retain_validation_config, model=None) if retain_validation_config else None

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
                    result[key] = WMDPDataset.collate_fn(collected_batch)
            return result

    return CombinedDataset(forget_dataset, retain_dataset, dynamic_dataset, retain_validation_dataset)
