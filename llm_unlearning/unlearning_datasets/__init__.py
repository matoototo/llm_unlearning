from .tofu import *

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

def load_unlearning_dataset(dataset_cfg: DictConfig, tokenizer: PreTrainedTokenizer) -> Dataset:
    if dataset_cfg.name == "tofu":
        dataset = get_tofu_dataset(tokenizer, dataset_cfg)
        collate_fn = dataset.collate_fn
    else:
        raise ValueError(f"Invalid dataset name: {dataset_cfg.name}")
    return dataset, collate_fn
