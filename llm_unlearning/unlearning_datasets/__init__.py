from .hp import *
from .tofu import *
from .wmdp import *
from .augment import *

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

def load_unlearning_dataset(dataset_cfg: DictConfig, tokenizer: PreTrainedTokenizer) -> Dataset:
    if dataset_cfg.name == "tofu":
        dataset = get_tofu_dataset(tokenizer, dataset_cfg)
    elif dataset_cfg.name == "wmdp":
        dataset = get_wmdp_dataset(tokenizer, dataset_cfg)
    elif dataset_cfg.name == "hp":
        dataset = get_hp_dataset(tokenizer, dataset_cfg)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_cfg.name}")

    return dataset, dataset.collate_fn
