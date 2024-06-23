from .tofu import *

from omegaconf import DictConfig
from datasets import load_dataset
from transformers import PreTrainedTokenizer

def load_unlearning_dataset(dataset_cfg: DictConfig, tokenizer: PreTrainedTokenizer) -> Dataset:
    if dataset_cfg.path == "locuslab/TOFU":
        dataset = TofuDataset(tokenizer, dataset_cfg)
        collate_fn = TofuDataset.collate_fn
    else:
        print(f"Warning: This dataset is not natively supported, but trying to load it using HuggingFace datasets library anyway.")
        dataset = load_dataset(dataset_cfg.path, dataset_cfg.split)
        collate_fn = None
    return dataset, collate_fn
