from omegaconf import DictConfig
from datasets import load_dataset

def load_unlearning_dataset(dataset_cfg: DictConfig):
    dataset = load_dataset(dataset_cfg.path, dataset_cfg.split)
    return dataset
