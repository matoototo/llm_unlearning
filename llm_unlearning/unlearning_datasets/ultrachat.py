import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List
from omegaconf import DictConfig
import datasets

class UltraChatDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DictConfig, model=None):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length
        self.data = self._load_dataset()

    def _load_dataset(self):
        dataset = datasets.load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item['messages']

        full_text = ""
        for message in messages:
            role = message['role']
            content = message['content']
            full_text += f"{role.capitalize()}: {content}\n\n"

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()
        labels = input_ids.clone()

        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0].keys()
        }
