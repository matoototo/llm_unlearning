import torch
from torch.utils.data import Dataset
import datasets
from transformers import PreTrainedTokenizer
from typing import Dict, List
from omegaconf import DictConfig

class TofuDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length
        self.split = config.split

        if self.split.startswith('forget') or self.split.startswith('retain'):
            self.data = datasets.load_dataset("locuslab/TOFU", self.split)["train"]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must start with 'forget' or 'retain'.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._process_item(self.data[idx])

    def _process_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        question = item[self.config.question_key]
        answer = item[self.config.answer_key]

        result = self._encode_qa_pair(question, answer)

        if hasattr(self.config, 'perturbed_answer_key'):
            _result_keys = list(result.keys())
            perturbed_answers = item[self.config.perturbed_answer_key]
            for key in _result_keys:
                result[f"perturbed_{key}"] = [
                    self._encode_qa_pair(question, perturbed_answer)[key]
                    for perturbed_answer in perturbed_answers
                ]

        return result

    def _encode_qa_pair(self, question: str, answer: str) -> Dict[str, torch.Tensor]:
        question_start_token = self.config.question_start_tag
        question_end_token = self.config.question_end_tag
        answer_token = self.config.answer_tag

        # Encode question separately to get its length
        question_encoded = self.tokenizer(
            f"{question_start_token}{question}{question_end_token}",
            add_special_tokens=False,
            return_tensors="pt"
        )
        question_length = question_encoded.input_ids.size(1)

        full_text = f"{question_start_token}{question}{question_end_token}{answer_token}{answer}"

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()
        labels = input_ids.clone()

        # Set labels for the question part (including question tokens) to -100
        labels[:question_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        result = {
            k: torch.stack([item[k] for item in batch])
            for k in batch[0].keys() if not k.startswith("perturbed_")
        }

        perturbed_keys = [k for k in batch[0].keys() if k.startswith("perturbed_")]
        for key in perturbed_keys:
            result[key] = [
                torch.stack([item[key][i] for item in batch])
                for i in range(len(batch[0][key]))
            ]

        return result

def get_tofu_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DictConfig
) -> Dataset:
    forget_config = config.forget
    retain_config = config.retain

    forget_dataset = TofuDataset(tokenizer, forget_config)
    retain_dataset = TofuDataset(tokenizer, retain_config)

    class CombinedDataset(Dataset):
        def __init__(self, forget_dataset, retain_dataset):
            self.forget_dataset = forget_dataset
            self.retain_dataset = retain_dataset

        def __len__(self):
            return len(self.forget_dataset)

        def __getitem__(self, idx):
            forget_item = self.forget_dataset[idx]
            retain_idx = torch.randint(0, len(self.retain_dataset), (1,)).item()
            retain_item = self.retain_dataset[retain_idx]

            return {
                "forget_inputs": forget_item,
                "retain_inputs": retain_item
            }

        @staticmethod
        def collate_fn(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
            result = {}
            for key in ['forget_inputs', 'retain_inputs']:
                result[key] = {
                    k: torch.stack([item[key][k] for item in batch])
                    for k in batch[0][key].keys() if not k.startswith("perturbed_")
                }

                perturbed_keys = [k for k in batch[0][key].keys() if k.startswith("perturbed_")]
                for p_key in perturbed_keys:
                    result[key][p_key] = [
                        torch.stack([item[key][p_key][i] for item in batch])
                        for i in range(len(batch[0][key][p_key]))
                    ]

            return result

    return CombinedDataset(forget_dataset, retain_dataset)
