import torch
from torch.utils.data import Dataset
import datasets
from transformers import PreTrainedTokenizer
from typing import Dict, List, Tuple
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
        self.forget_split = config.forget_split

        self.forget_data = datasets.load_dataset("locuslab/TOFU", self.forget_split)["train"]
        retain_split = f"retain{100 - int(self.forget_split.replace('forget', '')):02d}"
        self.retain_data = datasets.load_dataset("locuslab/TOFU", retain_split)["train"]

    def __len__(self) -> int:
        return len(self.forget_data)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        forget_item = self._process_item(self.forget_data[idx])
        retain_idx = torch.randint(0, len(self.retain_data), (1,)).item()
        retain_item = self._process_item(self.retain_data[retain_idx])

        result = {
            "forget_inputs": forget_item,
            "retain_inputs": retain_item
        }

        return result

    def _process_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        question = item["question"]
        answer = item["answer"]

        input_ids, attention_mask, labels = self._encode_qa_pair(question, answer)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def _encode_qa_pair(self, question: str, answer: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        question_start_token = self.config.question_start_tag
        question_end_token = self.config.question_end_tag
        answer_token = self.config.answer_tag

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

        # -100 -> ignore loss
        question_end = (input_ids == self.tokenizer.convert_tokens_to_ids(question_end_token)).nonzero()[0].item()
        labels[:question_end + 1] = -100

        return input_ids, attention_mask, labels

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            key: {
                k: torch.stack([item[key][k] for item in batch])
                for k in batch[0][key].keys()
            }
            for key in ['forget_inputs', 'retain_inputs']
        }
