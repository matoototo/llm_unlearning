import torch
from torch.utils.data import Dataset
import datasets
from transformers import PreTrainedTokenizer
from typing import Dict, List
from omegaconf import DictConfig

class WMDPDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.data = self._load_split()

    def _load_split(self):
        key = "train" if "corpus" in self.config.path else "test"
        return datasets.load_dataset(self.config.path, self.config.split)[key]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._process_item(self.data[idx])

    def _process_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        if 'text' in item:
            return self._encode_text(item['text'])
        question = item[self.config.question_key]
        answer_index = item[self.config.answer_key]
        paraphrased_answers = item[self.config.paraphrased_answer_key]
        return self._encode_qa_pair(question, paraphrased_answers[answer_index], answer_index, paraphrased_answers)

    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            max_length=self.config.max_length,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()
        labels = input_ids.clone()

        # Set padding tokens to -100 in labels
        padding_mask = (attention_mask == 0).long()
        # First non-zero padding mask element is eos token, don't mask it
        padding_mask[padding_mask.argmax()] = False
        labels = labels.masked_fill(padding_mask.bool(), -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _encode_qa_pair(self, question: str, answer: str, answer_index: int, paraphrased_answers: List[str]) -> Dict[str, torch.Tensor]:
        full_text = f"{self.config.question_start_tag}{question}{self.config.question_end_tag}{self.config.answer_tag}{answer}"

        encoded = self.tokenizer(
            full_text,
            max_length=self.config.max_length,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()
        labels = input_ids.clone()

        # Set padding tokens to -100 in labels
        padding_mask = (attention_mask == 0).long()
        # First non-zero padding mask element is eos token, don't mask it
        padding_mask[padding_mask.argmax()] = False
        labels = labels.masked_fill(padding_mask.bool(), -100)

        # Encode paraphrased answers
        paraphrased_encodings = [
            self._encode_text(f"{self.config.question_start_tag}{question}{self.config.question_end_tag}{self.config.answer_tag}{pa}")
            for pa in paraphrased_answers[:answer_index] + paraphrased_answers[answer_index + 1:]
        ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "answer_index": torch.tensor(answer_index, dtype=torch.long),
            "paraphrased_input_ids": torch.stack([pe["input_ids"] for pe in paraphrased_encodings]),
            "paraphrased_attention_mask": torch.stack([pe["attention_mask"] for pe in paraphrased_encodings]),
            "paraphrased_labels": torch.stack([pe["labels"] for pe in paraphrased_encodings]),
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        result = {}
        for key in batch[0].keys():
            if key in ["input_ids", "attention_mask", "labels"]:
                result[key] = torch.stack([item[key] for item in batch])
            elif key == "answer_index":
                result[key] = torch.stack([item[key] for item in batch if "answer_index" in item])
            elif key.startswith("paraphrased_"):
                result[key] = torch.stack([item[key] for item in batch if key in item])
        return result

class WikiTextDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length
        self.data = self._load_dataset()

    def _load_dataset(self):
        return datasets.load_dataset("wikitext", "wikitext-103-v1", split="train")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._process_item(self.data[idx])

    def _process_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        text = item['text']
        return self._encode_text(text)

    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()
        labels = input_ids.clone()

        # Set padding tokens to -100 in labels
        padding_mask = (attention_mask == 0).long()
        # First non-zero padding mask element is eos token, don't mask it
        padding_mask[padding_mask.argmax()] = False
        labels = labels.masked_fill(padding_mask.bool(), -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            k: torch.stack([item[k] for item in batch])
            for k in batch[0].keys()
        }

def get_wmdp_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DictConfig
) -> Dataset:
    forget_config = config.forget
    retain_config = config.retain

    forget_dataset = WMDPDataset(tokenizer, forget_config)
    retain_dataset = WikiTextDataset(tokenizer, retain_config)

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
                collected_batch = [item[key] for item in batch]
                result[key] = WMDPDataset.collate_fn(collected_batch)  # Both datasets use the same collate_fn
            return result

    return CombinedDataset(forget_dataset, retain_dataset)
