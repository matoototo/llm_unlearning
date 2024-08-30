import torch
from torch.utils.data import Dataset
import datasets
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Dict, List, Optional
from omegaconf import DictConfig, OmegaConf
from rouge_score import rouge_scorer
from collections import deque

class TofuDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        model: Optional[PreTrainedModel] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length
        self.split = config.split
        self.data = self._load_split()
        self.model = model
        self.generation_config = config.get("generation_config", {})
        self.use_dynamic_labels = config.get("use_dynamic_labels", False)
        self.regenerate_every = config.get("regenerate_every", 1)
        self.current_epoch = -1
        self.dynamic_data = None
        self.max_rouge_score = config.get("max_rouge_score", 1.0)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.max_regeneration_attempts = config.get("max_regeneration_attempts", 20)

    def _load_split(self):
        part = None
        split = self.split
        if ":" in split: split, part = split.split(":")
        try:
            dataset = datasets.load_dataset("locuslab/TOFU", split)["train"]
        except ValueError:
            if split.startswith("full"):
                suffix = int(split[4:])
                forget_set = f"forget{suffix}"
                retain_set = f"retain{100-suffix}"
                forget_data = self._create_custom_split(forget_set)
                retain_data = self._create_custom_split(retain_set)
                if not part: return datasets.concatenate_datasets([forget_data, retain_data])

                f_ind = self._part_indices(forget_data, int(part), is_forget=True)
                r_ind = self._part_indices(retain_data, int(part), is_forget=False)
                return datasets.concatenate_datasets([forget_data.select(f_ind), retain_data.select(r_ind)])
            dataset = self._create_custom_split(split)
        if not part: return dataset
        indices = self._part_indices(dataset, int(part), split.startswith("forget"))
        return dataset.select(indices)

    def _part_indices(self, dataset, part, is_forget):
        indices = [i for i in range(len(dataset))]
        # For forget sets larger than forget10, we need to split the last 400 evenly between the part_0 and part_1, this
        # ensures that forget10_perturbed is evenly represented in both train and val, we don't have larger perturbed splits
        # Likewise for retain, if it's smaller than retain90 then the first 400 should be split evenly (retain_perturbed)
        if is_forget and len(indices) <= 400:
            return indices[:len(indices) // 2] if part == 0 else indices[len(indices) // 2:]
        if not is_forget and len(indices) >= 3600:
            return indices[:len(indices) // 2] if part == 0 else indices[len(indices) // 2:]
        common_indices = indices[-400:] if is_forget else indices[:400]
        indices = indices[:-400] if is_forget else indices[400:] # Remove the common indices
        part_0 = common_indices[:200] + indices[:len(indices) // 2]
        part_1 = common_indices[200:] + indices[len(indices) // 2:]
        return part_0 if part == 0 else part_1

    def _create_custom_split(self, split):
        full_data = datasets.load_dataset("locuslab/TOFU", "full")["train"]
        prefix = split[:6]
        suffix = split[6:]
        if not prefix in ["forget", "retain"] or not suffix.isdigit():
            raise ValueError(f"Invalid split: {split}")
        percentage = int(suffix)
        split_size = int(len(full_data) * percentage / 100)
        indices = [i for i in range(len(full_data))]
        if prefix == "retain": return full_data.select(indices[:split_size])
        return full_data.select(indices[-split_size:])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_dynamic_labels and self.dynamic_data is not None:
            return self._process_item(self.dynamic_data[idx])
        return self._process_item(self.data[idx])

    def _process_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        question = item[self.config.question_key]
        answer = item[self.config.answer_key]

        result = self._encode_qa_pair(question, answer)

        _result_keys = list(result.keys())
        if hasattr(self.config, 'perturbed_answer_key') and self.config.perturbed_answer_key in item:
            perturbed_answers = item[self.config.perturbed_answer_key]
            for key in _result_keys:
                result[f"perturbed_{key}"] = [
                    self._encode_qa_pair(question, perturbed_answer)[key]
                    for perturbed_answer in perturbed_answers
                ]

        if hasattr(self.config, 'paraphrased_answer_key') and self.config.paraphrased_answer_key in item:
            paraphrased_answer = item[self.config.paraphrased_answer_key]
            paraphrased_result = self._encode_qa_pair(question, paraphrased_answer)
            for key in _result_keys:
                result[f"paraphrased_{key}"] = paraphrased_result[key]

        return result

    def _encode_qa_pair(self, question: str, answer: str) -> Dict[str, torch.Tensor]:
        question_start_tag = self.config.question_start_tag
        question_end_tag = self.config.question_end_tag
        answer_tag = self.config.answer_tag

        # Encode question separately to get its length
        question_encoded = self.tokenizer(
            f"{question_start_tag}{question}{question_end_tag}",
            add_special_tokens=True,
            return_tensors="pt"
        )
        question_length = question_encoded.input_ids.size(1)

        full_text = f"{question_start_tag}{question}{question_end_tag}{answer_tag}{answer}"

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()
        labels = input_ids.clone()

        # Set labels for the question part (including question tokens) to -100
        labels[:question_length] = -100

        # Set padding tokens to -100 in labels
        padding_mask = (attention_mask == 0).long()
        # First non-zero padding mask element is eos token, don't mask it
        padding_mask[padding_mask.argmax()] = False
        labels = labels.masked_fill(padding_mask.bool(), -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question_length": torch.tensor(question_length)
        }

    def _generate_answers_batch(self, questions: List[str], original_answers: List[str]) -> List[str]:
        if self.model is None:
            raise ValueError("Model is not set. Cannot generate dynamic labels.")

        # doesn't support pop, have to do this garbage
        generation_config = OmegaConf.to_container(self.generation_config, resolve=True)
        batch_size = generation_config.pop("batch_size", 8)
        generation_config = OmegaConf.create(generation_config)

        # temporarily set padding side to left
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        job_queue = deque(enumerate(zip(questions, original_answers)))
        all_answers = [None] * len(questions)
        regeneration_counts = [0] * len(questions)

        while job_queue:
            batch = [job_queue.popleft() for _ in range(min(batch_size, len(job_queue)))]
            batch_indices, batch_data = zip(*batch)
            batch_questions, batch_original_answers = zip(*batch_data)

            input_texts = [f"{self.config.question_start_tag}{q}{self.config.question_end_tag}" for q in batch_questions]
            inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generation_config
                )

            batch_answers = [
                output.split(self.config.question_end_tag)[-1].strip()[8:]
                for output in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            ]

            for idx, answer, original_answer in zip(batch_indices, batch_answers, batch_original_answers):
                rouge_score_enough = self.rouge_scorer.score(original_answer, answer)['rougeL'].fmeasure <= self.max_rouge_score
                too_many_attempts = regeneration_counts[idx] >= self.max_regeneration_attempts
                if rouge_score_enough or too_many_attempts:
                    all_answers[idx] = answer
                    continue
                regeneration_counts[idx] += 1
                job_queue.append((idx, (batch_questions[batch_indices.index(idx)], original_answer)))

        self.tokenizer.padding_side = original_padding_side
        return all_answers

    def set_epoch(self, epoch: float):
        epoch = int(epoch)
        if self.current_epoch == epoch: return
        if not self.use_dynamic_labels or (self.dynamic_data is not None and epoch % self.regenerate_every != 0):
            self.current_epoch = epoch
            return

        questions = [item[self.config.question_key] for item in self.data]
        original_answers = [item[self.config.answer_key] for item in self.data]
        generated_answers = self._generate_answers_batch(questions, original_answers)

        self.dynamic_data = [
            {**item, self.config.answer_key: new_answer}
            for item, new_answer in zip(self.data, generated_answers)
        ]
        self.current_epoch = epoch

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

        return TofuDataset.trim_batch(result)

    @staticmethod
    def trim_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        trimmed_batch = {}
        has_paraphrased = any(k.startswith("paraphrased_") for k in batch)
        has_generated = "generated_outputs" in batch

        for key in batch:
            if "question_length" not in key: continue
            trimmed_batch[key] = batch[key]

        max_length = batch["attention_mask"].sum(dim=1).max().item()
        for key in ["input_ids", "labels", "attention_mask"]:
            trimmed_batch[key] = batch[key][:, :max_length]

        if has_paraphrased:
            paraphrased_max_length = batch["paraphrased_attention_mask"].sum(dim=1).max().item()
            for key in ["paraphrased_input_ids", "paraphrased_labels", "paraphrased_attention_mask"]:
                trimmed_batch[key] = batch[key][:, :paraphrased_max_length]

        if has_generated:
            generated_max_length = batch["generated_masks"].sum(dim=-1).max().item()
            trimmed_batch["generated_outputs"] = batch["generated_outputs"][:, :, :generated_max_length]
            trimmed_batch["generated_masks"] = batch["generated_masks"][:, :, :generated_max_length]

        perturbed_keys = [k for k in batch if k.startswith("perturbed_") and k != "perturbed_question_length"]
        if not perturbed_keys: return trimmed_batch

        perturbed_max_length = batch["perturbed_attention_mask"][0].sum(dim=1).max().item()
        for key in perturbed_keys:
            trimmed_batch[key] = [tensor[:, :perturbed_max_length] for tensor in batch[key]]

        return trimmed_batch

def get_tofu_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DictConfig
) -> Dataset:
    forget_dataset = TofuDataset(tokenizer, config.get("forget")) if config.get("forget") else None
    retain_dataset = TofuDataset(tokenizer, config.get("retain")) if config.get("retain") else None
    dynamic_dataset = TofuDataset(tokenizer, config.get("dynamic")) if config.get("dynamic") else None

    class CombinedDataset(Dataset):
        def __init__(self, forget_dataset, retain_dataset, dynamic_dataset):
            self.forget_dataset = forget_dataset
            self.retain_dataset = retain_dataset
            self.dynamic_dataset = dynamic_dataset

        def __len__(self):
            return len(self.forget_dataset)

        def __getitem__(self, idx):
            result = {"forget_inputs": self.forget_dataset[idx]}

            if self.dynamic_dataset:
                result["dynamic_inputs"] = self.dynamic_dataset[idx]

            if self.retain_dataset:
                retain_idx = torch.randint(0, len(self.retain_dataset), (1,)).item()
                result["retain_inputs"] = self.retain_dataset[retain_idx]

            return result

        def set_epoch(self, epoch):
            for dataset in [self.forget_dataset, self.retain_dataset, self.dynamic_dataset]:
                if dataset: dataset.set_epoch(epoch)

        def set_model(self, model):
            for dataset in [self.forget_dataset, self.retain_dataset, self.dynamic_dataset]:
                if dataset: dataset.model = model

        @staticmethod
        def collate_fn(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
            result = {}
            for key in ['forget_inputs', 'dynamic_inputs', 'retain_inputs']:
                if key not in batch[0]:
                    continue
                collected_batch = [item[key] for item in batch]
                result[key] = TofuDataset.collate_fn(collected_batch)
            return result

    return CombinedDataset(forget_dataset, retain_dataset, dynamic_dataset)
