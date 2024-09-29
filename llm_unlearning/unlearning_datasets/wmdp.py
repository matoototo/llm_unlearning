import torch
from torch.utils.data import Dataset
import datasets
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Dict, List, Optional
from omegaconf import DictConfig, OmegaConf
from rouge_score import rouge_scorer
from collections import deque

from llm_unlearning.evals.utils import probability

class WMDPDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        model: Optional[PreTrainedModel] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.data = self._load_split()
        self.model = model
        self.use_dynamic_labels = config.get("use_dynamic_labels", False)
        self.generation_config = config.get("generation_config", {})
        self.regenerate_every = config.get("regenerate_every", 1)
        self.current_epoch = -1
        self.dynamic_data = None
        self.max_length = config.max_length
        self.max_length_difference = config.get("max_length_difference", 999999)
        self.min_prefix_length = config.get("min_prefix_length", 100)
        self.max_prefix_length = config.get("max_prefix_length", 200)
        self.max_offset = config.get("max_offset", 999999)
        self.max_rouge_score = config.get("max_rouge_score", 1.0)
        self.max_logprob_difference = config.get("max_logprob_difference", float('inf'))
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.max_regeneration_attempts = config.get("max_regeneration_attempts", 20)
        self.original_logprobs_cache = {}

    def _load_split(self):
        return datasets.load_dataset(self.config.path, self.config.split)["train"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_dynamic_labels and self.dynamic_data is not None:
            item = self.dynamic_data[idx]
            return self._process_dynamic_item(item)
        return self._process_item(self.data[idx])

    def _process_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        text = item['text']

        offset = min(self.max_offset, len(text) - 1) if self.max_offset > 0 else 0
        if offset > 0:
            offset = torch.randint(0, offset + 1, (1,)).item()
            text = text[offset:]

        return self._encode_text(text)

    def _process_dynamic_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        # Concatenate prompt and generated_text
        text = item['prompt'] + item['generated_text']
        prefix_length = item['prefix_length']
        return self._encode_text(text, prefix_length)

    def _encode_text(self, text: str, prefix_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
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

        if prefix_length is None:
            prefix_length = 0

        labels = input_ids.clone()

        # Mask out labels for the prompt
        labels[:prefix_length] = -100

        # Mask out padding tokens
        padding_mask = (attention_mask == 0)
        labels = labels.masked_fill(padding_mask, -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prefix_length": torch.tensor(prefix_length)
        }

    def _generate_texts_batch(self, prompts: List[str], original_texts: List[str]) -> List[str]:
        if self.model is None:
            raise ValueError("Model is not set. Cannot generate dynamic labels.")

        generation_config = OmegaConf.to_container(self.generation_config, resolve=True)
        batch_size = generation_config.pop("batch_size", 8)
        generation_config = OmegaConf.create(generation_config)

        # Temporarily set padding side to left
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        if self.max_logprob_difference != float('inf'):
            self._compute_and_cache_original_logprobs(prompts, original_texts)

        job_queue = deque(enumerate(zip(prompts, original_texts)))
        all_generated_texts = [None] * len(prompts)
        regeneration_counts = [0] * len(prompts)

        while job_queue:
            current_batch_size = min(batch_size, len(job_queue))
            batch = [job_queue.popleft() for _ in range(current_batch_size)]
            batch_indices, batch_data = zip(*batch)
            batch_prompts, batch_original_texts = zip(*batch_data)

            inputs = self.tokenizer(list(batch_prompts), return_tensors="pt", padding=True, truncation=True).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generation_config
                )

            batch_generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            prompt_lengths = []
            generated_texts_full = []

            for idx, prompt, generated_text in zip(batch_indices, batch_prompts, batch_generated_texts):
                # Remove the prompt from the generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):]
                else:
                    # Can't remove prompt, proceed with generated_text as is
                    print("will this ever happen? L137")
                    pass

                # Compute Rouge score between generated_text and original continuation
                original_text = batch_original_texts[batch_indices.index(idx)]
                original_continuation = original_text[len(prompt):]
                rouge_score_value = self.rouge_scorer.score(original_continuation, generated_text)['rougeL'].fmeasure
                rouge_score_enough = rouge_score_value <= self.max_rouge_score
                length_difference_acceptable = abs(len(generated_text) - len(original_continuation)) / max(len(original_continuation), 1) <= self.max_length_difference

                # Prepare texts for logprob computation
                generated_text_full = prompt + generated_text
                generated_texts_full.append(generated_text_full)
                prompt_tokenized = self.tokenizer.encode(prompt, add_special_tokens=False)
                prompt_lengths.append(len(prompt_tokenized))

            if self.max_logprob_difference != float('inf'):
                generated_encodings = self.tokenizer(generated_texts_full, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
                labels_generated = generated_encodings['input_ids'].clone()

                for i in range(len(batch_indices)):
                    labels_generated[i][:prompt_lengths[i]] = -100

                with torch.no_grad():
                    outputs_generated = self.model(**generated_encodings)

                logprob_generated = probability(outputs_generated.logits, labels_generated, logprobs=True)

            for i, idx in enumerate(batch_indices):
                too_many_attempts = regeneration_counts[idx] >= self.max_regeneration_attempts
                accept_generation = (rouge_score_enough and length_difference_acceptable) or too_many_attempts

                if self.max_logprob_difference != float('inf'):
                    logprob_gen = logprob_generated[i].item()
                    logprob_orig = self.original_logprobs_cache[idx]
                    logprob_difference = logprob_gen - logprob_orig
                    accept_generation = accept_generation and (logprob_difference >= -self.max_logprob_difference)

                if accept_generation:
                    all_generated_texts[idx] = batch_generated_texts[i][len(batch_prompts[i]):]
                else:
                    regeneration_counts[idx] += 1
                    job_queue.append((idx, (batch_prompts[i], batch_original_texts[i])))

        self.tokenizer.padding_side = original_padding_side
        return all_generated_texts

    def _compute_and_cache_original_logprobs(self, prompts: List[str], original_texts: List[str]):
        """
        Compute and cache original log probabilities for all prompts and original texts.
        """
        if self.original_logprobs_cache:
            return

        batch_size = len(prompts)
        batch_indices = list(range(batch_size))

        original_texts_full = []
        prompt_lengths = []

        for idx, prompt, original_text in zip(batch_indices, prompts, original_texts):
            original_text_full = original_text
            original_texts_full.append(original_text_full)
            prompt_tokenized = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_lengths.append(len(prompt_tokenized))

        original_encodings = self.tokenizer(original_texts_full, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        labels_original = original_encodings['input_ids'].clone()

        for i in range(len(batch_indices)):
            labels_original[i][:prompt_lengths[i]] = -100

        with torch.no_grad():
            outputs_original = self.model(**original_encodings)

        logprob_original = probability(outputs_original.logits, labels_original, logprobs=True)

        for i, idx in enumerate(batch_indices):
            self.original_logprobs_cache[idx] = logprob_original[i].item()

    def set_epoch(self, epoch: float):
        epoch = int(epoch)
        if self.current_epoch == epoch:
            return
        if not self.use_dynamic_labels or (self.dynamic_data is not None and epoch % self.regenerate_every != 0):
            self.current_epoch = epoch
            return

        prompts = []
        original_texts = []
        prefix_lengths = []

        for item in self.data:
            text = item['text']

            offset = min(self.max_offset, len(text) - 1) if self.max_offset > 0 else 0
            if offset > 0:
                offset = torch.randint(0, offset + 1, (1,)).item()
                text = text[offset:]

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            full_length = len(tokens)

            min_prefix = min(self.min_prefix_length, full_length - 1)
            max_prefix = min(self.max_prefix_length, full_length - 1)

            if min_prefix >= max_prefix:
                prefix_length = min_prefix
            else:
                prefix_length = torch.randint(min_prefix, max_prefix + 1, (1,)).item()

            prompt_tokens = tokens[:prefix_length]
            prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            prompts.append(prompt)
            original_texts.append(text)
            prefix_lengths.append(prefix_length)

        generated_texts = self._generate_texts_batch(prompts, original_texts)

        self.dynamic_data = [
            {'prompt': prompt, 'generated_text': generated_text, 'prefix_length': prefix_length}
            for prompt, generated_text, prefix_length in zip(prompts, generated_texts, prefix_lengths)
        ]
        self.current_epoch = epoch

    def set_model(self, model):
        self.model = model

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_dict = {
            k: torch.stack([item[k] for item in batch])
            for k in batch[0].keys()
        }
        return WMDPDataset.trim_batch(batch_dict)

    @staticmethod
    def trim_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        trimmed_batch = {}
        max_length = batch["attention_mask"].sum(dim=1).max().item()
        for key in ["input_ids", "labels", "attention_mask"]:
            trimmed_batch[key] = batch[key][:, :max_length]

        if "prefix_length" in batch:
            trimmed_batch["prefix_length"] = batch["prefix_length"]

        return trimmed_batch

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
    dynamic_config = config.get("dynamic", None)

    forget_dataset = WMDPDataset(tokenizer, forget_config, model=None)
    retain_dataset = WikiTextDataset(tokenizer, retain_config)
    dynamic_dataset = WMDPDataset(tokenizer, dynamic_config, model=None) if dynamic_config else None

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

            retain_idx = torch.randint(0, len(self.retain_dataset), (1,)).item()
            result["retain_inputs"] = self.retain_dataset[retain_idx]

            # some wikitext rows are empty, here we resample that case
            while (result["retain_inputs"]['input_ids'] == 50256).all():
                retain_idx = torch.randint(0, len(self.retain_dataset), (1,)).item()
                result["retain_inputs"] = self.retain_dataset[retain_idx]

            return result

        def set_epoch(self, epoch):
            for dataset in [self.forget_dataset, self.retain_dataset, self.dynamic_dataset]:
                if hasattr(dataset, 'set_epoch'):
                    dataset.set_epoch(epoch)

        def set_model(self, model):
            for dataset in [self.forget_dataset, self.dynamic_dataset]:
                if hasattr(dataset, 'set_model'):
                    dataset.set_model(model)

        @staticmethod
        def collate_fn(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
            result = {}
            for key in ['forget_inputs', 'dynamic_inputs', 'retain_inputs']:
                if key not in batch[0]:
                    continue
                collected_batch = [item[key] for item in batch]
                if key == 'retain_inputs':
                    result[key] = WikiTextDataset.collate_fn(collected_batch)
                else:
                    result[key] = WMDPDataset.collate_fn(collected_batch)
            return result

    return CombinedDataset(forget_dataset, retain_dataset, dynamic_dataset)
