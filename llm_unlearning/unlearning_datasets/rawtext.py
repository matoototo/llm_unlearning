import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Dict, List, Optional
from omegaconf import DictConfig, OmegaConf
from rouge_score import rouge_scorer
from collections import deque

import os
import copy
import warnings

from llm_unlearning.evals.utils import probability

class RawTextDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        model: Optional[PreTrainedModel] = None,
        full_context_mode: bool = False,
        num_samples: Optional[int] = None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.model = model
        self.original_model = copy.deepcopy(model).cpu() if model is not None else None
        self.max_length = config.max_length
        self.full_context_mode = full_context_mode
        self.num_samples = num_samples
        self.use_dynamic_labels = config.get("use_dynamic_labels", False)
        self.generation_config = config.get("generation_config", {})
        self.regenerate_every = config.get("regenerate_every", 1)
        self.current_epoch = -1
        self.dynamic_data = None
        self.min_prefix_length = config.get("min_prefix_length", 50)
        self.max_prefix_length = config.get("max_prefix_length", 100)
        self.max_offset = config.get("max_offset", 999999)
        self.max_rouge_score = config.get("max_rouge_score", 1.0)
        self.max_logprob_difference = config.get("max_logprob_difference", float('inf'))
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.max_regeneration_attempts = config.get("max_regeneration_attempts", 20)
        self.original_logprobs_cache = {}

        if self.max_offset > 0 and self.full_context_mode:
            warnings.warn("Using offsets and full_context_mode is probably not what you want – it likely won't be full context anymore")

        self.data = self._load_dataset()

    def _load_dataset(self):
        if self.config.get("dataset_type") == "text_file":
            return self._load_text_dataset()
        else:
            raise NotImplementedError("This method should be implemented by subclasses or use 'text_file' dataset_type")

    def _load_text_dataset(self):
        file_path = self.config.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"Invalid or non-existent file path: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []

        for i in range(0, len(tokens), self.max_length):
            chunk_tokens = tokens[i:i + self.max_length]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append({"text": chunk_text})

        if self.full_context_mode:
            return self._create_full_context_items(chunks)

        if self.num_samples is not None:
            chunks = chunks[:self.num_samples]

        return chunks

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_dynamic_labels and self.dynamic_data is not None:
            item = self.dynamic_data[idx]
            return self._process_dynamic_item(item)
        return self._process_multi_text_item(idx)

    def _process_multi_text_item(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self._get_multi_text(idx)
        return self._encode_text(text)

    def _process_dynamic_item(self, item: Dict) -> Dict[str, torch.Tensor]:
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
        labels = input_ids.clone()

        # Set padding tokens to -100 in labels
        padding_mask = (attention_mask == 0).long()
        # First non-zero padding mask element is eos token, don't mask it
        padding_mask[padding_mask.argmax()] = False
        labels = labels.masked_fill(padding_mask.bool(), -100)

        if prefix_length is not None:
            labels[:prefix_length] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if prefix_length is not None:
            result['prefix_length'] = torch.tensor(prefix_length)

        return result

    def _create_full_context_items(self, dataset):
        full_context_items = []
        current_text = ""

        for item in dataset:
            if self.num_samples is not None and len(full_context_items) >= self.num_samples:
                break
            current_text += " " + item['text']
            encoded = self.tokenizer(current_text, max_length=self.max_length, truncation=True, return_overflowing_tokens=True, return_length=True)

            while encoded.length[0] == self.max_length:
                full_context_items.append({'text': self.tokenizer.decode(encoded.input_ids[0])})
                if len(encoded.input_ids) > 1:
                    current_text = self.tokenizer.decode(encoded.input_ids[1])
                else:
                    current_text = ""
                encoded = self.tokenizer(current_text, max_length=self.max_length, truncation=True, return_overflowing_tokens=True, return_length=True)

        lengths = [self.tokenizer(item['text']).input_ids.__len__() for item in full_context_items]
        full_context_items = list(map(lambda i : full_context_items[i],
                                filter(lambda i: lengths[i] == self.max_length, range(len(full_context_items)))))

        return full_context_items

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

            for idx, prompt, generated_text, original_text in zip(batch_indices, batch_prompts, batch_generated_texts, batch_original_texts):
                # Remove the prompt from the generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):]
                else:
                    generated_text = self.remove_prompt_prefix(prompt, generated_text)

                # Compute Rouge score between generated_text and original continuation
                original_text = batch_original_texts[batch_indices.index(idx)]
                original_continuation = original_text[len(prompt):]
                rouge_score_value = self.rouge_scorer.score(original_continuation, generated_text)['rougeL'].fmeasure
                rouge_score_enough = rouge_score_value <= self.max_rouge_score

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
                accept_generation = rouge_score_enough or too_many_attempts

                if self.max_logprob_difference != float('inf') and not too_many_attempts:
                    logprob_gen = logprob_generated[i].item()
                    logprob_orig = self.original_logprobs_cache[idx]
                    logprob_difference = logprob_gen - logprob_orig
                    accept_generation = accept_generation and (logprob_difference >= -self.max_logprob_difference)

                if accept_generation or too_many_attempts:
                    all_generated_texts[idx] = batch_generated_texts[i][len(batch_prompts[i]):]
                else:
                    regeneration_counts[idx] += 1
                    job_queue.append((idx, (batch_prompts[i], batch_original_texts[i])))

        self.tokenizer.padding_side = original_padding_side
        return all_generated_texts

    def _load_original_model_to_cuda(self):
        if self.original_model is not None:
            self.original_model = self.original_model.cuda()

    def _remove_original_model_from_cuda(self):
        if self.original_model is not None:
            self.original_model = self.original_model.cpu()
            torch.cuda.empty_cache()

    def _compute_and_cache_original_logprobs(self, prompts: List[str], original_texts: List[str]):
        if self.original_logprobs_cache:
            return

        self._load_original_model_to_cuda()

        generation_config = OmegaConf.to_container(self.generation_config, resolve=True)
        batch_size = generation_config.pop("batch_size", 8)

        job_queue = deque(enumerate(zip(prompts, original_texts)))

        while job_queue:
            current_batch_size = min(batch_size, len(job_queue))
            batch = [job_queue.popleft() for _ in range(current_batch_size)]
            batch_indices, batch_data = zip(*batch)
            batch_prompts, batch_original_texts = zip(*batch_data)

            prompt_lengths = []
            for prompt in batch_prompts:
                prompt_tokenized = self.tokenizer.encode(prompt, add_special_tokens=False)
                prompt_lengths.append(len(prompt_tokenized))

            original_encodings = self.tokenizer(batch_original_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.original_model.device)
            labels_original = original_encodings['input_ids'].clone()

            for i in range(len(batch_indices)):
                labels_original[i][:prompt_lengths[i]] = -100

            with torch.no_grad():
                outputs_original = self.original_model(**original_encodings)

            logprob_original = probability(outputs_original.logits, labels_original, logprobs=True)

            for i, idx in enumerate(batch_indices):
                self.original_logprobs_cache[idx] = logprob_original[i].item()

        self._remove_original_model_from_cuda()

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

        for idx in range(len(self.data)):
            text = self._get_multi_text(idx)

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

    def _get_multi_text(self, idx: int) -> str:
        text = ""
        current_idx = idx
        total_length = 0

        initial_text = self.data[current_idx]['text']

        offset = torch.randint(0, min(self.max_offset, len(initial_text)), (1,)).item()
        # snap right to closest space
        while offset < len(initial_text) and initial_text[offset] != " ":
            offset += 1
        offset += 1 # skip space
        # this will never happen, but just in case
        if offset >= len(initial_text):
            offset = 0

        text += initial_text[offset:]
        total_length = self.tokenizer.encode(text, add_special_tokens=False).__len__()

        while total_length < self.max_length and current_idx < len(self.data) - 1:
            current_idx += 1
            next_text = self.data[current_idx]['text']
            text += next_text
            total_length = self.tokenizer.encode(text, add_special_tokens=False).__len__()

        # tokens = self.tokenizer.encode(text, add_special_tokens=False)
        # text = self.tokenizer.decode(tokens[:self.max_length])
        return text

    def set_model(self, model):
        if not self.model:  # only copy first time
            self.original_model = copy.deepcopy(model).cpu()
        self.model = model

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_dict = {
            k: torch.stack([item[k] for item in batch])
            for k in batch[0].keys()
        }
        return RawTextDataset.trim_batch(batch_dict)

    @staticmethod
    def trim_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        trimmed_batch = {}
        max_length = batch["attention_mask"].sum(dim=1).max().item()
        for key in ["input_ids", "labels", "attention_mask"]:
            trimmed_batch[key] = batch[key][:, :max_length]

        if "prefix_length" in batch:
            trimmed_batch["prefix_length"] = batch["prefix_length"]

        return trimmed_batch

    def remove_prompt_prefix(self, prompt, generated_text):
        for i in range(len(prompt)):
            if prompt[i] != generated_text[i]:
                return generated_text[i:]
        return generated_text[len(prompt):]
