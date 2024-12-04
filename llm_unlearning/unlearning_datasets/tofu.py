import torch
from torch.utils.data import Dataset
import datasets
from transformers import PreTrainedTokenizer, PreTrainedModel, LogitsProcessor, LogitsProcessorList
from typing import Dict, List, Optional
from omegaconf import DictConfig, OmegaConf
from rouge_score import rouge_scorer
from collections import deque
import copy

import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from llm_unlearning.evals.utils import probability

class PerSequenceBadTokenLogitsProcessor(LogitsProcessor):
    """Prevents specific tokens from being generated for each sequence in a batch."""
    def __init__(self, bad_tokens_per_seq: List[int]):
        """
        Args:
            bad_tokens_per_seq: List of token ids to prevent for each sequence in batch.
                              Length of list should match batch size.
        """
        self.bad_tokens_per_seq = bad_tokens_per_seq

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # For each sequence in the batch, mask its corresponding bad token
        for seq_idx, token_id in enumerate(self.bad_tokens_per_seq):
            scores[seq_idx, token_id] = float('-inf')

        return scores

class TofuDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        model: Optional[PreTrainedModel] = None,
        reference_model: Optional[PreTrainedModel] = None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.config = config
        self.max_length = config.max_length
        self.split = config.split
        self.data = self._load_split()
        self.model = model
        self.reference_model = reference_model
        self.original_model = copy.deepcopy(model).cpu() if model is not None else None
        self.generation_config = config.get("generation_config", {})
        self.use_dynamic_labels = config.get("use_dynamic_labels", False)
        self.use_branching_mode = config.get("use_branching_mode", False)
        self.branch_use_previous = config.get("branch_use_previous", False)
        self.mask_gt_token = config.get("mask_gt_token", False)
        if self.mask_gt_token and not self.use_branching_mode:
            raise ValueError("mask_gt_token can only be True when use_branching_mode is True")
        self.branch_temperature = config.get("branch_temperature", 0.0)
        self.regenerate_every = config.get("regenerate_every", 1)
        self.current_epoch = -1
        self.dynamic_data = None
        self.max_rouge_score = config.get("max_rouge_score", 1.0)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.max_regeneration_attempts = config.get("max_regeneration_attempts", 20)
        self.max_length_difference = config.get("max_length_difference", 999999)
        self.max_logprob_difference = config.get("max_logprob_difference", float('inf'))
        self.use_original_for_logdiff = config.get("use_original_for_logdiff", True)
        self.original_logprobs_cache = {}

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

        question_text = f"{question_start_tag}{question}{question_end_tag}"
        question_encoded = self.tokenizer.encode(question_text, add_special_tokens=True)

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

        # Find the maximum overlapping prefix
        question_length = 0
        for i in range(min(len(question_encoded), len(input_ids))):
            if question_encoded[i] != input_ids[i]: break
            question_length += 1

        labels = input_ids.clone()

        # Set labels for the question part (including question tokens) to -100
        labels[:question_length] = -100

        # Set padding tokens to -100 in labels
        padding_mask = (attention_mask == 0).long()
        # First non-zero padding mask element is eos token, don't mask it
        eos_index = padding_mask.argmax()
         # Check if eos_index is not the first token
        if eos_index > 0: padding_mask[eos_index] = 0
        labels = labels.masked_fill(padding_mask.bool(), -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question_length": torch.tensor(question_length)
        }

    def _load_original_model_to_cuda(self):
        if self.original_model is not None:
            self.original_model = self.original_model.cuda()

    def _remove_original_model_from_cuda(self):
        if self.original_model is not None:
            self.original_model = self.original_model.cpu()
            torch.cuda.empty_cache()

    def _compute_and_cache_original_logprobs(self, questions: List[str], original_answers: List[str]):
        if self.original_logprobs_cache:
            return

        self._load_original_model_to_cuda()

        generation_config = OmegaConf.to_container(self.generation_config, resolve=True)
        batch_size = generation_config.pop("batch_size", 8)

        job_queue = deque(enumerate(zip(questions, original_answers)))

        while job_queue:
            current_batch_size = min(batch_size, len(job_queue))
            batch = [job_queue.popleft() for _ in range(current_batch_size)]
            batch_indices, batch_data = zip(*batch)
            batch_questions, batch_original_answers = zip(*batch_data)

            input_texts = [
                f"{self.config.question_start_tag}{q}{self.config.question_end_tag}{self.config.answer_tag}{a}"
                for q, a in zip(batch_questions, batch_original_answers)
            ]
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.original_model.device)

            labels = inputs['input_ids'].clone()
            attention_mask = inputs['attention_mask']
            question_lengths = []

            for i, q in enumerate(batch_questions):
                question_text = f"{self.config.question_start_tag}{q}{self.config.question_end_tag}"
                question_encoded = self.tokenizer.encode(question_text, add_special_tokens=True)
                question_length = len(question_encoded)
                question_lengths.append(question_length)
                labels[i][:question_length] = -100

            # Mask out padding tokens
            padding_mask = (attention_mask == 0)
            labels = labels.masked_fill(padding_mask, -100)

            with torch.no_grad():
                outputs = self.original_model(**inputs)

            logprob_original = probability(outputs.logits, labels, logprobs=True)

            for i, idx in enumerate(batch_indices):
                self.original_logprobs_cache[idx] = logprob_original[i].item()

        self._remove_original_model_from_cuda()

    def compute_token_disagreement(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Compute token-wise disagreement between model and reference model. Returns scores where high values indicate tokens where model assigns high prob but reference model assigns low prob."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            ref_outputs = self.reference_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            model_probs = F.softmax(outputs.logits[:, start_pos:], dim=-1)
            ref_probs = F.softmax(ref_outputs.logits[:, start_pos:], dim=-1)

            next_tokens = input_ids[:, start_pos+1:]
            model_token_probs = torch.gather(model_probs, 2, next_tokens.unsqueeze(-1)).squeeze(-1)
            ref_token_probs = torch.gather(ref_probs, 2, next_tokens.unsqueeze(-1)).squeeze(-1)

            disagreement = model_token_probs - ref_token_probs
            disagreement = torch.clamp(disagreement, min=0)

            mask = attention_mask[:, start_pos+1:].float()
            disagreement = disagreement * mask

            return disagreement

    def sample_branching_point(self, disagreement_scores: torch.Tensor, temperature: float = 1.0) -> int:
        """Sample a branching point based on disagreement scores."""
        if disagreement_scores.numel() == 0 or torch.all(disagreement_scores == 0):
            return torch.randint(0, max(1, disagreement_scores.size(0) - 1), (1,)).item()

        if temperature == 0:
            return disagreement_scores.argmax().item()

        scores = disagreement_scores / (temperature + 1e-10)
        probs = F.softmax(scores, dim=-1)

        if torch.isnan(probs).any():
            probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
            probs = probs / probs.sum()

        return min(torch.distributions.Categorical(probs).sample().item(), disagreement_scores.size(0) - 2)

    def _generate_answers_batch(self, questions: List[str], base_answers: List[str], original_answers: List[str]) -> List[str]:
        if self.model is None:
            raise ValueError("Model is not set. Cannot generate dynamic labels.")
        if self.use_branching_mode and self.reference_model is None:
            raise ValueError("Reference model is not set. Cannot use branching mode.")

        # doesn't support pop, have to do this garbage
        generation_config = OmegaConf.to_container(self.generation_config, resolve=True)
        batch_size = generation_config.pop("batch_size", 8)
        generation_config = OmegaConf.create(generation_config)

        # temporarily set padding side to left
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        if self.max_logprob_difference != float('inf'):
            self._compute_and_cache_original_logprobs(questions, original_answers)

        job_queue = deque(enumerate(zip(questions, base_answers, original_answers)))
        all_answers = [None] * len(questions)
        regeneration_counts = [0] * len(questions)

        while job_queue:
            current_batch_size = min(batch_size, len(job_queue))
            batch = [job_queue.popleft() for _ in range(current_batch_size)]
            batch_indices, batch_data = zip(*batch)
            batch_questions, batch_base_answers, batch_original_answers = zip(*batch_data)

            if self.use_branching_mode:
                input_texts = [f"{self.config.question_start_tag}{q}{self.config.question_end_tag}{self.config.answer_tag}{a}" for q, a in zip(batch_questions, batch_base_answers)]
                encoded = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.model.device)

                answer_starts = []
                for q in batch_questions:
                    prefix = f"{self.config.question_start_tag}{q}{self.config.question_end_tag}{self.config.answer_tag}"
                    answer_starts.append(len(self.tokenizer.encode(prefix, add_special_tokens=True)) - 1)

                disagreements = []
                branch_positions = []
                for i, content_start_pos in enumerate(answer_starts):
                    seq_start = (encoded.attention_mask[i] == 1).nonzero()[0].item()
                    rel_start_pos = content_start_pos + seq_start

                    scores = self.compute_token_disagreement(encoded.input_ids[i:i+1], encoded.attention_mask[i:i+1], rel_start_pos)
                    disagreements.append(scores[0])

                    rel_branch_pos = self.sample_branching_point(scores[0], self.branch_temperature) + rel_start_pos + 1
                    branch_positions.append(rel_branch_pos)

                prefix_ids = []
                prefix_mask = []
                for i, branch_pos in enumerate(branch_positions):
                    seq_start = (encoded.attention_mask[i] == 1).nonzero()[0].item()
                    seq_prefix = encoded.input_ids[i, seq_start:branch_pos]
                    mask_prefix = encoded.attention_mask[i, seq_start:branch_pos]
                    # left pad to old size
                    prefix_ids.append(F.pad(seq_prefix, (self.max_length - len(seq_prefix), 0), value=self.tokenizer.pad_token_id))
                    prefix_mask.append(F.pad(mask_prefix, (self.max_length - len(mask_prefix), 0), value=0))

                prefix_ids = torch.stack(prefix_ids)
                prefix_mask = torch.stack(prefix_mask)

                generation_kwargs = {**generation_config}
                if self.mask_gt_token:
                    # Get tokens to prevent for each sequence
                    bad_tokens = [encoded.input_ids[i, branch_positions[i]].item() for i in range(current_batch_size)]
                    logits_processor = LogitsProcessorList([PerSequenceBadTokenLogitsProcessor(bad_tokens)])
                    generation_kwargs["logits_processor"] = logits_processor
                    generation_kwargs["renormalize_logits"] = True
                    if generation_kwargs.get("temperature", 1.0) == 0:
                        del generation_kwargs["temperature"]
                        generation_kwargs["do_sample"] = False

                outputs = self.model.generate(input_ids=prefix_ids, attention_mask=prefix_mask, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.max_length, **generation_kwargs)

                batch_answers = []
                for output in self.tokenizer.batch_decode(outputs, skip_special_tokens=True):
                    answer = output.split(self.config.answer_tag)[-1].strip()
                    batch_answers.append(answer)
            else:
                input_texts = [f"{self.config.question_start_tag}{q}{self.config.question_end_tag}" for q in batch_questions]
                inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=self.max_length, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id, **generation_config)

                batch_answers = [output.split(self.config.question_end_tag)[-1].strip()[len(self.config.answer_tag):] for output in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]

            if self.max_logprob_difference != float('inf'):
                # Prepare inputs for computing logprobs of generated answers
                generated_input_texts = [
                    f"{self.config.question_start_tag}{q}{self.config.question_end_tag}{self.config.answer_tag}{a}"
                    for q, a in zip(batch_questions, batch_answers)
                ]
                generated_inputs = self.tokenizer(
                    generated_input_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.model.device)

                labels_generated = generated_inputs['input_ids'].clone()
                attention_mask_generated = generated_inputs['attention_mask']
                question_lengths = []

                for i, q in enumerate(batch_questions):
                    question_text = f"{self.config.question_start_tag}{q}{self.config.question_end_tag}"
                    question_encoded = self.tokenizer.encode(question_text, add_special_tokens=True)
                    question_length = len(question_encoded)
                    question_lengths.append(question_length)
                    labels_generated[i][:question_length] = -100

                # Mask out padding tokens
                padding_mask_generated = (attention_mask_generated == 0)
                labels_generated = labels_generated.masked_fill(padding_mask_generated, -100)

                with torch.no_grad():
                    if self.use_original_for_logdiff:
                        self._load_original_model_to_cuda()
                        outputs_generated = self.original_model(**generated_inputs)
                        self._remove_original_model_from_cuda()
                    else: outputs_generated = self.model(**generated_inputs)

                logprob_generated = probability(outputs_generated.logits, labels_generated, logprobs=True)

            for i, idx in enumerate(batch_indices):
                answer = batch_answers[i]
                original_answer = batch_original_answers[i]
                rouge_score_value = self.rouge_scorer.score(original_answer, answer)['rougeL'].fmeasure
                rouge_score_enough = rouge_score_value <= self.max_rouge_score
                length_difference_acceptable = abs(len(answer) - len(original_answer)) / len(original_answer) <= self.max_length_difference
                too_many_attempts = regeneration_counts[idx] >= self.max_regeneration_attempts

                accept_generation = (rouge_score_enough and length_difference_acceptable) or too_many_attempts

                if self.max_logprob_difference != float('inf') and not too_many_attempts:
                    logprob_gen = logprob_generated[i].item()
                    logprob_orig = self.original_logprobs_cache[idx]
                    logprob_difference = logprob_gen - logprob_orig
                    accept_generation = accept_generation and (logprob_difference >= -self.max_logprob_difference)

                if accept_generation or too_many_attempts:
                    all_answers[idx] = answer
                else:
                    regeneration_counts[idx] += 1
                    job_queue.append((idx, (batch_questions[i], batch_base_answers[i], original_answer)))

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

        # Use previously generated answers if they exist and branch_use_previous is True (for branching)
        base_answers = (
            [item[self.config.answer_key] for item in self.dynamic_data]
            if self.dynamic_data is not None and self.branch_use_previous
            else original_answers
        )

        generated_answers = self._generate_answers_batch(questions, base_answers, original_answers)

        self.dynamic_data = [
            {**item, self.config.answer_key: new_answer}
            for item, new_answer in zip(self.data, generated_answers)
        ]
        self.current_epoch = epoch

    def set_model(self, model):
        if not self.model:  # Only copy the first time
            self.original_model = copy.deepcopy(model).cpu()
        self.model = model

    def set_reference_model(self, reference_model):
        self.reference_model = reference_model

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
    retain_validation_dataset = TofuDataset(tokenizer, config.get("retain_validation")) if config.get("retain_validation") else None

    class CombinedDataset(Dataset):
        def __init__(self, forget_dataset, retain_dataset, dynamic_dataset, retain_validation_dataset):
            self.forget_dataset = forget_dataset
            self.retain_dataset = retain_dataset
            self.dynamic_dataset = dynamic_dataset
            self.retain_validation_dataset = retain_validation_dataset

        def __len__(self):
            return len(self.forget_dataset)

        def __getitem__(self, idx):
            result = {"forget_inputs": self.forget_dataset[idx]}

            if self.dynamic_dataset:
                result["dynamic_inputs"] = self.dynamic_dataset[idx]

            if self.retain_dataset:
                retain_idx = torch.randint(0, len(self.retain_dataset), (1,)).item()
                result["retain_inputs"] = self.retain_dataset[retain_idx]

            if self.retain_validation_dataset:
                retain_val_idx = torch.randint(0, len(self.retain_validation_dataset), (1,)).item()
                result["retain_validation_inputs"] = self.retain_validation_dataset[retain_val_idx]

            return result

        def set_epoch(self, epoch):
            for dataset in [self.forget_dataset, self.retain_dataset, self.dynamic_dataset, self.retain_validation_dataset]:
                if dataset: dataset.set_epoch(epoch)

        def set_model(self, model):
            for dataset in [self.forget_dataset, self.retain_dataset, self.dynamic_dataset, self.retain_validation_dataset]:
                if dataset and hasattr(dataset, 'set_model') and dataset.use_dynamic_labels:
                    dataset.set_model(model)

        @staticmethod
        def collate_fn(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
            result = {}
            for key in ['forget_inputs', 'dynamic_inputs', 'retain_inputs', 'retain_validation_inputs']:
                if key not in batch[0]:
                    continue
                collected_batch = [item[key] for item in batch]
                result[key] = TofuDataset.collate_fn(collected_batch)
            return result

    return CombinedDataset(forget_dataset, retain_dataset, dynamic_dataset, retain_validation_dataset)
