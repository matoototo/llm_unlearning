import torch
import einops

from tqdm import tqdm
from typing import Dict, Any, Callable
from torch.utils.data import DataLoader

from llm_unlearning.unlearning_datasets.tofu import TofuDataset
from llm_unlearning.evals.tofu_evals import truth_ratio, probability, rouge

class Evaluator:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _get_dataloader(self, dataset: TofuDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )

    def _process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        def move_to_device(item):
            if isinstance(item, torch.Tensor):
                return item.to(self.device)
            elif isinstance(item, list) and all(isinstance(x, torch.Tensor) for x in item):
                return [x.to(self.device) for x in item]
            return item

        return {k: move_to_device(v) for k, v in batch.items()}

    def _compute_metric(self, dataset: TofuDataset, metric_fn: Callable, desc: str) -> Dict[str, float]:
        dataloader = self._get_dataloader(dataset)
        total_metric = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                batch = self._process_batch(batch)
                batch_metric = metric_fn(batch)
                total_metric += batch_metric.sum().item()
                total_samples += batch_metric.size(0)

        avg_metric = total_metric / total_samples
        return {desc.lower().replace(" ", "_"): avg_metric}

    def compute_truth_ratio(self, dataset: TofuDataset) -> Dict[str, float]:
        def truth_ratio_fn(batch):
            input_keys = ["input_ids", "attention_mask", "labels"]
            gt_outputs = self.model(**{k: v for k, v in batch.items() if k in input_keys})
            gt_logits = gt_outputs.logits
            gt_labels = batch["labels"]

            perturbed_logits = []
            perturbed_labels = []

            for i in range(len(batch["perturbed_input_ids"])):
                perturbed_output = self.model(
                    input_ids=batch["perturbed_input_ids"][i],
                    attention_mask=batch["perturbed_attention_mask"][i]
                )
                perturbed_logits.append(perturbed_output.logits)
                perturbed_labels.append(batch["perturbed_labels"][i])

            perturbed_logits = torch.stack(perturbed_logits, dim=1)
            perturbed_labels = torch.stack(perturbed_labels, dim=1)

            return truth_ratio(gt_logits, gt_labels, perturbed_logits, perturbed_labels)

        return self._compute_metric(dataset, truth_ratio_fn, "Truth Ratio")

    def compute_probability(self, dataset: TofuDataset) -> Dict[str, float]:
        def probability_fn(batch):
            input_keys = ["input_ids", "attention_mask", "labels"]
            outputs = self.model(**{k: v for k, v in batch.items() if k in input_keys})
            return probability(outputs.logits, batch["labels"])

        return self._compute_metric(dataset, probability_fn, "Probability")

    def compute_rouge(self, dataset: TofuDataset) -> Dict[str, float]:
        dataloader = self._get_dataloader(dataset)
        pad_token_id = self.tokenizer.pad_token_id
        predictions = []
        references = []

        def extract_question_tokens(batch):
            question_length = batch["question_length"]  # (batch_size,)
            input_ids = batch["input_ids"]  # (batch_size, seq_len)
            attention_mask = batch["attention_mask"]  # (batch_size, seq_len)
            batch_size, seq_len = input_ids.shape

            # Extract question tokens (right-padded)
            mask = einops.repeat(torch.arange(seq_len, device=question_length.device), 's -> b s', b=batch_size) < question_length[:, None]
            max_question_length = question_length.max().item()
            extracted_input_ids = torch.where(mask[:, :max_question_length], input_ids[:, :max_question_length], pad_token_id)
            extracted_attention_mask = torch.where(mask[:, :max_question_length], attention_mask[:, :max_question_length], 0)

            # Rotate to convert right-padded to left-padded
            rotation_amounts = max_question_length - question_length
            rotated_input_ids = torch.stack([torch.roll(seq, shift.item()) for seq, shift in zip(extracted_input_ids, rotation_amounts)])
            rotated_attention_mask = torch.stack([torch.roll(seq, shift.item()) for seq, shift in zip(extracted_attention_mask, rotation_amounts)])

            return rotated_input_ids, rotated_attention_mask

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing ROUGE"):
                batch = self._process_batch(batch)
                input_ids, attention_mask = extract_question_tokens(batch)
                labels = batch["input_ids"]

                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.max_length,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

                decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                predictions.extend(decoded_outputs)
                references.extend(decoded_labels)

        rouge_scores = rouge(predictions, references)
        return {f"rouge_{k}": v for k, v in rouge_scores.items()}

    def evaluate(self, dataset: TofuDataset) -> Dict[str, Any]:
        results = {}

        metric_functions = {
            "truth_ratio": self.compute_truth_ratio,
            "probability": self.compute_probability,
            "rouge": self.compute_rouge
        }

        for metric in self.config.metrics:
            if metric in metric_functions:
                results.update(metric_functions[metric](dataset))
            else:
                print(f"Warning: Metric '{metric}' not recognized and will be skipped.")

        return results
