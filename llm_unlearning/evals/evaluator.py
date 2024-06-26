import torch
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
            gt_outputs = self.model(**{k: v for k, v in batch.items() if not k.startswith("perturbed_")})
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
            outputs = self.model(**{k: v for k, v in batch.items() if not k.startswith("perturbed_")})
            return probability(outputs.logits, batch["labels"])

        return self._compute_metric(dataset, probability_fn, "Probability")

    def compute_rouge(self, dataset: TofuDataset) -> Dict[str, float]:
        dataloader = self._get_dataloader(dataset)
        predictions = []
        references = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing ROUGE"):
                batch = self._process_batch(batch)

                outputs = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=self.config.max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2
                )

                decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

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
