from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig

import torch

from llm_unlearning.unlearning_datasets.tofu import TofuDataset
from llm_unlearning.evals import Evaluation, all_metrics, all_aggregate_metrics

class Evaluator:
    def __init__(self, model, tokenizer, config, group_name=None):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.group_name = group_name
        if model: self.model.to(self.device)
        if model: self.model.eval()
        if model: self.model.group = group_name
        self.metrics = {}
        self.batch_size_factors = config.get("batch_size_factors", {})

        for metric_entry in self.config.metrics:
            if isinstance(metric_entry, str):
                metric_name = metric_entry
                metric_config = {}
            elif isinstance(metric_entry, DictConfig):
                metric_name = metric_entry['name']
                metric_config = metric_entry.get('config', {})
            else:
                raise ValueError(f"Invalid metric entry: {metric_entry}")

            if metric_name not in all_metrics:
                print(f"Warning: Metric '{metric_name}' not recognized and will be skipped.")
                continue

            merged_config = OmegaConf.merge(self.config, OmegaConf.create(metric_config))
            self.metrics[metric_name] = all_metrics[metric_name](merged_config)

        self.aggregate_metrics = {}
        for metric in self.config.get("aggregate_metrics", []):
            if metric not in all_aggregate_metrics:
                print(f"Warning: Aggregate metric '{metric}' not recognized and will be skipped.")
                continue
            self.aggregate_metrics[metric] = all_aggregate_metrics[metric](self.config)

    def _get_dataloader(self, dataset: TofuDataset, batch_size_factor: float = 1.0) -> DataLoader:
        adjusted_batch_size = max(1, int(self.config.batch_size * batch_size_factor))
        return DataLoader(
            dataset,
            batch_size=adjusted_batch_size,
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

    def _compute_metric(self, dataset: TofuDataset, eval: Evaluation, desc: str) -> Dict[str, Any]:
        batch_size_factor = self.batch_size_factors.get(desc.lower(), 1.0)
        dataloader = self._get_dataloader(dataset, batch_size_factor)
        perturb_probability = dataset.config.perturb_probability
        total_metrics = {}
        total_samples = 0
        eval_scores = {}
        per_item_data = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                batch = self._process_batch(batch)
                batch_metric = eval.compute(self.model, batch, self.tokenizer, perturb_probability=perturb_probability)

                if isinstance(batch_metric, dict):
                    if 'metadata' in batch_metric:
                        per_item_data.extend(batch_metric['metadata'])
                        del batch_metric['metadata']

                    for key, value in batch_metric.items():
                        if key not in total_metrics:
                            total_metrics[key] = 0.0
                            eval_scores[key] = []
                        total_metrics[key] += value.sum().item()
                        eval_scores[key].extend(value.cpu().numpy().tolist())
                else:
                    if 'default' not in total_metrics:
                        total_metrics['default'] = 0.0
                        eval_scores['default'] = []
                    total_metrics['default'] += batch_metric.sum().item()
                    eval_scores['default'].extend(batch_metric.cpu().numpy().tolist())

                total_samples += batch_metric.size(0) if isinstance(batch_metric, torch.Tensor) else len(next(iter(batch_metric.values())))

        results = {}
        name = desc.lower().replace(" ", "_")

        for key in total_metrics:
            avg_metric = total_metrics[key] / total_samples
            metric_name = f"{name}_{key}" if key != 'default' else name
            results[metric_name] = avg_metric
            results[f"{metric_name}_metadata"] = eval_scores[key]

        if per_item_data: results['per_item_data'] = per_item_data

        return results

    def evaluate(self, dataset: TofuDataset) -> Dict[str, Any]:
        results = {}

        for name, eval in self.metrics.items():
            results.update(self._compute_metric(dataset, eval, name.capitalize()))

        return results

    def compute_aggregate_metrics(self, retain_results: Dict[str, Dict[str, Any]], checkpoint_results: Dict[str, Any]) -> Dict[str, Any]:
        aggregate_results = {}

        for name, metric in self.aggregate_metrics.items():
            aggregate_results[name] = metric.compute(retain_results, checkpoint_results).item()

        return aggregate_results
