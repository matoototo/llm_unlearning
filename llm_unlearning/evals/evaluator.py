from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any
import torch
from llm_unlearning.unlearning_datasets.tofu import TofuDataset
from llm_unlearning.evals.tofu_evals import Evaluation, all_metrics

class Evaluator:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.metrics = {}
        for metric in self.config.metrics:
            if metric not in all_metrics:
                print(f"Warning: Metric '{metric}' not recognized and will be skipped.")
                continue
            self.metrics[metric] = all_metrics[metric](self.config)

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

    def _compute_metric(self, dataset: TofuDataset, eval: Evaluation, desc: str) -> Dict[str, float]:
        dataloader = self._get_dataloader(dataset)
        perturb_probability = dataset.config.perturb_probability
        total_metric = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                batch = self._process_batch(batch)
                batch_metric = eval.compute(self.model, batch, self.tokenizer, perturb_probability=perturb_probability)
                total_metric += batch_metric.sum().item()
                total_samples += batch_metric.size(0)

        avg_metric = total_metric / total_samples
        return {desc.lower().replace(" ", "_"): avg_metric}

    def evaluate(self, dataset: TofuDataset) -> Dict[str, Any]:
        results = {}

        for name, eval in self.metrics.items():
            results.update(self._compute_metric(dataset, eval, name.capitalize()))

        return results
