import os
import re
import json
import hydra
import torch
import shutil
import wandb

from typing import List, Dict, Any, Tuple
from omegaconf import DictConfig, OmegaConf

from llm_unlearning.evals.evaluator import Evaluator
from llm_unlearning.unlearning_datasets import TofuDataset
from llm_unlearning.methods import EmbeddingBoundary
from llm_unlearning.models import load_model_and_tokenizer, EmbeddingRemappingModelWrapper, LogitMaskingModelWrapper

def get_checkpoint_paths(cfg: DictConfig) -> List[str]:
    paths = []

    if cfg.model.get("base_path"): paths.append("checkpoint-0")
    if cfg.model.get("retain_path"): paths.append("retain")

    normalised_path = os.path.normpath(cfg.model.path)
    if re.match(r'checkpoint-\d+$', os.path.basename(normalised_path)):
        paths.append(normalised_path)
        return paths

    checkpoint_dirs = [
        os.path.join(normalised_path, d) for d in os.listdir(normalised_path)
        if os.path.isdir(os.path.join(normalised_path, d)) and re.match(r'checkpoint-\d+$', d)
    ]

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {normalised_path}")

    # Sort checkpoint directories by number
    checkpoint_dirs.sort(key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
    paths.extend(checkpoint_dirs)

    return paths

def load_model_for_evaluation(cfg: DictConfig, checkpoint_path: str) -> Tuple[Any, Any]:
    if checkpoint_path == "checkpoint-0":
        cfg.model.path = cfg.model.base_path
        cfg.model.tokenizer_path = cfg.model.base_tokenizer_path
    elif checkpoint_path == "retain":
        cfg.model.path = cfg.model.retain_path
        cfg.model.tokenizer_path = cfg.model.retain_tokenizer_path
    else:
        cfg.model.path = checkpoint_path
        cfg.model.tokenizer_path = checkpoint_path

    model, tokenizer = load_model_and_tokenizer(cfg.model)

    wrapper_config = cfg.get("wrapper", {})
    if wrapper_config and os.path.exists(os.path.join(cfg.model.path, "embedding_boundaries.pt")):
        print("Loading embedding boundaries, wrapping model with EmbeddingRemappingModelWrapper")
        embedding_boundary_config = EmbeddingBoundary.load_config(cfg.model.path)
        embedding_boundary = EmbeddingBoundary(**embedding_boundary_config)
        embedding_boundary.boundaries = EmbeddingBoundary.load_boundaries(cfg.model.path)
        if wrapper_config.get("name") == "logit_masking":
            model = LogitMaskingModelWrapper(model, embedding_boundary, **wrapper_config.get("kwargs", {}))
        elif wrapper_config.get("name") == "embedding_remapping":
            model = EmbeddingRemappingModelWrapper(model, embedding_boundary)
        else:
            raise ValueError(f"Unknown model wrapper: {wrapper_config.get('name')}")

    return model, tokenizer

def evaluate_checkpoint(model: Any, tokenizer: Any, evaluation_groups: List[Dict[str, Any]], cfg: DictConfig) -> Dict[str, Dict[str, Any]]:
    checkpoint_results = {}

    for group in evaluation_groups:
        print(f"\nEvaluating group: {group['name']}")
        group_cfg = OmegaConf.create({
            "model": cfg.model,
            "batch_size": cfg.batch_size,
            "max_length": cfg.max_length,
            **group
        })

        evaluator = Evaluator(model=model, tokenizer=tokenizer, config=group_cfg, group_name=group['name'])
        group_results = {"metrics": {}, "aggregate_metrics": {}}

        for dataset_name, dataset_config in group_cfg['datasets'].items():
            dataset = TofuDataset(tokenizer=tokenizer, config=dataset_config)
            print(f"Evaluating dataset: {dataset_config['name']}")
            results = evaluator.evaluate(dataset=dataset)
            group_results["metrics"][dataset_config['name']] = results

        checkpoint_results[group['name']] = group_results

    if hasattr(model, 'hook') and hasattr(model.hook, 'total_count'):
        checkpoint_results["boundary_stats"] = {
            "total_count": model.hook.total_count,
            "inside_boundary_count": model.hook.inside_boundary_count
        }

    return checkpoint_results

def delete_checkpoint(checkpoint_path: str):
    """Delete the checkpoint directory."""
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
        print(f"Deleted checkpoint: {checkpoint_path}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")

def evaluate_with_config(cfg: DictConfig, wandb_run: wandb.sdk.wandb_run.Run = None) -> None:
    print(OmegaConf.to_yaml(cfg))

    checkpoint_paths = get_checkpoint_paths(cfg)
    all_results = {}
    wandb_log_data = {}

    for checkpoint_path in checkpoint_paths:
        checkpoint_name = os.path.basename(checkpoint_path)
        checkpoint_number = int(checkpoint_name.split('-')[-1]) if checkpoint_name != "retain" else -1
        print(f"\nEvaluating {checkpoint_name}")

        model, tokenizer = load_model_for_evaluation(cfg, checkpoint_path)
        checkpoint_results = evaluate_checkpoint(model, tokenizer, cfg.evaluation_groups, cfg)
        all_results[checkpoint_name] = checkpoint_results

        for group_name, group_results in checkpoint_results.items():
            for dataset_name, dataset_metrics in group_results['metrics'].items():
                for metric_name, metric_value in dataset_metrics.items():
                    if not metric_name.endswith('_metadata'):
                        metric_key = f"{group_name}/{dataset_name}/{metric_name}"
                        if metric_key not in wandb_log_data:
                            wandb_log_data[metric_key] = {}
                        wandb_log_data[metric_key][checkpoint_number] = metric_value

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cfg.get('delete_after_eval', False) and checkpoint_name not in ["checkpoint-0", "retain"]:
            delete_checkpoint(checkpoint_path)

    retain_results = all_results.get("retain", {})
    for checkpoint_name, checkpoint_results in all_results.items():
        if checkpoint_name == "retain":
            continue
        checkpoint_number = int(checkpoint_name.split('-')[-1])
        for group in cfg.evaluation_groups:
            group_cfg = OmegaConf.create({
                "model": cfg.model,
                "batch_size": cfg.batch_size,
                "max_length": cfg.max_length,
                **group
            })
            evaluator = Evaluator(model=None, tokenizer=None, config=group_cfg)
            aggregate_metrics = evaluator.compute_aggregate_metrics(
                retain_results=retain_results.get(group['name'], {}),
                checkpoint_results=checkpoint_results[group['name']]
            )
            all_results[checkpoint_name][group['name']]["aggregate_metrics"] = aggregate_metrics

            for metric_name, metric_value in aggregate_metrics.items():
                metric_key = f"{group['name']}/aggregate/{metric_name}"
                if metric_key not in wandb_log_data:
                    wandb_log_data[metric_key] = {}
                wandb_log_data[metric_key][checkpoint_number] = metric_value

    if wandb_run:
        for metric_key, checkpoint_values in wandb_log_data.items():
            data: List[Tuple[int, float]] = sorted(checkpoint_values.items())
            table = wandb.Table(data=data, columns=["checkpoint", "value"])
            wandb_run.log({metric_key: wandb.plot.line(
                table,
                x="checkpoint",
                y="value",
                title=metric_key
            )})

    for group in cfg.evaluation_groups:
        if group.get('save_results_path'):
            group_results = {checkpoint: results[group['name']] for checkpoint, results in all_results.items()}
            os.makedirs(os.path.dirname(group['save_results_path']), exist_ok=True)
            with open(group['save_results_path'], 'w') as f:
                json.dump(group_results, f, indent=2)
            print(f"Results for group {group['name']} saved to {group['save_results_path']}")

    print("\nEvaluation Results Summary:")
    print(json.dumps(all_results, indent=2))

@hydra.main(config_path="configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    evaluate_with_config(cfg)

if __name__ == "__main__":
    main()
