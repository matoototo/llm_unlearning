import os
import re
import json
import hydra

from typing import List, Dict, Any, Tuple
from omegaconf import DictConfig, OmegaConf

from llm_unlearning.evals import Evaluator
from llm_unlearning.models import load_model_and_tokenizer
from llm_unlearning.unlearning_datasets import TofuDataset

def get_checkpoint_paths(cfg: DictConfig) -> List[str]:
    paths = []

    # Add base model as checkpoint-0 if base_path is provided
    if cfg.model.get("base_path"): paths.append("checkpoint-0")
    # Add retain model if retain_path is provided
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

    return load_model_and_tokenizer(cfg.model)

def evaluate_group(cfg: DictConfig, group_config: Dict[str, Any], checkpoint_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    group_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for checkpoint_path in checkpoint_paths:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\nEvaluating {checkpoint_name} for group: {group_config['name']}")

        group_results[checkpoint_name] = {"metrics": {}, "aggregate_metrics": {}}

        model, tokenizer = load_model_for_evaluation(cfg, checkpoint_path)
        evaluator = Evaluator(model=model, tokenizer=tokenizer, config=group_config)

        for dataset_name, dataset_config in group_config['datasets'].items():
            dataset = TofuDataset(tokenizer=tokenizer, config=dataset_config)

            print(f"Evaluating dataset: {dataset_config['name']}")
            results = evaluator.evaluate(dataset=dataset)
            group_results[checkpoint_name]["metrics"][dataset_config['name']] = results

    retain_results = group_results.get("retain", {})

    for checkpoint_name, checkpoint_results in group_results.items():
        if checkpoint_name == "retain": continue
        group_results[checkpoint_name]["aggregate_metrics"] = evaluator.compute_aggregate_metrics(
            retain_results=retain_results,
            checkpoint_results=checkpoint_results
        )

    return group_results

@hydra.main(config_path="configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    checkpoint_paths = get_checkpoint_paths(cfg)

    for group in cfg.evaluation_groups:
        print(f"\nEvaluating group: {group.name}")

        # Create a temporary config for the group
        group_cfg = OmegaConf.create({
            "model": cfg.model,
            "batch_size": cfg.batch_size,
            "max_length": cfg.max_length,
            **group
        })

        group_results = evaluate_group(cfg, group_cfg, checkpoint_paths)

        print(f"\nEvaluation Results Summary for group {group.name}:")
        print(json.dumps(group_results, indent=2))

        if group.save_results_path:
            os.makedirs(os.path.dirname(group.save_results_path), exist_ok=True)
            with open(group.save_results_path, 'w') as f:
                json.dump(group_results, f, indent=2)
            print(f"Results for group {group.name} saved to {group.save_results_path}")

if __name__ == "__main__":
    main()
