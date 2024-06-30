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
    if cfg.model.base_path: paths.append("checkpoint-0")

    if not os.path.isdir(cfg.model.path):
        if re.match(r'checkpoint-\d+$', os.path.basename(cfg.model.path)):
            paths.append(cfg.model.path)
        else:
            raise ValueError(f"The path {cfg.model.path} is not a valid checkpoint directory")
    else:
        checkpoint_dirs = [
            os.path.join(cfg.model.path, d) for d in os.listdir(cfg.model.path)
            if os.path.isdir(os.path.join(cfg.model.path, d)) and re.match(r'checkpoint-\d+$', d)
        ]

        if not checkpoint_dirs:
            raise ValueError(f"No checkpoint directories found in {cfg.model.path}")

        # Sort checkpoint directories by number
        checkpoint_dirs.sort(key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
        paths.extend(checkpoint_dirs)

    return paths

def load_model_for_evaluation(cfg: DictConfig, checkpoint_path: str) -> Tuple[Any, Any]:
    if checkpoint_path == "checkpoint-0":
        cfg.model.path = cfg.model.base_path
        cfg.model.tokenizer_path = cfg.model.base_tokenizer_path
    else:
        cfg.model.path = checkpoint_path
        cfg.model.tokenizer_path = checkpoint_path

    return load_model_and_tokenizer(cfg.model)

@hydra.main(config_path="configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    checkpoint_paths = get_checkpoint_paths(cfg)
    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for checkpoint_path in checkpoint_paths:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\nEvaluating {checkpoint_name}")

        all_results[checkpoint_name] = {}

        model, tokenizer = load_model_for_evaluation(cfg, checkpoint_path)
        evaluator = Evaluator(model=model, tokenizer=tokenizer, config=cfg)

        for dataset_name, dataset_config in cfg.datasets.items():
            dataset = TofuDataset(tokenizer=tokenizer, config=dataset_config)

            print(f"Evaluating dataset: {dataset_config.name}")
            results = evaluator.evaluate(dataset=dataset)
            all_results[checkpoint_name][dataset_config.name] = results

    print("\nEvaluation Results Summary:")
    print(json.dumps(all_results, indent=2))

    if cfg.save_results_path:
        os.makedirs(os.path.dirname(cfg.save_results_path), exist_ok=True)
        with open(cfg.save_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {cfg.save_results_path}")

if __name__ == "__main__":
    main()
