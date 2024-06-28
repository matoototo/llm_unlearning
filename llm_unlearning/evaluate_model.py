import os
import re
import json
import hydra

from typing import List, Dict, Any
from omegaconf import DictConfig, OmegaConf

from llm_unlearning.evals import Evaluator
from llm_unlearning.models import load_model_and_tokenizer
from llm_unlearning.unlearning_datasets import TofuDataset

def get_checkpoint_paths(path: str) -> List[str]:
    if not os.path.isdir(path):
        raise ValueError(f"The path {path} is not a directory")

    if re.match(r'checkpoint-\d+$', os.path.basename(path)):
        return [path]

    checkpoint_dirs = [
        os.path.join(path, d) for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and re.match(r'checkpoint-\d+$', d)
    ]

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {path}")

    # Sort checkpoint directories by creation time
    checkpoint_dirs.sort(key=lambda x: os.path.getctime(x))

    return checkpoint_dirs

@hydra.main(config_path="configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    checkpoint_paths = get_checkpoint_paths(cfg.model.path)
    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for checkpoint_path in checkpoint_paths:
        print(f"\nEvaluating checkpoint: {checkpoint_path}")

        checkpoint_name = os.path.basename(checkpoint_path)
        all_results[checkpoint_name] = {}

        # Update the model path in the config
        cfg.model.path = checkpoint_path
        cfg.model.tokenizer_path = checkpoint_path

        model, tokenizer = load_model_and_tokenizer(cfg.model)
        evaluator = Evaluator(model=model, tokenizer=tokenizer, config=cfg)

        # Iterate over all datasets except the base configuration
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
