import hydra
from omegaconf import DictConfig, OmegaConf

from llm_unlearning.evals import Evaluator
from llm_unlearning.models import load_model_and_tokenizer
from llm_unlearning.unlearning_datasets import TofuDataset

@hydra.main(config_path="configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    evaluator = Evaluator(model=model, tokenizer=tokenizer, config=cfg)

    # Iterate over all datasets except the base configuration
    for dataset_name, dataset_config in cfg.datasets.items():
        dataset = TofuDataset(tokenizer=tokenizer, config=dataset_config)

        print(f"Evaluating dataset: {dataset_config.name}")
        results = evaluator.evaluate(dataset=dataset)
        print(results)
        print("\n")

if __name__ == "__main__":
    main()
