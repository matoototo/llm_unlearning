import hydra
from omegaconf import DictConfig, OmegaConf

from llm_unlearning.unlearning_datasets import load_unlearning_dataset
from llm_unlearning.models import load_model_and_tokenizer

@hydra.main(config_path="configs", config_name="unlearn", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dataset = load_unlearning_dataset(cfg.dataset)
    model, tokenizer = load_model_and_tokenizer(cfg.model)

    print(dataset)
    print(model)

if __name__ == "__main__":
    main()
