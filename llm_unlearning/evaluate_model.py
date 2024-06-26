import hydra
from omegaconf import DictConfig, OmegaConf

from llm_unlearning.evals import Evaluator
from llm_unlearning.models import load_model_and_tokenizer
from llm_unlearning.unlearning_datasets import TofuDataset

@hydra.main(config_path="configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    dataset = TofuDataset(tokenizer=tokenizer, config=cfg.dataset)

    evaluator = Evaluator(model=model, tokenizer=tokenizer, config=cfg)

    print(evaluator.evaluate(dataset=dataset))

if __name__ == "__main__":
    main()
