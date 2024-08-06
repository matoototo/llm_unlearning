import hydra
import wandb

from omegaconf import DictConfig, OmegaConf

from llm_unlearning.utils import cfg_to_training_args
from llm_unlearning.models import load_model_and_tokenizer
from llm_unlearning.trainer import UnlearningTrainer
from llm_unlearning.unlearning_datasets import load_unlearning_dataset

@hydra.main(config_path="configs", config_name="unlearn", version_base=None)
def main(cfg: DictConfig) -> None:
    wandb.init(
        project="llm-unlearning",
        config=OmegaConf.to_container(cfg, resolve=True),
        group=cfg.get("group", None),
        job_type="unlearn",
        reinit=True,
    )
    print(OmegaConf.to_yaml(cfg))

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    reference_model, _ = load_model_and_tokenizer(cfg.reference_model) if cfg.get("reference_model") else (None, None)
    dataset, collate_fn = load_unlearning_dataset(cfg.dataset, tokenizer)
    training_args = cfg_to_training_args(cfg.training_arguments)

    trainer = UnlearningTrainer(
        model=model,
        reference_model=reference_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        method=cfg.unlearning.method,
        unlearning_kwargs=cfg.unlearning.get("kwargs", {}),
        adversarial_attack=cfg.unlearning.get("adversarial_attack", None),
        attack_kwargs=cfg.unlearning.get("attack_kwargs", {}),
    )

    trainer.train()

    wandb.finish()

if __name__ == "__main__":
    main()
