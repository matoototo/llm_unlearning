import hydra
import os
from omegaconf import DictConfig, OmegaConf

from llm_unlearning.unlearn import main as unlearn_main
from llm_unlearning.evaluate_model import main as evaluate_main

def run_unlearn(cfg: DictConfig) -> str:
    print("\nStarting Unlearning Process:")
    unlearn_main(cfg)
    output_dir = cfg.training_arguments.output_dir
    return output_dir

def run_evaluate(cfg: DictConfig) -> None:
    print("\nStarting Evaluation Process:")
    evaluate_main(cfg)

@hydra.main(config_path="configs", config_name="unlearn", version_base=None)
def main(cfg: DictConfig) -> None:
    unlearn_cfg = cfg

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "configs")
    eval_config_path = os.path.join(config_dir, cfg.evaluate_config)

    if not os.path.exists(eval_config_path):
        raise FileNotFoundError(f"Evaluate config file not found: {eval_config_path}")

    eval_cfg = OmegaConf.load(eval_config_path)

    unlearn_output_dir = run_unlearn(unlearn_cfg)

    if cfg.rewrite_eval_model_path:
        print(f"\nSince cfg.rewrite_eval_model_path = true, config is being rewritten to use the unlearned model at: {unlearn_output_dir}")
        eval_cfg.model.path = unlearn_output_dir
    else:
        print("\nUsing eval model path from config, use cfg.rewrite_eval_model_path = true to ensure checkpoints from the unlearning step are used.")

    run_evaluate(eval_cfg)

if __name__ == "__main__":
    main()
