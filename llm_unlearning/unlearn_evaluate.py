import hydra
import os
import gc
import glob
import torch
from omegaconf import DictConfig, OmegaConf

from llm_unlearning.unlearn import main as unlearn_main
from llm_unlearning.evaluate_model import main as evaluate_main

def flush_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()

def run_unlearn(cfg: DictConfig) -> str:
    print("\nStarting Unlearning Process:")
    unlearn_main(cfg)
    output_dir = cfg.training_arguments.output_dir

    flush_cache()
    return output_dir

def run_finetune(cfg: DictConfig) -> str:
    print("\nStarting Finetuning Process:")
    unlearn_main(cfg)
    output_dir = cfg.training_arguments.output_dir

    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {output_dir}")
    last_checkpoint = max(checkpoints, key=os.path.getctime)

    flush_cache()
    return last_checkpoint

def run_evaluate(cfg: DictConfig) -> None:
    print("\nStarting Evaluation Process:")
    evaluate_main(cfg)
    flush_cache()

@hydra.main(config_path="configs", config_name="unlearn", version_base=None)
def main(cfg: DictConfig) -> None:
    unlearn_cfg = cfg
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "configs")
    eval_config_path = os.path.join(config_dir, cfg.evaluate_config)

    if not os.path.exists(eval_config_path):
        raise FileNotFoundError(f"Evaluate config file not found: {eval_config_path}")

    eval_cfg = OmegaConf.load(eval_config_path)

    # Check if we need to finetune
    finetune_needed = (
        'retain_path' not in eval_cfg.model or
        not eval_cfg.model.retain_path or
        cfg.get('finetune_again', False)
    )

    if finetune_needed:
        print("\nRetain model not found or finetune_again is set. Starting finetuning process.")
        finetune_cfg = OmegaConf.load(os.path.join(config_dir, cfg.finetune_config))
        retain_model_path = run_finetune(finetune_cfg)
        eval_cfg.model.retain_path = retain_model_path
        eval_cfg.model.retain_tokenizer_path = retain_model_path
        print(f"\nFinetuning complete. Retain model path: {retain_model_path}")
    else:
        print(f"\nUsing existing retain model for evaluation from path: {eval_cfg.model.retain_path}. Use cfg.finetune_again = true to force finetuning.")

    unlearn_output_dir = run_unlearn(unlearn_cfg)

    if cfg.rewrite_eval_model_path:
        print(f"\nSince cfg.rewrite_eval_model_path = true, config is being rewritten to use the unlearned model at: {unlearn_output_dir}")
        eval_cfg.model.path = unlearn_output_dir
    else:
        print("\nUsing eval model path from config, use cfg.rewrite_eval_model_path = true to ensure checkpoints from the unlearning step are used.")

    run_evaluate(eval_cfg)

if __name__ == "__main__":
    main()
