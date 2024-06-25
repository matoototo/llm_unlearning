# models.py
import os
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_cfg: DictConfig):
    model_path = model_cfg.path
    tokenizer_path = model_cfg.path

    if model_cfg.tokenizer_path is not None:
        tokenizer_path = model_cfg.tokenizer_path

    # Check if the model path is a local directory
    if os.path.isdir(model_path):
        print(f"Loading model from local checkpoint: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if model_cfg.get("fp16", False) else torch.float32
        )
    else:
        print(f"Loading model from Hugging Face Hub: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if model_cfg.get("fp16", False) else torch.float32
        )

    # Load tokenizer
    if os.path.isdir(tokenizer_path):
        print(f"Loading tokenizer from local checkpoint: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    else:
        print(f"Loading tokenizer from Hugging Face Hub: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
