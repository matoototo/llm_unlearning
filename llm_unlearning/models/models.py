import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_cfg: DictConfig):
    model_path = model_cfg.path
    tokenizer_path = model_cfg.path

    if model_cfg.tokenizer_path is not None: tokenizer_path = model_cfg.tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
