import transformers
from omegaconf import DictConfig

def cfg_to_training_args(args_cfg: DictConfig) -> transformers.TrainingArguments:
    valid_args = {}
    # Valid parameters for TrainingArguments
    training_arg_params = set(transformers.TrainingArguments.__init__.__code__.co_varnames)
    for arg, value in args_cfg.items():
        if arg in training_arg_params:
            valid_args[arg] = value
        else:
            print(f"Warning: '{arg}' is not a valid parameter for TrainingArguments and will be ignored.")

    return transformers.TrainingArguments(**valid_args)
