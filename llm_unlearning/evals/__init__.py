from .evals import *
from .tofu_evals import *

all_metrics = {
    "truth_ratio": lambda config: TruthRatio(),
    "probability": lambda config: Probability(),

    "rouge_1": lambda config: Rouge(max_length=config.max_length, rouge_type='rouge1'),
    "rouge_2": lambda config: Rouge(max_length=config.max_length, rouge_type='rouge2'),
    "rouge_l": lambda config: Rouge(max_length=config.max_length, rouge_type='rougeL'),
    "rouge_lsum": lambda config: Rouge(max_length=config.max_length, rouge_type='rougeLsum'),
}

all_aggregate_metrics = {
    "ks_test": lambda config: KSTest(),
    "model_utility": lambda config: ModelUtility()
}
