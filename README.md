## Installation

The package with all dependencies can be installed locally with:

```bash
pip install -e .
```

This makes it an editable package, and all imports will update dynamically.


## Usage steps

There are three different operations that can be performed using this package: finetuning, unlearning, and evaluation. They are usually performed in that order, where finetuning only needs to be done once.

### 1. Finetuning
This step is needed to finetune base models on the full dataset to use for unlearning, or create models finetuned on TOFU subsets for evaluation. The latter is needed for computing the KS-test in the final evaluation (see TODO), where we compare the TOFU model unlearned on forget10 to a finetuned retain90 model.

**Inputs**: Finetuning dataset and split, config file, base model\
**Outputs**: Model finetuned on the dataset and split

```bash
python unlearn.py --config-name finetune.example.yaml # alternatively python -m llm_unlearning.unlearn ...
```

### 2. Unlearning
This is the main step. We need a model that is finetuned on the full dataset here. TOFU supplies FT versions of [Phi1.5](https://huggingface.co/locuslab/tofu_ft_phi-1.5) and [Llama2-7b](https://huggingface.co/locuslab/tofu_ft_llama2-7b), but you can also finetune a model with the previous step and use it here.

**Inputs**: Forget dataset and (optionally) retain dataset and splits, config file, finetuned model\
**Outputs**: Model unlearned on the forget dataset

```bash
python unlearn.py --config-name unlearn.example.yaml # alternatively python -m llm_unlearning.unlearn ...
```

### 3. Evaluation
We need the unlearned model from step 2 and a finetuned "reference" model from step 1. This reference model is needed to compute the KS-test for the Forget Quality metric. The reference model needs to be finetuned on the corresponding retain dataset. For example, if the unlearning was done on forget10, the reference model needs to be finetuned on retain90.

**Inputs**: Unlearning and reference models, evaluation datasets and splits, config file\
**Outputs**: Evaluation results in JSON format

```bash
python evaluate_model.py --config-name evaluate.example.yaml # alternatively python -m llm_unlearning.evaluate_model ...
```

## Combined Unlearning and Evaluation (and FT)

For convenience, there's a combined script that performs both unlearning and evaluation in sequence. This is useful when you want to immediately evaluate the results without having to manually orchestrate the process.

### Usage

To use the combined unlearning and evaluation script, run:

```bash
python unlearn_evaluate.py # alternatively python -m llm_unlearning.unlearn_evaluate
```

This script uses the config located in `llm_unlearning/configs/unlearn.yaml` by default.

### Config Requirements

Your `unlearn.yaml` config relies on these fields to enable combined unlearning and evaluation:

- `evaluate_config`: Path to the evaluation config file, relative to the `configs` directory.
- `finetune_config`: Path to the finetuning config file, relative to the `configs` directory. Needed if also finetuning.
- `rewrite_eval_model_path`: True -> update the evaluation config to use the newly unlearned model (usually should be left at `true`).
- `finetune_again`: True -> finetunes model irrespective of retain_path.

Example additions to `unlearn.yaml`:

```yaml
# For doing joint unlearning and evaluation
evaluate_config: evaluate.yaml
finetune_config: finetune.example.yaml
rewrite_eval_model_path: true
finetune_again: false
```

### Process

1. The script first performs the unlearning process using the configuration in `unlearn.yaml`.
2. If `rewrite_eval_model_path` is set to `true`, it updates the evaluation config to use the path of the newly unlearned model.
3. It then runs the evaluation process using the (potentially updated) evaluation config.

## Unlearning

The entry point for unlearning a model is `llm_unlearning.unlearn`. To start unlearning a model, simply cd into llm_unlearning and call:

```bash
python unlearn.py # alternatively python -m llm_unlearning.unlearn
```

This will by default use the config located in `llm_unlearning/configs/unlearn.yaml`. An easy way to get a starting point is to copy the example config file located in the same directory. This example config unlearns Phi1.5 on the TOFU forget10 subset using gradient difference.

### Unlearning Config

The unlearning config has the following fields:

#### base_dataset

This describes the YAML anchor of the base format of the huggingface dataset, and the question-answer format. Usually you don't need to change anything here, only exception is if you're making a new dataset or you need different question-formatting (e.g. to conform to some model-specific format).
- `path`: Path to the huggingface dataset
- `max_length`: Maximum length of the input, used for generation (eval) and truncation
- `question_start_tag`: start tag of the question prompt
- `question_end_tag`: end tag of the question prompt
- `answer_tag`: start tag of the answer prompt
- `question_key`: key of the question field
- `answer_key`: key of the answer field

#### model

Model and tokenizer paths for the model to be unlearned. Can also be a local path, in which case it should point to the checkpoint folder. Tokenizer path is optional defaults to the model path.
- `path`: Path to the model
- `tokenizer_path`: Path to the tokenizer
- `fp16`: Whether to load the model in fp16

#### dataset

The dataset(s) and their splits to be used in unlearning. The example inherits the base_dataset anchor and just specifies the split. You can specify a forget and a retain dataset, the latter of which is optional (only needed if the method requires it). Currently supported dataset names can be found in `llm_unlearning/unlearning_datasets/__init__.py`.
- `name`: Name of the dataset
- `forget`: Forget dataset
- `retain`: Retain dataset

The TOFU dataset supports non-standard (larger) splits and held-out sets for testing. The latter is activated through ":0" and ":1" split suffixes. 0 will use the first half of the split, 1 will use the second half. It's intended to treat these as validation/test splits – recommendation is to use the first half for hyperparameter tuning and the second half as a final test.


#### unlearning

Specifies a unlearning method and its parameters. All available methods can be found at the bottom of `llm_unlearning/methods/methods.py`.
- `method`: Name of the unlearning method
- `kwargs`: A dictionary of keyword arguments for the unlearning method
  - You can include scheduling configs here
- `adversarial_attack`: (Optional) Type of adversarial attack to use during unlearning

Example of unlearning configuration with parameter scheduling:

```yaml
unlearning:
    method: "npo"
    kwargs:
        beta: 0.01
        schedule_beta:
            name: "cosine"
            start_factor: 10.0
            end_factor: 1.0
            time_scale: 0.8
    adversarial_attack: "pgd"
```

In this example, the method is "NPO" with a cosine scheduled `beta` parameter:
- The initial `beta` value is set to 0.01
- `schedule_beta` defines a cosine annealing schedule for the `beta` parameter:
  - It starts at 0.1 (10 * 0.01)
  - Decreases with cosine annealing to 0.01 over 80% of the steps
  - For the final 20% of training, beta remains at 0.01

All available schedules can be found in `llm_unlearning.utils.schedules`.

#### output_dir

The base directory, results will be in a subdirectory with the current timestamp.

#### training_arguments

This supports all arguments from huggingface [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).

#### hydra

This specifies the hydra output directory and job settings, normally you don't need to change this.


## Finetuning

Finetuning uses the same entrypoint as unlearning, since it's the same operation just using gradient descent. As a result, the config files have the same structure. There is an example config under `llm_unlearning/configs/finetune.example.yaml` which uses gradient_descent to finetune a base Phi1.5 on the TOFU retain90 dataset.


## Evaluation
The entry point for evaluating a model is `llm_unlearning.evaluate_model`. To start evaluating a model, simply cd into llm_unlearning and call:
```bash
python evaluate_model.py # alternatively python -m llm_unlearning.evaluate_model
```
This will by default use the config located in `llm_unlearning/configs/evaluate.yaml`. An example config is located in the same directory. This example config evaluates Phi1.5 on various TOFU datasets using different metrics.

### Evaluation Config
The evaluation config is very similar to the unlearning config. Therefore, only the subfields that are different are listed.

#### tofu_base
This is similar to the base dataset in the unlearning config. perturb_probability specifies whether to make the calculation of the probability relative to the perturbed set. This affects the probability metric. Perturbed answers are also used for calculating the truth ratio, so a valid perturbed_answer_key is needed if truth_ratio is enabled. These are the two additional fields, rest are same as unlearning#tofu_base:
- `perturbed_answer_key`: Key of the perturbed answer field
- `perturb_probability`: Whether to use perturb probability in probability evaluation

#### model
Similar to the model in the unlearning config. Can also point to a directory with checkpoint directories, in which case it will evaluate all checkpoints in the directory. Additionally, the model specified under base_path is treated as checkpoint-0.
- `base_path`: Path to the base model (treated as checkpoint-0)
- `base_tokenizer_path`: Path to the base tokenizer
- `retain_path`: Path to the main retain model
- `retain_tokenizer_path`: Path to the main retain model's tokenizer
- `additional_retain_models`: A list of additional retain models for computing forget quality
  - Each entry should have `path` and `tokenizer_path` fields

Example configuration for additional retain models:

```yaml
model:
  # ... other model configurations ...
  retain_path: "/path/to/main/retain_model"
  retain_tokenizer_path: "/path/to/main/retain_model"
  additional_retain_models:
    - path: "/path/to/additional_retain_model_1"
      tokenizer_path: "/path/to/additional_retain_model_1"
    - path: "/path/to/additional_retain_model_2"
      tokenizer_path: "/path/to/additional_retain_model_2"
```

#### evaluation_groups
A list of evaluation groups, each containing:
- `name`: Name of the evaluation group
- `datasets`: The datasets to be used in this evaluation group (see below)
- `metrics`: A list of metrics to be used for evaluation. Available metrics are found in `llm_unlearning/evals/tofu_evals.py`. Specified in order of evaluation.
- `batch_size_factors`: A map of metric names to their batch size factors (defaulting to 1 if not specified). Used to scale batch sizes for different metrics. Currently the primary use case is to lower the truth_ratio batch size since it's memory-intensive
- `aggregate_metrics`: A list of aggregate metrics (e.g., KS-test)
- `save_results_path`: The file path where the results for this group will be saved (JSON format)

This enables specifying both forget and retain evaluation in a single config file, as shown in the example config.

#### datasets
The datasets to be used in evaluation. Multiple datasets can be specified within each evaluation group, and they will be evaluated in order. Each dataset inherits from the tofu_base anchor.
- `name`: Name of the dataset
- `split`: Split of the dataset to use
- `perturb_probability`: Whether to use perturb probability (overrides tofu_base setting)

#### batch_size
The batch size to use during evaluation.

#### max_length
The maximum length for generation during evaluation (only relevant for ROUGE).

### Additional Retain Models

The evaluation script supports using multiple retain models for computing forget quality. This feature allows for a more robust evaluation of the unlearning process.

#### Configuration
To use additional retain models:
1. In the `model` section of the evaluation config, add an `additional_retain_models` list.
2. Each entry in this list should be a dictionary with `path` and `tokenizer_path` fields, pointing to the model and its tokenizer.

#### Evaluation Process
When additional retain models are specified:
1. The script evaluates these models only on the forget dataset.
2. For each checkpoint being evaluated, the script computes FQ using each of the additional retain models, as well as the main retain model.
3. The results include individual scores for each retain model and summary statistics (mean and standard deviation) across all retain models.

## Evaluating WMDP and MMLU

To evaluate WMDP and MMLU, use the LM evaluation harness. See [here](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install) for installation. Usage example:

```bash
lm_eval --model hf \
    --model_args pretrained=path \
    --tasks wmdp,mmlu \
    --batch_size=auto
```

There is also a Python script in `llm_unlearning/utils/lm_eval.py` that can be used to run the harness on all model checkpoints under a specific root. It saves the extracted results tables under parent(checkpoint_dir)/results/lm_eval_{checkpoint}.md.

## Plotting

Evaluation results can be plotted using `llm_unlearning.utils.plotting` or `llm_unlearning.utils.plotting_joint`.

The former is for plotting a single evaluation file (such as retain_results.json), and it plots all metrics broken down per dataset and checkpoint. More information can be found by running:
```bash
python -m llm_unlearning.utils.plotting --help
```

The latter plots model utility (x-axis) vs forget quality (y-axis) for multiple evaluation files. It operates on a folder of evaluation files, and assumes the following structure:
```
input_folder/
├── agg_run_1_name/     # Has runs to aggregate (e.g., different seeds)
│   ├── arbitrary_name/ # Contains retain/forget pair for a single run
│   │   ├── retain_results.json
│   │   └── forget_results.json
│   ├── arbitrary_name_2/
│   │   ├── retain_results.json
│   │   └── forget_results.json
│   └── ...
├── agg_run_2_name/
│   ├── arbitrary_name_3/
│   │   ├── retain_results.json
│   │   └── forget_results.json
│   └── ...
├── # Single runs are prefixed with their name
├── run_3_name_retain_results.json
├── run_3_name_forget_results.json
├── run_4_name_retain_results.json
└── run_4_name_forget_results.json
```
More information:
```bash
python -m llm_unlearning.utils.plotting_joint --help
```

## Possible issues

### Loading Llama-3.1-8B-Instruct for UnlearningCoherency

`ValueError: rope_scaling must be a dictionary with two fields, type and factor`

This is due to an outdated `transformers` version, see [here](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/discussions/15). To fix, simply:

```bash
pip install --upgrade transformers
```
