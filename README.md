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

## Combined Unlearning and Evaluation

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
- `rewrite_eval_model_path`: Boolean flag to determine whether to automatically update the evaluation config to use the newly unlearned model (usually shuold be left at `true`).

Example additions to `unlearn.yaml`:

```yaml
# For doing joint unlearning and evaluation
evaluate_config: evaluate.yaml
rewrite_eval_model_path: true
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

#### unlearning

Currently only has one field, which identifies which unlearning method to use. All available methods can be found at the bottom of `llm_unlearning/methods/methods.py`.
- `method`: Name of the unlearning method

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

#### datasets
The datasets to be used in evaluation. Multiple datasets can be specified, and they will be evaluated in order. Each dataset inherits from the tofu_base anchor.
- `name`: Name of the dataset
- `split`: Split of the dataset to use
- `perturb_probability`: Whether to use perturb probability (overrides tofu_base setting)

#### metrics
A list of metrics to be used for evaluation. Available metrics can be found at the bottom of `llm_unlearning/evals/tofu_evals.py`. They are specified in order of evaluation.

#### batch_size
The batch size to use during evaluation.

#### max_length
The maximum length for generation during evaluation (only relevant for ROUGE).

#### save_results_path
The file path where the results will be saved, JSON format.


## Plotting

The evaluation results can be plotted using `llm_unlearning.utils.plotting`. More information is given by invoking:
```bash
python -m llm_unlearning.utils.plotting --help
```
