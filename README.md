// llm_unlearning/README.md

## Installation

The package with all dependencies can be installed locally with:

```bash
pip install -e .
```

This makes it an editable package, and all imports will update dynamically.


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
