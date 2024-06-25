### Installation

The package with all dependencies can be installed locally with:

```bash
pip install -e .
```


### Unlearning

The entry point for unlearning a model is `llm_unlearning.unlearn`. To start unlearning a model, simply cd into llm_unlearning and call:

```bash
python unlearn.py
```

This will by default use the config located in `llm_unlearning/configs/unlearn.yaml`. There is an example config file in the same directory, which unlearns a Phi1.5 model on the TOFU forget10 subset using [gradient difference](https://github.com/matoototo/llm_unlearning/blob/6299fe4d4994a324bb7d3957dbe93c08ffa551ce/llm_unlearning/methods/methods.py#L36).

See the [config section](#config) for more information.

### Evaluation

TODO...

### Config system

The config  uses [Hydra](https://hydra.cc/docs/intro) to manage the configuration. The default assumes that the config is located in `llm_unlearning/configs/unlearn.yaml`. There is an example config in `llm_unlearning/configs/unlearn.example.yaml`.

TODO...
