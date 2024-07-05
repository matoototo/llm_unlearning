from setuptools import setup, find_packages

setup(name='llm_unlearning',
      version='0.1',
      packages=find_packages(),
      install_requires=[
          'torch',
          'transformers',
          'accelerate',
          'evaluate',
          'einops',
          'datasets',
          'hydra-core',
          'omegaconf',
          'wandb',
          'matplotlib',
          'rouge_score',
      ],
)