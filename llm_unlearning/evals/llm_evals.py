import os
import re
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Tuple, List

from llm_unlearning.evals import Evaluation
from llm_unlearning.evals.utils import generate_and_extract
from llm_unlearning.evals.prompts import prompts

class UnlearningCoherency(Evaluation):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.prompt_name = config.get('prompt_name', 'unlearning_coherency')
        self.prompt_template = prompts[self.prompt_name]

        self.max_length = config.get('max_length', 512)
        self.model_backend = config.get('model_backend', 'openai').lower()
        if self.model_backend == 'openai':
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            openai.api_key = self.api_key
            self.openai_model = config.get('openai_model', 'gpt-4o-mini')
        elif self.model_backend == 'huggingface':
            raise NotImplementedError("HuggingFace model backend not yet supported.")
        else:
            raise ValueError(f"Unsupported model backend: {self.model_backend}")

    def compute(self, model, batch: Dict[str, Any], tokenizer=None, **kwargs) -> Dict[str, torch.Tensor]:
        if self.model_backend == 'huggingface' and model is None:
            raise ValueError("HuggingFace model must be provided for HuggingFace backend.")
        device = model.device if model else torch.device('cpu')

        question_texts, decoded_labels, decoded_outputs = generate_and_extract(
            model=model,
            batch=batch,
            tokenizer=tokenizer,
            max_length=self.max_length,
            device=device
        )

        scores = []
        for q, gt, ga in zip(question_texts, decoded_labels, decoded_outputs):
            prompt = self.prompt_template.format(question=q, ground_truth=gt, generated_answer=ga)
            messages = [{"role": "user", "content": prompt}]
            response = self._get_llm_response(messages)
            unlearning_score, coherency_score = self._parse_response(response)
            scores.append([unlearning_score, coherency_score])

        return {
            "unlearning": torch.tensor([s[0] for s in scores], dtype=torch.float32),
            "coherency": torch.tensor([s[1] for s in scores], dtype=torch.float32)
        }

    def _get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        if self.model_backend == 'openai':
            try:
                response = openai.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=150,
                    temperature=0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI API error: {e}")
                return ""
        elif self.model_backend == 'huggingface':
            raise NotImplementedError("HuggingFace model backend not yet supported.")
        else:
            raise ValueError(f"Unsupported model backend: {self.model_backend}")

    def _parse_response(self, response: str) -> Tuple[int, int]:
        unlearning_score = 0
        coherency_score = 0

        unlearning_pattern = re.compile(r'Unlearning Score:\s*(\d)', re.IGNORECASE)
        coherency_pattern = re.compile(r'Coherency Score:\s*(\d)', re.IGNORECASE)

        unlearning_match = unlearning_pattern.search(response)
        coherency_match = coherency_pattern.search(response)

        if unlearning_match:
            unlearning_score = int(unlearning_match.group(1))
        if coherency_match:
            coherency_score = int(coherency_match.group(1))

        return unlearning_score, coherency_score
