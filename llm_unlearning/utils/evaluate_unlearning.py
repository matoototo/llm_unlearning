import argparse
import math
import os
import csv
import yaml
import json
from typing import List, Dict
from omegaconf import DictConfig
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from llm_unlearning.evals.llm_evals import UnlearningCoherency
from llm_unlearning.unlearning_datasets import TofuDataset
from llm_unlearning.models import load_model_and_tokenizer

def find_checkpoints(root_dir: str) -> List[str]:
    checkpoints = []
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            if dir.startswith("checkpoint-"):
                checkpoints.append(os.path.join(root, dir))
    return sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

def load_config(config_path: str) -> DictConfig:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return DictConfig(config)

def trim(batch: Dict[str, torch.tensor], space_left: int):
    batch = { k: v[:space_left] for k, v in batch.items() }
    return batch, space_left - batch["input_ids"].size(0)

def save_checkpoint_csv(checkpoint: str, results: List[Dict], results_folder: str):
    os.makedirs(os.path.join(results_folder, 'csv'), exist_ok=True)
    checkpoint_name = "_".join(checkpoint.split("/")[-2:])
    csv_path = os.path.join(results_folder, 'csv', f'{checkpoint_name}.csv')

    fieldnames = ['Question', 'Generated Answer', 'Ground Truth Answer', 'Unlearning', 'Coherency']

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for item in results:
            writer.writerow({
                'Question': item['question'].replace('Question: ', ''),
                'Generated Answer': item['generated_answer'],
                'Ground Truth Answer': item['ground_truth'],
                'Unlearning': item['unlearning_score'],
                'Coherency': item['coherency_score']
            })

def json_to_csvs(json_path: str, output_folder: str):
    with open(json_path, 'r') as f:
        results = json.load(f)

    os.makedirs(os.path.join(output_folder, 'csv'), exist_ok=True)

    for checkpoint, checkpoint_results in results.items():
        checkpoint_name = os.path.basename(checkpoint)
        save_checkpoint_csv(checkpoint, checkpoint_results['metadata'], output_folder)

def evaluate_checkpoints(input_dir: str, dataset_config: DictConfig, model_name: str, model_backend: str, first_n: int, results_folder: str):
    os.makedirs(results_folder, exist_ok=True)
    json_path = os.path.join(results_folder, 'results.json')

    checkpoints = find_checkpoints(input_dir)
    print(f"Found {len(checkpoints)} checkpoints")

    eval_config = {
        "model_backend": model_backend,
        "openai_model": model_name if model_backend == "openai" else None,
        "model_name": model_name if model_backend == "huggingface" else None,
    }
    unlearning_coherency = UnlearningCoherency(eval_config)

    results = {}

    for checkpoint in checkpoints:
        print(f"\nEvaluating checkpoint: {checkpoint}")

        model, tokenizer = load_model_and_tokenizer(DictConfig({
            "path": checkpoint,
            "tokenizer_path": checkpoint,
            "fp16": True
        }))
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        batch_size = 32
        dataset = TofuDataset(tokenizer=tokenizer, config=dataset_config)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=TofuDataset.collate_fn)

        checkpoint_results = []
        space_left = first_n

        for batch in tqdm(dataloader, desc="Processing batches", total=min(len(dataloader), math.ceil(first_n / batch_size))):
            input_keys = ['input_ids', 'labels', 'attention_mask', 'question_length']
            batch, space_left = trim({k: v.to(device) for k, v in batch.items() if k in input_keys}, space_left)
            if batch['input_ids'].size(0) == 0: break
            results_batch = unlearning_coherency.compute(model, batch, tokenizer)
            checkpoint_results.extend(results_batch['metadata'])

        checkpoint_results = {
            "metadata": checkpoint_results,
            "unlearning": sum(map(lambda x : x["unlearning_score"], checkpoint_results)) / len(checkpoint_results),
            "coherency": sum(map(lambda x : x["coherency_score"], checkpoint_results)) / len(checkpoint_results)
        }

        results[f"{checkpoint}_{model_name}"] = checkpoint_results

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        save_checkpoint_csv(checkpoint, checkpoint_results['metadata'], results_folder)

        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates all checkpoints given a tree root using UnlearningCoherency")
    parser.add_argument("input_dir", help="Tree root containing checkpoints (not necessarily as immediate children)")
    parser.add_argument("dataset_config", help="Evaluation config containing the dataset config under the forget_evaluation name")
    parser.add_argument("--results_folder", default="unlearning_coherency_results", help="Folder to save results")
    parser.add_argument("--first_n", default=1e9, help="Evaluates only the first N dataset items", type=int)
    parser.add_argument("--model_name", default="gpt-4o-2024-08-06", help="Model name or path")
    parser.add_argument("--model_backend", default="openai", choices=["openai", "huggingface"], help="Model backend to use")
    parser.add_argument("--convert_json", help="Convert an existing JSON file to CSVs")

    args = parser.parse_args()

    if args.convert_json:
        json_to_csvs(args.convert_json, args.results_folder)
    else:
        dataset_config = load_config(args.dataset_config)
        forget_eval_config = next((group for group in dataset_config['evaluation_groups'] if group['name'] == 'forget_evaluation'), None)
        if not forget_eval_config:
            raise ValueError("Could not find 'forget_evaluation' in the dataset config")

        tofu_forget_config = forget_eval_config['datasets']['tofu_forget']
        evaluate_checkpoints(args.input_dir, tofu_forget_config, args.model_name, args.model_backend, args.first_n, args.results_folder)
