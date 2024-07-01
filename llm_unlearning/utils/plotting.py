import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def harmonic_mean(arr):
    return len(arr) / np.sum(1.0 / np.array(arr))

def gather_metrics(data, excluded_metrics=None):
    if excluded_metrics is None:
        excluded_metrics = []

    checkpoints = list(data.keys())
    datasets = list(data[checkpoints[0]].keys())
    metrics = [m for m in list(data[checkpoints[0]][datasets[0]].keys()) if m not in excluded_metrics]

    gathered_data = {
        "checkpoints": checkpoints,
        "datasets": datasets,
        "metrics": metrics,
        "metric_data": {dataset: {metric: [] for metric in metrics} for dataset in datasets},
        "harmonic_means": {dataset: [] for dataset in datasets},
        "overall_harmonic_means": []
    }

    for checkpoint in checkpoints:
        checkpoint_metrics = []
        for dataset in datasets:
            dataset_metrics = data[checkpoint].get(dataset, {})
            for metric in metrics:
                value = dataset_metrics.get(metric, np.nan)
                gathered_data["metric_data"][dataset][metric].append(value)
                if not np.isnan(value):
                    checkpoint_metrics.append(value)

            dataset_harmonic_mean = harmonic_mean([v for k, v in dataset_metrics.items() if k not in excluded_metrics and not np.isnan(v)])
            gathered_data["harmonic_means"][dataset].append(dataset_harmonic_mean)

        overall_harmonic_mean = harmonic_mean([m for m in checkpoint_metrics if not np.isnan(m)])
        gathered_data["overall_harmonic_means"].append(overall_harmonic_mean)

    return gathered_data

def plot_metrics(gathered_data, output_dir):
    checkpoints = gathered_data["checkpoints"]
    datasets = gathered_data["datasets"]
    metrics = gathered_data["metrics"]

    n_datasets = len(datasets)
    n_metrics = len(metrics) + 1
    fig, axs = plt.subplots(n_datasets, n_metrics, figsize=(8 * n_metrics, 6 * n_datasets), sharex=True)
    if n_datasets == 1:
        axs = axs.reshape(1, -1)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    label_map = {
        'harmonic_mean': 'Harmonic Mean',
        'rouge_1': 'ROUGE_1',
        'rouge_2': 'ROUGE_2',
        'rouge_l': 'ROUGE_L',
        'rouge_lsum': 'ROUGE_LSUM',
        'probability': 'Probability (Log Scale)',
        'truth_ratio': 'Truth Ratio'
    }

    for i, dataset in enumerate(datasets):
        for j, (metric, color) in enumerate(zip(['harmonic_mean'] + metrics, colors)):
            if metric == 'harmonic_mean':
                y_data = gathered_data["harmonic_means"][dataset]
            else:
                y_data = gathered_data["metric_data"][dataset][metric]
            label = label_map.get(metric, metric.capitalize())
            axs[i, j].plot(checkpoints, y_data, marker='o', color=color)
            axs[i, j].set_title(f'{label} ({dataset})')
            axs[i, j].set_ylabel(label)
            if label == 'Probability (Log Scale)':
                axs[i, j].set_yscale('log')
            axs[i, j].grid(True)
            axs[i, j].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    metrics_plot_path = os.path.join(output_dir, 'across_checkpoints.png')
    plt.savefig(metrics_plot_path)
    plt.close()

    # Overall harmonic mean plot
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoints, gathered_data["overall_harmonic_means"], marker='o', color='red')
    plt.xlabel('Checkpoint')
    plt.ylabel('Overall Harmonic Mean of Metrics')
    plt.title('Overall Harmonic Mean')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    overall_plot_path = os.path.join(output_dir, 'overall.png')
    plt.savefig(overall_plot_path)
    plt.close()

    return metrics_plot_path, overall_plot_path

def plot_metrics_across_checkpoints(data, output_dir, excluded_metrics=None):
    os.makedirs(output_dir, exist_ok=True)
    gathered_data = gather_metrics(data, excluded_metrics)
    return plot_metrics(gathered_data, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to JSON created by llm_unlearning.evaluate_model.")
    parser.add_argument("output_dir", help="Directory to save plots in.")
    parser.add_argument("--exclude_metrics", nargs="*", default=["truth_ratio"], help="List of metrics to exclude from plotting and harmonic mean computation.")
    args = parser.parse_args()

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    metrics_plot, overall_plot = plot_metrics_across_checkpoints(data, args.output_dir, args.exclude_metrics)
    print(f"Metrics plot saved to: {metrics_plot}")
    print(f"Overall harmonic mean plot saved to: {overall_plot}")
