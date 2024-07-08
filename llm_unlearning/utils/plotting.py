import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

class Results:
    def __init__(self, json_data):
        self.data = json_data

    def __getitem__(self, key):
        return self.data[key]

    def checkpoints(self, exclude = None):
        if exclude is None: exclude = ["retain"]
        checkpoints = [k for k in self.data.keys() if not k in exclude]
        checkpoints.sort(key=lambda x: int(x.split('-')[1]) if x != "checkpoint-0" else -1)
        return checkpoints

    def datasets(self, checkpoint, exclude = None):
        if exclude is None: exclude = []
        return [k for k in self[checkpoint]["metrics"].keys() if not k in exclude]

    def metrics(self, checkpoint, dataset, exclude = None):
        if exclude is None: exclude = []
        return [k for k in self[checkpoint]["metrics"][dataset].keys() if not k in exclude and not k.endswith('_metadata')]

    def metric_values(self, checkpoint, dataset, metric):
        return self[checkpoint]["metrics"][dataset][metric]

    def checkpoint_metric_data(self, dataset, metric):
        checkpoints = self.checkpoints()
        return [self.metric_values(checkpoint, dataset, metric) for checkpoint in checkpoints]

    def harmonic_mean(self, checkpoint, dataset):
        metrics = self.metrics(checkpoint, dataset)
        flat_values = [self.metric_values(checkpoint, dataset, metric) for metric in metrics]
        return harmonic_mean(flat_values)


def harmonic_mean(arr):
    return len(arr) / np.sum(1.0 / np.array(arr))

def gather_metrics(data, excluded_metrics=None):
    if excluded_metrics is None:
        excluded_metrics = []

    results = Results(data)

    checkpoints = results.checkpoints()
    datasets = results.datasets(checkpoints[0])
    included_metrics = results.metrics(checkpoints[0], datasets[0], excluded_metrics)

    gathered_data = {
        "checkpoints": checkpoints,
        "datasets": datasets,
        "metrics": included_metrics,
        "metric_data": {dataset: {metric: results.checkpoint_metric_data(dataset, metric) for metric in included_metrics} for dataset in datasets},
        "harmonic_means": {dataset: [] for dataset in datasets},
        "overall_harmonic_means": [],
        "aggregate_metrics": []
    }

    # Calculate harmonic means
    for checkpoint in checkpoints:
        for dataset in datasets:
            gathered_data["harmonic_means"][dataset].append(results.harmonic_mean(checkpoint, dataset))
        gathered_data["overall_harmonic_means"].append(harmonic_mean([hm for dataset in datasets for hm in gathered_data["harmonic_means"][dataset]]))

    return gathered_data

def plot_metrics(gathered_data, output_dir, log_scale = None):
    if log_scale is None: log_scale = ['probability']
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
            if metric == 'harmonic_mean': y_data = gathered_data["harmonic_means"][dataset]
            else: y_data = gathered_data["metric_data"][dataset][metric]
            if metric in log_scale: axs[i, j].set_yscale('log')
            label = label_map.get(metric, metric.capitalize())
            axs[i, j].plot(checkpoints, y_data, marker='o', color=color)
            axs[i, j].set_title(f'{label} ({dataset})')
            axs[i, j].set_ylabel(label)
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

def plot_aggregate_metrics(data, output_dir, log_scale = None):
    if log_scale is None: log_scale = ['ks_test', 'model_utility']
    checkpoints = [cp for cp in data.keys() if cp != "retain"]
    checkpoints.sort(key=lambda x: int(x.split('-')[1]) if x != "checkpoint-0" else -1)

    aggregate_metrics = list(data[checkpoints[0]]["aggregate_metrics"].keys())

    n_metrics = len(aggregate_metrics)
    fig, axs = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False)

    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for i, metric in enumerate(aggregate_metrics):
        y_data = [data[cp]["aggregate_metrics"][metric] for cp in checkpoints]
        if metric in log_scale: axs[0, i].set_yscale('log')
        axs[0, i].plot(checkpoints, y_data, marker='o', color=colors[i % len(colors)])
        axs[0, i].set_title(metric.replace('_', ' ').title())
        axs[0, i].set_ylabel('Value')
        axs[0, i].grid(True)
        axs[0, i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    aggregate_plot_path = os.path.join(output_dir, 'aggregate_metrics.png')
    plt.savefig(aggregate_plot_path)
    plt.close()

    return aggregate_plot_path


def plot_metrics_across_checkpoints(data, output_dir, excluded_metrics=None):
    os.makedirs(output_dir, exist_ok=True)
    gathered_data = gather_metrics(data, excluded_metrics)
    metrics_plot, overall_plot = plot_metrics(gathered_data, output_dir)
    aggregate_plot = plot_aggregate_metrics(data, output_dir)
    return metrics_plot, overall_plot, aggregate_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to JSON created by llm_unlearning.evaluate_model.")
    parser.add_argument("output_dir", help="Directory to save plots in.")
    parser.add_argument("--exclude_metrics", nargs="*", help="List of metrics to exclude from plotting and harmonic mean computation.")
    args = parser.parse_args()

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    metrics_plot, overall_plot, aggregate_plot = plot_metrics_across_checkpoints(data, args.output_dir, args.exclude_metrics)
    print(f"Metrics plot saved to: {metrics_plot}")
    print(f"Overall harmonic mean plot saved to: {overall_plot}")
    print(f"Aggregate metrics plot saved to: {aggregate_plot}")
