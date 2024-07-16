import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from matplotlib.scale import ScaleBase, register_scale
from matplotlib.transforms import Transform
from matplotlib.ticker import AutoLocator, ScalarFormatter, FuncFormatter

class MixedLogLinearScale(ScaleBase):
    name = 'mixedloglinear'

    def __init__(self, axis, threshold=0.1, **kwargs):
        ScaleBase.__init__(self, axis)
        self.threshold = threshold

    def get_transform(self):
        return MixedLogLinearTransform(self.threshold)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(MixedLogLinearLocator(self.threshold))
        axis.set_major_formatter(MixedLogLinearFormatter(self.threshold))

class MixedLogLinearTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, threshold):
        Transform.__init__(self)
        self.threshold = threshold

    def transform_non_affine(self, a):
        mask = a < self.threshold
        out = np.empty(a.shape, dtype=np.float64)
        out[mask] = np.log10(a[mask] / self.threshold) + 1
        out[~mask] = a[~mask] / self.threshold
        return out

    def inverted(self):
        return InvertedMixedLogLinearTransform(self.threshold)

class InvertedMixedLogLinearTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, threshold):
        Transform.__init__(self)
        self.threshold = threshold

    def transform_non_affine(self, a):
        mask = a < 1
        out = np.empty(a.shape, dtype=np.float64)
        out[mask] = 10 ** (a[mask] - 1) * self.threshold
        out[~mask] = a[~mask] * self.threshold
        return out

    def inverted(self):
        return MixedLogLinearTransform(self.threshold)

class MixedLogLinearLocator(AutoLocator):
    def __init__(self, threshold):
        self.threshold = threshold
        super().__init__()

    def tick_values(self, vmin, vmax):
        trans = MixedLogLinearTransform(self.threshold)
        linear_vmin, linear_vmax = trans.transform_non_affine(vmin), trans.transform_non_affine(vmax)

        log_ticks = np.logspace(np.log10(vmin), np.log10(self.threshold), num=5)
        log_ticks = log_ticks[log_ticks < self.threshold]

        ticks = list(log_ticks) + [self.threshold]

        # Add linear ticks above the threshold
        if linear_vmax > 1:
            linear_ticks = np.linspace(1, linear_vmax, num=10)
            linear_ticks = linear_ticks[linear_ticks > 1]  # Remove any tick at or below 1
            ticks.extend(trans.inverted().transform_non_affine(linear_ticks))

        return np.array(ticks)

class MixedLogLinearFormatter(ScalarFormatter):
    def __init__(self, threshold):
        self.threshold = threshold
        super().__init__()

    def __call__(self, x, pos=None):
        if x < self.threshold:
            return f"{x:.0e}"
        else:
            return f"{x:.2f}"

register_scale(MixedLogLinearScale)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_metrics(data, metric_name):
    checkpoints = sorted([k for k in data.keys() if k.startswith("checkpoint-")],
                         key=lambda x: int(x.split('-')[1]))
    values = [data[checkpoint]['aggregate_metrics'][metric_name] for checkpoint in checkpoints]
    return checkpoints, values

def plot_metrics(data_pairs, output_file, log_scale=True, mixed_scale=False):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(data_pairs)))

    for (name, retain_data, forget_data), color in zip(data_pairs, colors):
        retain_checkpoints, retain_values = extract_metrics(retain_data, 'model_utility')
        forget_checkpoints, forget_values = extract_metrics(forget_data, 'ks_test')

        checkpoint_indices = list(range(len(retain_checkpoints)))
        marker_sizes = [40 + 10 * i for i in checkpoint_indices]

        scatter = ax1.scatter(retain_values, forget_values, c=[color], marker='o',
                              s=marker_sizes, label=name, alpha=0.7)

        # Connecting lines
        ax1.plot(retain_values, forget_values, c=color, alpha=0.5)

    ax1.set_xlabel('Model Utility')
    ax1.set_ylabel('Forget Quality (ks_test)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    if mixed_scale:
        ax1.set_yscale('mixedloglinear', threshold=0.1)
        ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.7)
        ax1.text(ax1.get_xlim()[1], 0.1, 'Linear', ha='right', va='bottom', color='red', alpha=0.7)
        ax1.text(ax1.get_xlim()[1], 0.09, 'Log', ha='right', va='top', color='red', alpha=0.7)
    elif log_scale:
        ax1.set_yscale('log')
    else:
        ax1.set_yscale('linear')

    if mixed_scale:
        ax1.axhline(y=-1, color='gray', linestyle='--')

    plt.title('Model Utility vs Forget Quality')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def process_folder(folder_path):
    data_pairs = []

    retain_files = glob(os.path.join(folder_path, '*_retain_results.json'))
    for retain_file in retain_files:
        name = os.path.basename(retain_file).replace('_retain_results.json', '')
        forget_file = retain_file.replace('_retain_results.json', '_forget_results.json')

        if os.path.exists(forget_file):
            retain_data = load_json(retain_file)
            forget_data = load_json(forget_file)
            data_pairs.append((name, retain_data, forget_data))
        else:
            print(f"Warning: No matching forget file found for {retain_file}")

    retain_file = os.path.join(folder_path, 'retain_results.json')
    forget_file = os.path.join(folder_path, 'forget_results.json')
    if os.path.exists(retain_file) and os.path.exists(forget_file):
        retain_data = load_json(retain_file)
        forget_data = load_json(forget_file)
        name = os.path.basename(folder_path)  # Use folder name as the dataset name
        data_pairs.append((name, retain_data, forget_data))

    return data_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help='Path to input folder containing retain and forget JSON files')
    parser.add_argument('output_file', type=str, help='Path to save the output plot')
    parser.add_argument('--linear', action='store_true', help='Use linear scale for forget quality')
    parser.add_argument('--mixed', action='store_true', help='Use mixed log-linear scale for FQ (log until 0.1, linear above)')
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        raise ValueError("Input must be a folder containing retain and forget JSON files")

    data_pairs = process_folder(args.input_folder)

    if not data_pairs:
        raise ValueError("No valid pairs of retain and forget files found in the input folder")

    plot_metrics(data_pairs, args.output_file, log_scale=not args.linear, mixed_scale=args.mixed)
    print(f"Plot saved to {args.output_file}")

if __name__ == "__main__":
    main()
