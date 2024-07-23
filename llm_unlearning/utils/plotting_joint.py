import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from glob import glob
from matplotlib.transforms import Transform
from matplotlib.scale import ScaleBase, register_scale
from matplotlib.ticker import AutoLocator, ScalarFormatter

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

        if linear_vmax > 1:
            linear_ticks = np.linspace(1, linear_vmax, num=10)
            linear_ticks = linear_ticks[linear_ticks > 1]
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

def process_run_folder(folder_path):
    retain_file = os.path.join(folder_path, 'retain_results.json')
    forget_file = os.path.join(folder_path, 'forget_results.json')

    retain_data = load_json(retain_file) if os.path.exists(retain_file) else None
    forget_data = load_json(forget_file) if os.path.exists(forget_file) else None

    if retain_data is None and forget_data is None:
        print(f"Warning: No retain or forget file found in {folder_path}")
    return retain_data, forget_data

def aggregate_results(results, confidence):
    retain_values = [r[0] for r in results if r[0] is not None]
    forget_values = [r[1] for r in results if r[1] is not None]

    if not retain_values and not forget_values:
        return None, None, None, None, 0, 0

    model_utility = [extract_metrics(r, 'model_utility')[1] for r in retain_values] if retain_values else None
    forget_quality = [extract_metrics(r, 'ks_test')[1] for r in forget_values] if forget_values else None

    n_retain = len(model_utility) if model_utility else 0
    n_forget = len(forget_quality) if forget_quality else 0

    if model_utility:
        model_utility_mean = np.mean(model_utility, axis=0)
        model_utility_se = stats.sem(model_utility, axis=0)
        t_value_retain = stats.t.ppf((1 + confidence) / 2, n_retain - 1)
        model_utility_ci = t_value_retain * model_utility_se
    else:
        model_utility_mean = model_utility_ci = None

    if forget_quality:
        forget_quality_mean = np.mean(forget_quality, axis=0)
        forget_quality_se = stats.sem(forget_quality, axis=0)
        t_value_forget = stats.t.ppf((1 + confidence) / 2, n_forget - 1)
        forget_quality_ci = t_value_forget * forget_quality_se
    else:
        forget_quality_mean = forget_quality_ci = None

    return model_utility_mean, forget_quality_mean, model_utility_ci, forget_quality_ci, n_retain, n_forget

def process_folder(folder_path, confidence):
    data_pairs = []

    # Process folders (aggregated data)
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            run_folders = [f for f in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, f))]
            if run_folders:
                results = [process_run_folder(os.path.join(item_path, run)) for run in run_folders]
                retain_agg, forget_agg, retain_ci, forget_ci, n_retain, n_forget = aggregate_results(results, confidence)
                if retain_agg is not None or forget_agg is not None:
                    data_pairs.append((item, retain_agg, forget_agg, retain_ci, forget_ci, n_retain, n_forget))
            else:
                retain_data, forget_data = process_run_folder(item_path)
                if retain_data is not None or forget_data is not None:
                    data_pairs.append((item, retain_data, forget_data, None, None, 1, 1))

    # Process single file pairs in the upper folder
    retain_files = glob(os.path.join(folder_path, '*_retain_results.json'))
    forget_files = glob(os.path.join(folder_path, '*_forget_results.json'))
    all_files = set([f.replace('_retain_results.json', '').replace('_forget_results.json', '') for f in retain_files + forget_files])

    for base_name in all_files:
        retain_file = f"{base_name}_retain_results.json"
        forget_file = f"{base_name}_forget_results.json"
        
        retain_data = load_json(retain_file) if os.path.exists(retain_file) else None
        forget_data = load_json(forget_file) if os.path.exists(forget_file) else None

        if retain_data is not None or forget_data is not None:
            name = os.path.basename(base_name)
            data_pairs.append((name, retain_data, forget_data, None, None, 1 if retain_data else 0, 1 if forget_data else 0))

    return data_pairs

def plot_metrics(data_pairs, output_file, log_scale=True, mixed_scale=False, ci_mode='y'):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(data_pairs)))

    for (name, retain_data, forget_data, retain_ci, forget_ci, n_retain, n_forget), color in zip(data_pairs, colors):
        if isinstance(retain_data, np.ndarray) or isinstance(forget_data, np.ndarray):  # Aggregated data
            retain_values = retain_data if retain_data is not None else []
            forget_values = forget_data if forget_data is not None else []
            checkpoint_indices = list(range(max(len(retain_values), len(forget_values))))
        else:
            retain_values = extract_metrics(retain_data, 'model_utility')[1] if retain_data else []
            forget_values = extract_metrics(forget_data, 'ks_test')[1] if forget_data else []
            checkpoint_indices = list(range(max(len(retain_values), len(forget_values))))

        marker_sizes = [40 + 10 * i for i in checkpoint_indices]

        legend_name = f"{name} (r:{n_retain}, f:{n_forget})"

        if retain_ci is not None and forget_ci is not None:
            if ci_mode == 'y' and len(forget_values) > 0:
                ax1.fill_between(retain_values,
                                 forget_values - forget_ci,
                                 forget_values + forget_ci,
                                 color=color, alpha=0.2)
            elif ci_mode == 'x' and len(retain_values) > 0:
                ax1.fill_betweenx(forget_values,
                                  retain_values - retain_ci,
                                  retain_values + retain_ci,
                                  color=color, alpha=0.2)

        if len(retain_values) > 0 and len(forget_values) > 0:
            scatter = ax1.scatter(retain_values, forget_values, c=[color], marker='o',
                                  s=marker_sizes, label=legend_name, alpha=0.7)
            # Connecting lines
            ax1.plot(retain_values, forget_values, c=color, alpha=0.5)
        elif len(retain_values) > 0:
            ax1.axvline(x=np.mean(retain_values), color=color, linestyle='--', label=f"{legend_name} (Retain Only)")
        elif len(forget_values) > 0:
            ax1.axhline(y=np.mean(forget_values), color=color, linestyle='--', label=f"{legend_name} (Forget Only)")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help='Path to input folder containing retain and forget JSON files')
    parser.add_argument('output_file', type=str, help='Path to save the output plot')
    parser.add_argument('--linear', action='store_true', help='Use linear scale for forget quality')
    parser.add_argument('--mixed', action='store_true', help='Use mixed log-linear scale for FQ (log until 0.1, linear above)')
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence level for the interval (default: 0.95)')
    parser.add_argument('--ci-mode', choices=['x', 'y'], default='y', help='Axis for confidence interval (x or y, default: y)')
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        raise ValueError("Input must be a folder containing retain and forget JSON files")

    data_pairs = process_folder(args.input_folder, args.confidence)

    if not data_pairs:
        raise ValueError("No valid pairs of retain and forget files found in the input folder")

    plot_metrics(data_pairs, args.output_file, log_scale=not args.linear, mixed_scale=args.mixed, ci_mode=args.ci_mode)
    print(f"Plot saved to {args.output_file}")

if __name__ == "__main__":
    main()
