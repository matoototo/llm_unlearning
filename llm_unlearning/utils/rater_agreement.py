import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import argparse

def load_data(filename):
    return pd.read_csv(filename)

def compute_observed_agreement(data1, data2):
    return np.mean(data1 == data2)

def compute_agreement_matrix(data, raters):
    agreement_kappa = pd.DataFrame(index=raters, columns=raters)
    agreement_po = pd.DataFrame(index=raters, columns=raters)
    for rater1 in raters:
        for rater2 in raters:
            if rater1 != rater2:
                kappa = cohen_kappa_score(data[rater1], data[rater2])
                po = compute_observed_agreement(data[rater1], data[rater2])
                agreement_kappa.loc[rater1, rater2] = kappa
                agreement_po.loc[rater1, rater2] = po
            else:
                agreement_kappa.loc[rater1, rater2] = 1.0
                agreement_po.loc[rater1, rater2] = 1.0
    return agreement_kappa, agreement_po

def compute_aggregate_human(human_ratings):
    return mode(np.array(human_ratings).T, axis=1).mode.flatten()

def visualize_agreement_matrix(agreement_kappa, agreement_po, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    norm_kappa = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im1 = ax1.imshow(agreement_kappa.astype(float), cmap='coolwarm', norm=norm_kappa)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xticks(np.arange(len(agreement_kappa.columns)))
    ax1.set_yticks(np.arange(len(agreement_kappa.index)))
    ax1.set_xticklabels(agreement_kappa.columns, rotation=90)
    ax1.set_yticklabels(agreement_kappa.index)
    ax1.set_title("Cohen's Kappa")

    norm_po = colors.Normalize(vmin=0, vmax=1)
    im2 = ax2.imshow(agreement_po.astype(float), cmap='viridis', norm=norm_po)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_xticks(np.arange(len(agreement_po.columns)))
    ax2.set_yticks(np.arange(len(agreement_po.index)))
    ax2.set_xticklabels(agreement_po.columns, rotation=90)
    ax2.set_yticklabels(agreement_po.index)
    ax2.set_title("Observed Agreement (p_o)")

    for ax, data in [(ax1, agreement_kappa), (ax2, agreement_po)]:
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                ax.text(j, i, f"{data.iloc[i, j]:.2f}",
                        ha="center", va="center", color="black")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def compute_model_agreement(data, models):
    kappa_agreement = {model: cohen_kappa_score(data[model], data['Aggregate_Human']) for model in models}
    po_agreement = {model: compute_observed_agreement(data[model], data['Aggregate_Human']) for model in models}
    return kappa_agreement, po_agreement

def visualize_model_agreement(model_agreement_kappa, model_agreement_po):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_kappa = pd.DataFrame(model_agreement_kappa)
    df_po = pd.DataFrame(model_agreement_po)

    df_kappa.plot(kind='bar', ax=ax1)
    ax1.set_title("Model Agreement with Aggregate Human Ratings (Cohen's Kappa)")
    ax1.set_ylabel("Cohen's Kappa")
    ax1.set_ylim(-1, 1)

    df_po.plot(kind='bar', ax=ax2)
    ax2.set_title("Model Agreement with Aggregate Human Ratings (Observed Agreement)")
    ax2.set_ylabel("Observed Agreement (p_o)")
    ax2.set_ylim(0, 1)

    for ax in [ax1, ax2]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

def main(data_dir):
    # Get all CSV files in the specified directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    csv_files.sort()

    # Separate models and human raters
    models = [f[:-4] for f in csv_files if f.startswith('model_')]
    human_raters = [f[:-4] for f in csv_files if not f.startswith('model_')]
    all_raters = human_raters + models

    # Load data from CSV files
    data = {rater: load_data(os.path.join(data_dir, f"{rater}.csv")) for rater in all_raters}
    data_col1 = pd.DataFrame({rater: data[rater]['unlearning'] for rater in all_raters})
    data_col2 = pd.DataFrame({rater: data[rater]['coherency'] for rater in all_raters})

    # Compute agreement matrices
    agreement_kappa_col1, agreement_po_col1 = compute_agreement_matrix(data_col1, all_raters)
    agreement_kappa_col2, agreement_po_col2 = compute_agreement_matrix(data_col2, all_raters)

    # Compute aggregate human ratings
    data_col1['Aggregate_Human'] = compute_aggregate_human([data_col1[rater] for rater in human_raters])
    data_col2['Aggregate_Human'] = compute_aggregate_human([data_col2[rater] for rater in human_raters])

    # Recompute agreement matrices with aggregate human
    raters_with_aggregate = all_raters + ['Aggregate_Human']
    agreement_kappa_col1, agreement_po_col1 = compute_agreement_matrix(data_col1, raters_with_aggregate)
    agreement_kappa_col2, agreement_po_col2 = compute_agreement_matrix(data_col2, raters_with_aggregate)

    # Visualize agreement matrices
    visualize_agreement_matrix(agreement_kappa_col1, agreement_po_col1, "Agreement Matrix - Unlearning")
    visualize_agreement_matrix(agreement_kappa_col2, agreement_po_col2, "Agreement Matrix - Coherency")

    # Compute and visualize model agreement
    model_kappa_col1, model_po_col1 = compute_model_agreement(data_col1, models)
    model_kappa_col2, model_po_col2 = compute_model_agreement(data_col2, models)
    visualize_model_agreement(
        {'Unlearning': model_kappa_col1, 'Coherency': model_kappa_col2},
        {'Unlearning': model_po_col1, 'Coherency': model_po_col2}
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory containing CSV files with rating data")
    args = parser.parse_args()

    main(args.data_dir)
