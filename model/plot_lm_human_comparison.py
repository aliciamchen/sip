#!/usr/bin/env python3
"""
Visualize LM vs human ratings for the risk (saliva transfer) task.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils import get_project_root


def main():
    sns.set_context("talk")
    sns.set_style("white")
    plt.rc("axes.spines", top=False, right=False)

    data_path = get_project_root() / "data" / "pilots" / "risk" / "lm_human_comparison.csv"
    df = pd.read_csv(data_path)

    correlation = df["lm_rating"].corr(df["human_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]

    plasma = plt.get_cmap('plasma')
    colors = [plasma(i * 0.85 / 3) for i in range(4)]
    action_labels = ['0', '1', '2', '3']

    human_data_path = get_project_root() / "data" / "pilots" / "risk" / "risk_summary.csv"
    human_df = pd.read_csv(human_data_path)
    df_with_ci = df.merge(human_df[["scenario_label", "action", "ci_lower", "ci_upper"]],
                          on=["scenario_label", "action"], how="left")

    for action in range(4):
        subset = df_with_ci[df_with_ci["action"] == action]
        ax1.errorbar(subset["lm_rating"], subset["human_mean"],
                    yerr=[subset["human_mean"] - subset["ci_lower"],
                          subset["ci_upper"] - subset["human_mean"]],
                    fmt='none', ecolor=colors[action], alpha=0.4, capsize=0)
        ax1.scatter(subset["lm_rating"], subset["human_mean"],
                   c=[colors[action]], label=action_labels[action], s=80, alpha=0.8)

    ax1.plot([0, 6], [0, 6], 'k--', alpha=0.3)

    z = np.polyfit(df["lm_rating"], df["human_mean"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 6, 100)
    ax1.plot(x_line, p(x_line), color='gray', alpha=0.7, linewidth=2)

    ax1.set_xlabel("LM rating")
    ax1.set_ylabel("Human mean rating")
    ax1.set_title(f"LM vs human risk ratings (r = {correlation:.3f})")
    ax1.set_xlim(-0.2, 6.2)
    ax1.set_ylim(-0.2, 6.2)
    ax1.set_xticks(range(7))
    ax1.set_yticks(range(7))
    ax1.legend(title='Action', loc='upper left')
    ax1.set_aspect('equal')

    # --- Plot 2: Bar chart by action ---
    ax2 = axes[1]

    action_means = df.groupby("action").agg({
        "lm_rating": "mean",
        "human_mean": "mean"
    }).reset_index()

    x = np.arange(4)
    width = 0.35

    bars1 = ax2.bar(x - width/2, action_means["human_mean"], width,
                    label='Human', color='#555555', alpha=0.7)
    bars2 = ax2.bar(x + width/2, action_means["lm_rating"], width,
                    label='LM (Llama-3.1-8B)', color=colors[2], alpha=0.8)

    ax2.set_xlabel("Action")
    ax2.set_ylabel("Mean rating")
    ax2.set_title("Mean Ratings by Action")
    ax2.set_xticks(x)
    ax2.set_xticklabels(['0', '1', '2', '3'])
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 6.5)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12)

    plt.tight_layout()

    output_path = get_project_root() / "data" / "pilots" / "risk" / "lm_human_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
