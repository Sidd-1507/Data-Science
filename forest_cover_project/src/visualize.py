"""
visualize.py
------------
All plotting functions used in the notebook and training pipeline.
Each function saves a PNG to `save_dir` and returns the figure.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, classification_report

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

COVER_NAMES = {
    1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
    4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz",
}


# ── EDA plots ─────────────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame, save_dir: str = None):
    vc     = df["Cover_Type"].value_counts().sort_index()
    labels = [COVER_NAMES[i] for i in vc.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(labels, vc.values, color=sns.color_palette("Set2", 7),
                edgecolor="white", linewidth=0.8)
    axes[0].set_title("Cover Type — Sample Counts", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Cover Type")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=35)
    for i, v in enumerate(vc.values):
        axes[0].text(i, v + 100, str(v), ha="center", fontsize=9)

    axes[1].pie(vc.values, labels=labels, autopct="%1.1f%%",
                colors=sns.color_palette("Set2", 7), startangle=140,
                wedgeprops=dict(edgecolor="white"))
    axes[1].set_title("Cover Type — Proportion", fontweight="bold", fontsize=13)

    plt.suptitle("🌲 Target Variable Distribution", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, save_dir, "target_distribution.png")
    return fig


def plot_numeric_distributions(df: pd.DataFrame, num_cols: list, save_dir: str = None):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()
    for i, col in enumerate(num_cols[:12]):
        for ctype, grp in df.groupby("Cover_Type"):
            axes[i].hist(grp[col], bins=30, alpha=0.45,
                         label=COVER_NAMES[ctype], density=True)
        axes[i].set_title(col, fontsize=10, fontweight="bold")
        axes[i].tick_params(labelsize=8)
    axes[0].legend(fontsize=7, ncol=2, title="Cover Type")
    plt.suptitle("Numerical Feature Distributions by Cover Type",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, save_dir, "numeric_distributions.png")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: list, save_dir: str = None):
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df[num_cols + ["Cover_Type"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation Matrix — Numerical Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_dir, "correlation_heatmap.png")
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 30, save_dir: str = None):
    top = importance_df.head(top_n)
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    methods = ["F_Score", "MI_Score", "RF_Importance"]
    titles  = ["ANOVA F-Score", "Mutual Information", "RF Importance"]
    colors  = ["#0984e3", "#00b894", "#d63031"]
    for ax, method, title, color in zip(axes, methods, titles, colors):
        sv = top[method].sort_values(ascending=True)
        ax.barh(sv.index, sv.values, color=color, alpha=0.8)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Normalised Score")
        ax.tick_params(axis="y", labelsize=7)
    plt.suptitle(f"Feature Importance — Three Methods (Top {top_n})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_dir, "feature_importance.png")
    return fig


# ── Model comparison plots ────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame, cv_scores: dict, save_dir: str = None):
    model_names = results_df.index.tolist()
    test_accs   = results_df["Test_Accuracy"].values
    cv_means    = results_df["CV_Accuracy_Mean"].values
    cv_stds     = results_df["CV_Accuracy_Std"].values
    f1_macros   = results_df["F1_Macro"].values
    palette     = sns.color_palette("RdYlGn", len(model_names))[::-1]
    y_pos       = np.arange(len(model_names))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(y_pos, test_accs, color=palette, edgecolor="white",
                 alpha=0.85, height=0.5, label="Test Acc")
    axes[0].errorbar(cv_means, y_pos, xerr=cv_stds * 2, fmt="D",
                     color="navy", markersize=5, capsize=4, label="CV Mean ± 2σ")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(model_names, fontsize=10)
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Test Accuracy vs CV Accuracy", fontweight="bold", fontsize=12)
    axes[0].axvline(x=0.9, color="grey", linestyle="--", alpha=0.5, label="0.90 line")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim([0.5, 1.05])
    for i, t in enumerate(test_accs):
        axes[0].text(t + 0.002, i, f"{t:.4f}", va="center", fontsize=8)

    axes[1].barh(y_pos, f1_macros, color=palette, edgecolor="white", alpha=0.85, height=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(model_names, fontsize=10)
    axes[1].set_xlabel("F1-Score (Macro)")
    axes[1].set_title("Macro F1-Score by Model", fontweight="bold", fontsize=12)
    axes[1].set_xlim([0.4, 1.05])
    for i, f in enumerate(f1_macros):
        axes[1].text(f + 0.002, i, f"{f:.4f}", va="center", fontsize=8)

    plt.suptitle("🏆 Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_dir, "model_comparison.png")
    return fig


def plot_cv_distributions(cv_scores: dict, results_df: pd.DataFrame, save_dir: str = None):
    ordered = results_df.index.tolist()
    data    = [cv_scores[n] for n in ordered]
    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(data, patch_artist=True, notch=True)
    colors = sns.color_palette("RdYlGn", len(ordered))[::-1]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xticks(range(1, len(ordered) + 1))
    ax.set_xticklabels(ordered, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("CV Accuracy (5-Fold)")
    ax.set_title("Cross-Validation Accuracy Distribution per Model",
                 fontweight="bold", fontsize=13)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="0.90 line")
    ax.legend()
    plt.tight_layout()
    _save(fig, save_dir, "cv_distributions.png")
    return fig


def plot_confusion_matrices(results: dict, y_test, top_n: int = 4, save_dir: str = None):
    top_names  = list(results.keys())[:top_n]
    class_labels = [COVER_NAMES[c] for c in sorted(COVER_NAMES)]
    fig, axes  = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for i, name in enumerate(top_names):
        y_pred = results[name]["y_pred"]
        cm     = confusion_matrix(y_test, y_pred)
        disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=axes[i], cmap="Blues", colorbar=False)
        acc = results[name]["Test_Accuracy"]
        axes[i].set_title(f"{name}\nAcc: {acc:.4f}", fontweight="bold", fontsize=11)
        axes[i].tick_params(axis="x", rotation=40, labelsize=8)
        axes[i].tick_params(axis="y", labelsize=8)
    plt.suptitle("Confusion Matrices — Top 4 Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_dir, "confusion_matrices.png")
    return fig


def plot_radar_chart(results_df: pd.DataFrame, top_n: int = 5, save_dir: str = None):
    top     = results_df.head(top_n)
    metrics = ["Test_Accuracy", "F1_Macro", "F1_Weighted", "CV_Accuracy_Mean"]
    labels  = ["Test Accuracy", "F1 Macro", "F1 Weighted", "CV Accuracy"]
    angles  = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    palette = sns.color_palette("Set1", top_n)
    for (name, row), color in zip(top.iterrows(), palette):
        values = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=name)
        ax.fill(angles, values, alpha=0.1, color=color)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)
    ax.set_ylim(0.85, 1.0)
    ax.set_title("Radar Chart — Top Models", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    _save(fig, save_dir, "radar_chart.png")
    return fig


# ── Helper ────────────────────────────────────────────────────────────────────

def _save(fig, save_dir, filename):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
        print(f"  Saved: {filename}")
    plt.show()
