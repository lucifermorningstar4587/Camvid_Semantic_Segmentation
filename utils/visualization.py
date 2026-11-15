import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: str = "./outputs/plots/confusion_matrix.png",
    normalize: bool = True
):
    """Generates and saves a high-resolution confusion matrix heatmap."""
    cm = np.array(confusion_matrix, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=(row_sums > 0))

    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title("Normalized Confusion Matrix (CamVid 11 Classes)", fontsize=14, pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Ground Truth Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {Path(save_path).resolve()}")


def plot_class_distribution(
    stats: Dict[str, any],
    save_path: str = "./outputs/plots/class_pixel_imbalance.png"
):
    """Plots the dataset class pixel frequency showing severe imbalance."""
    classes = stats["classes"]
    frequencies = np.array(stats["frequencies"]) * 100.0 # to percentage

    plt.figure(figsize=(12, 5), dpi=300)
    bars = plt.bar(classes, frequencies, color="royalblue", edgecolor="black", alpha=0.85)
    plt.yscale("log")
    plt.title("CamVid Class Pixel Distribution (Logarithmic Scale - Showing Imbalance)", fontsize=13)
    plt.ylabel("Pixel Percentage (%) [Log Scale]", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.2f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Class distribution plot saved to {Path(save_path).resolve()}")
