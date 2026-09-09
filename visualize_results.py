"""
visualize_results.py

Generates all graphical results for the seq2seq bAbI QA project:

    1. Training/validation loss curves      -> results/training_loss.png
    2. Training/validation accuracy curves  -> results/training_accuracy.png
    3. Confusion matrix heatmap             -> results/confusion_matrix.png
    4. Per-class (per-answer) accuracy bar  -> results/per_class_accuracy.png
    5. Ground-truth vs prediction distribution -> results/answer_distribution.png

Requirements:
    pip install matplotlib seaborn scikit-learn pandas --break-system-packages

Run from the project root AFTER running predict.py (with the
results/predictions.csv addition) and after training has produced
a training log CSV (e.g. models/attention_training_log.csv).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ============================================================
# CONFIGURATION - edit these paths if yours differ
# ============================================================

TRAINING_LOG_CSV = "models/attention_bidir_training_log.csv"
PREDICTIONS_CSV = "results/predictions.csv"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 1 & 2. TRAINING CURVES
# ============================================================

def plot_training_curves():
    if not os.path.exists(TRAINING_LOG_CSV):
        print(f"Skipping training curves - {TRAINING_LOG_CSV} not found.")
        return

    log_df = pd.read_csv(TRAINING_LOG_CSV)
    print("Training log columns found:", list(log_df.columns))

    epochs = log_df["epoch"] if "epoch" in log_df.columns else range(len(log_df))

    # --- Loss ---
    plt.figure(figsize=(8, 5))
    if "loss" in log_df.columns:
        plt.plot(epochs, log_df["loss"], label="Train Loss", marker="o")
    if "val_loss" in log_df.columns:
        plt.plot(epochs, log_df["val_loss"], label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/training_loss.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

    # --- Accuracy ---
    acc_col = "accuracy" if "accuracy" in log_df.columns else "acc"
    val_acc_col = "val_accuracy" if "val_accuracy" in log_df.columns else "val_acc"

    plt.figure(figsize=(8, 5))
    if acc_col in log_df.columns:
        plt.plot(epochs, log_df[acc_col], label="Train Accuracy", marker="o")
    if val_acc_col in log_df.columns:
        plt.plot(epochs, log_df[val_acc_col], label="Validation Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/training_accuracy.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ============================================================
# 3. CONFUSION MATRIX HEATMAP
# ============================================================

def plot_confusion_matrix(preds_df):
    labels = sorted(preds_df["ground_truth"].unique())

    cm = confusion_matrix(
        preds_df["ground_truth"],
        preds_df["prediction"],
        labels=labels,
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix - Test Set Predictions")
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ============================================================
# 4. PER-CLASS ACCURACY BAR CHART
# ============================================================

def plot_per_class_accuracy(preds_df):
    grouped = preds_df.groupby("ground_truth")["correct"].mean().sort_index()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(grouped.index, grouped.values, color="steelblue")
    plt.axhline(
        preds_df["correct"].mean(),
        color="red",
        linestyle="--",
        label=f"Overall accuracy ({preds_df['correct'].mean():.2%})",
    )
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Per-Answer-Class Accuracy")
    plt.legend()

    for bar, val in zip(bars, grouped.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/per_class_accuracy.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ============================================================
# 5. GROUND-TRUTH VS PREDICTION DISTRIBUTION
# ============================================================

def plot_answer_distribution(preds_df):
    truth_counts = preds_df["ground_truth"].value_counts().sort_index()
    pred_counts = preds_df["prediction"].value_counts().reindex(
        truth_counts.index, fill_value=0
    )

    x = range(len(truth_counts))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(
        [i - width / 2 for i in x],
        truth_counts.values,
        width,
        label="Ground Truth",
        color="seagreen",
    )
    plt.bar(
        [i + width / 2 for i in x],
        pred_counts.values,
        width,
        label="Model Predictions",
        color="indianred",
    )
    plt.xticks(list(x), truth_counts.index)
    plt.ylabel("Count")
    plt.title("Ground Truth vs Predicted Answer Distribution")
    plt.legend()
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/answer_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    plot_training_curves()

    if not os.path.exists(PREDICTIONS_CSV):
        print(
            f"\n{PREDICTIONS_CSV} not found. Add the results-saving "
            f"snippet to predict.py and re-run it first."
        )
        return

    preds_df = pd.read_csv(PREDICTIONS_CSV)

    plot_confusion_matrix(preds_df)
    plot_per_class_accuracy(preds_df)
    plot_answer_distribution(preds_df)

    print(f"\nAll plots saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()