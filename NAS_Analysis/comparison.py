import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

steel = "steelblue"
dm_color = "seagreen"
error_color = "red"


def topK_recall(df, k, pred_rank_col, true_rank_col="baseline_rank"):
    top_true = set(df[df[true_rank_col] <= k]["arch_id"])
    top_pred = set(df[df[pred_rank_col] <= k]["arch_id"])
    return len(top_true & top_pred) / k


def run_comparison(mtt_csv, dm_csv, baseline_csv, K_max=500):
    # 1. Load
    df_mtt = pd.read_csv(mtt_csv).rename(columns={"test_accuracy": "mtt_acc"})
    df_dm = pd.read_csv(dm_csv).rename(columns={"test_accuracy": "dm_acc"})
    df_base = pd.read_csv(baseline_csv).rename(columns={"test_accuracy": "baseline_acc"})

    df = pd.merge(df_mtt[["arch_id", "mtt_acc"]], df_base[["arch_id", "baseline_acc"]], on="arch_id")
    df = pd.merge(df, df_dm[["arch_id", "dm_acc"]], on="arch_id")

    df = df[df["arch_id"] <= 1000].copy()
    df["baseline_rank"] = df["baseline_acc"].rank(ascending=False, method="min")
    df["mtt_rank"] = df["mtt_acc"].rank(ascending=False, method="min")
    df["dm_rank"] = df["dm_acc"].rank(ascending=False, method="min")


    k_values = range(5, K_max + 1, 5)
    mtt_recalls = [topK_recall(df, k, "mtt_rank") for k in k_values]
    dm_recalls = [topK_recall(df, k, "dm_rank") for k in k_values]
    random_line = [k / len(df) for k in k_values]

    plt.figure(figsize=(6, 6))

    plt.plot(k_values, mtt_recalls, label="MTT Recall", color=steel, linewidth=2)
    plt.plot(k_values, dm_recalls, label="DM Recall", color=dm_color, linewidth=2)
    plt.plot(k_values, random_line, label="Random Chance", color=error_color, linestyle="--")

    plt.xlabel("K")
    plt.ylabel("Top-K Recall")

    plt.title("Top-K Recall Curve for DM and MTT")

    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlim(0, K_max)
    plt.ylim(0, 1.0)

    plt.tight_layout()
    plt.show()

run_comparison("MTT_synthetic_metrics_FULL.csv", "DM_synthetic_metrics_0-1000.csv", "cifar10_tss_full_metrics.csv", K_max=500)