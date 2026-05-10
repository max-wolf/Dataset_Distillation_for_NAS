import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def topK_recall(df, k, pred_rank_col, true_rank_col="baseline_rank"):
    top_true = set(df[df[true_rank_col] <= k]["arch_id"])
    top_pred = set(df[df[pred_rank_col] <= k]["arch_id"])
    return len(top_true & top_pred) / k


def summary_stats(x, name):
    print(f"\n--- {name} ---")
    print("Mean   :", np.mean(x))
    print("Median :", np.median(x))
    print("Std    :", np.std(x))
    print("Min    :", np.min(x))
    print("Max    :", np.max(x))
    print("Q1     :", np.percentile(x, 25))
    print("Q3     :", np.percentile(x, 75))

steel = "steelblue"

def styled_boxplot(data, labels, title):
    plt.figure(figsize=(6, 5))

    bp = plt.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
        meanline=True
    )

    for box in bp["boxes"]:
        box.set(color=steel, linewidth=1.5)

    for whisker in bp["whiskers"]:
        whisker.set(color=steel)

    for cap in bp["caps"]:
        cap.set(color=steel)

    for median in bp["medians"]:
        median.set(color=steel, linewidth=2)

    for flier in bp["fliers"]:
        flier.set(marker="o", markeredgecolor=steel, alpha=0.6, markersize=4)

    for mean in bp["means"]:
        mean.set(color="slategray", linewidth=1, linestyle="--")

    plt.title(title)
    plt.ylabel("Accuracy")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.axhline(y=10, color="red", linestyle="--", linewidth=1, label="Random (10%)")
    plt.legend()

    plt.show()


def evaluate_nas(
    synthetic_csv,
    baseline_csv,
    K=100,
    K_values=range(10, 501, 10),
    plot=True,
    color1=steel,
    color2="red",
    ):

    df_synthetic = pd.read_csv(synthetic_csv)
    df_baseline = pd.read_csv(baseline_csv)

    max_arch_id = df_synthetic["arch_id"].max()
    df_synthetic = df_synthetic[df_synthetic["arch_id"] <= max_arch_id]
    df_baseline = df_baseline[df_baseline["arch_id"] <= max_arch_id]

    df_baseline = df_baseline.rename(columns={
        "test_accuracy": "baseline_test_accuracy"
    })

    df_synthetic = df_synthetic.rename(columns={
        "test_accuracy": "synthetic_test_accuracy",
        "max_test_accuracy": "synthetic_max_test_accuracy"
    })

    df_baseline = df_baseline[["arch_id", "baseline_test_accuracy"]]

    df = pd.merge(df_synthetic, df_baseline, on="arch_id", how="inner")


    df["baseline_rank"] = df["baseline_test_accuracy"].rank(ascending=False, method="min")
    df["synthetic_rank"] = df["synthetic_test_accuracy"].rank(ascending=False, method="min")
    df["synthetic_max_rank"] = df["synthetic_max_test_accuracy"].rank(ascending=False, method="min")

    df_topK = df[df["baseline_rank"] <= K].copy()


    results = {}

    results["pearson_test"] = df["baseline_test_accuracy"].corr(df["synthetic_test_accuracy"])
    results["spearman_test"] = df["baseline_test_accuracy"].corr(df["synthetic_test_accuracy"], method="spearman")
    results["kendall_test"] = df["baseline_test_accuracy"].corr(df["synthetic_test_accuracy"], method="kendall")

    results["kendall_topK_test"] = df_topK["baseline_test_accuracy"].corr(
        df_topK["synthetic_test_accuracy"], method="kendall"
    )

    results["recall_test"] = topK_recall(df, K, "synthetic_rank")

    results["pearson_max"] = df["baseline_test_accuracy"].corr(df["synthetic_max_test_accuracy"])
    results["spearman_max"] = df["baseline_test_accuracy"].corr(df["synthetic_max_test_accuracy"], method="spearman")
    results["kendall_max"] = df["baseline_test_accuracy"].corr(df["synthetic_max_test_accuracy"], method="kendall")

    results["kendall_topK_max"] = df_topK["baseline_test_accuracy"].corr(
        df_topK["synthetic_max_test_accuracy"], method="kendall"
    )

    results["recall_max"] = topK_recall(df, K, "synthetic_max_rank")

    recall_test_curve = []
    recall_max_curve = []

    for k in K_values:
        recall_test_curve.append(topK_recall(df, k, "synthetic_rank"))
        recall_max_curve.append(topK_recall(df, k, "synthetic_max_rank"))

    if plot:

        random_curve = [k / len(df) for k in K_values]

        plt.figure(figsize=(6, 6))
        plt.plot(K_values, recall_test_curve, label="Ranked by Test Accuracy", color=color1)
        plt.plot(K_values, recall_max_curve, label="Ranked by Max Test Accuracy", linestyle="--", color=color1)
        plt.plot(K_values, random_curve, linestyle="--", label="Random", color=color2)

        plt.xlabel("K")
        plt.ylabel("Top-K Recall")
        plt.title("Top-K Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.show()


        plt.figure(figsize=(6, 6))
        plt.scatter(df["baseline_rank"], df["synthetic_rank"], s=5, alpha=0.6, color=color1)
        plt.plot([1, df["baseline_rank"].max()], [1, df["baseline_rank"].max()], color=color2)
        plt.xlabel("Baseline Rank")
        plt.ylabel("Synthetic Rank")
        plt.title("Rank Comparison (Ranked by Accuracy)")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.scatter(df["baseline_rank"], df["synthetic_max_rank"], s=5, alpha=0.6, color=color1)
        plt.plot([1, df["baseline_rank"].max()], [1, df["baseline_rank"].max()], color=color2)
        plt.xlabel("Baseline Rank")
        plt.ylabel("Synthetic Max Rank")
        plt.title("Rank Comparison (Ranked by Max Accuracy)")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.scatter(df["baseline_test_accuracy"], df["synthetic_test_accuracy"], alpha=0.1, color=color1, s=5)
        plt.xlabel("Baseline Accuracy")
        plt.ylabel("Synthetic Accuracy")
        plt.title("Accuracy Comparison")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.show()

        # Accuracy scatter (max)
        plt.figure(figsize=(6, 6))
        plt.scatter(df["baseline_test_accuracy"], df["synthetic_max_test_accuracy"], alpha=0.1, s=5, color=color1)
        plt.xlabel("Baseline Accuracy")
        plt.ylabel("Synthetic Max Accuracy")
        plt.title("Accuracy Comparison (Max Accuracy)")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.show()

    return results

# results = evaluate_nas(
#     synthetic_csv="MTT_synthetic_metrics_FULL.csv",
#     baseline_csv="cifar10_tss_full_metrics.csv",
# )
#
# print(results)


# # CODE FOR distribution of models (with random line)
# df_synthetic = pd.read_csv("MTT_synthetic_metrics_FULL.csv")
#
# max_arch_id = df_synthetic["arch_id"].max()
# df_synthetic = df_synthetic[df_synthetic["arch_id"] <= max_arch_id]
#
# df_synthetic = df_synthetic.rename(columns={
#     "test_accuracy": "synthetic_test_accuracy",
#     "max_test_accuracy": "synthetic_max_test_accuracy"
# })
#
# plt.figure(figsize=(6, 6))
# plt.hist(df_synthetic["synthetic_test_accuracy"], bins=50, alpha=0.5)
#
# plt.axvline(x=10, linestyle="--", color="red", label="Random (10%)")
#
# plt.xlabel("Accuracy")
# plt.ylabel("Count")
# plt.title("Distribution of Models")
#
# plt.legend()
# plt.show()


#CODE FOR DISTRIBUTION ANALYSIS (boxplots and stats)
df_synthetic = pd.read_csv("MTT_synthetic_metrics_FULL.csv")
df_baseline = pd.read_csv("cifar10_tss_full_metrics.csv")

# align range if needed
max_arch_id = df_synthetic["arch_id"].max()

df_synthetic = df_synthetic[df_synthetic["arch_id"] <= max_arch_id]
df_baseline = df_baseline[df_baseline["arch_id"] <= max_arch_id]

df_synthetic = df_synthetic.rename(columns={
    "test_accuracy": "synthetic_acc",
    "max_test_accuracy": "synthetic_max_acc"
})

df_baseline = df_baseline.rename(columns={
    "test_accuracy": "baseline_acc"
})

# merge
df = pd.merge(df_synthetic, df_baseline, on="arch_id", how="inner")


# summary_stats(df["baseline_acc"], "Baseline Accuracy")
# summary_stats(df["synthetic_acc"], "Synthetic Accuracy")
# summary_stats(df["synthetic_max_acc"], "Synthetic Max Accuracy")


# styled_boxplot(
#     [
#         df["baseline_acc"],
#         df["synthetic_acc"],
#         df["synthetic_max_acc"]
#     ],
#     ["Baseline", "Synthetic", "Synthetic Max"],
#     "Accuracy Distributions"
# )
#
# styled_boxplot(
#     [
#         df["synthetic_acc"],
#         df["synthetic_max_acc"]
#     ],
#     ["Synthetic", "Synthetic Max"],
#     "Synthetic Accuracy Distributions"
# )

# df["baseline_rank"] = df["baseline_acc"].rank(ascending=False, method="min")
# df["synthetic_rank"] = df["synthetic_acc"].rank(ascending=False, method="min")
#
# df["rank_error"] =  df["baseline_rank"] - df["synthetic_rank"]
#
# avg_error = df["rank_error"].mean()
# mae_rank = df["rank_error"].abs().mean()
#
# print(f"Average Rank Error: {avg_error:.2f}")
# print(f"Mean Absolute Rank Error: {mae_rank:.2f}")
#
# K = 100
# df_top = df[df["synthetic_rank"] <= K]
# df_others = df[df["synthetic_rank"] > K]
#
# plt.figure(figsize=(10, 6))

# plt.scatter(
#     df_others["baseline_rank"],
#     df_others["rank_error"],
#     alpha=0.3,
#     s=8,
#     color=steel,
#     label="Other Models"
# )
#
# # Highlight the Top K models in red
# plt.scatter(
#     df_top["baseline_rank"],
#     df_top["rank_error"],
#     alpha=0.8,
#     s=12,
#     color="red",
#     label=f"Top {K} Baseline Models"
# )
#
#
# plt.title("Rank Error across Architectures")
# plt.xlabel("Baseline Rank (1 = Best)")
# plt.ylabel("Rank Error")
# plt.grid(True, linestyle="--", alpha=0.4)
# plt.legend()
#
# plt.tight_layout()
# plt.show()
#
# df_sorted_by_pred = df.sort_values("synthetic_acc", ascending=False).reset_index(drop=True)
#
# df_sorted_by_pred['cumulative_max_baseline'] = df_sorted_by_pred['baseline_acc'].cummax()
# # This shows the 'Average Baseline Accuracy' of the models picked so far
# df_sorted_by_pred['cumulative_mean_baseline'] = df_sorted_by_pred['baseline_acc'].expanding().mean()
#
# plt.figure(figsize=(10, 6))
#
# plt.scatter(df_sorted_by_pred.index, df_sorted_by_pred['baseline_acc'],
#             color=steel, s=1, alpha=0.2, label="Individual Architecture (Baseline Acc)")
#
# plt.plot(df_sorted_by_pred.index, df_sorted_by_pred['cumulative_max_baseline'],
#          color="red", linewidth=2, label="Cumulative Max (Best Found)")
#
# plt.plot(df_sorted_by_pred.index, df_sorted_by_pred['cumulative_mean_baseline'],
#          color="black", linestyle="--", label="Cumulative Mean Accuracy")
#
# plt.title("Performance Curve: Baseline Accuracy vs. Synthetic Rank Order")
# plt.xlabel("Architectures Ranked by Synthetic Metric (Top 1 -> Bottom)")
# plt.ylabel("Baseline Accuracy (%)")
# plt.ylim(0, 100)
# plt.grid(True, alpha=0.3)
# plt.legend()
#
# plt.show()