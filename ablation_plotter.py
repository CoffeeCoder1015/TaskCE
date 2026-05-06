import os

import pandas as pd


def load_ablation_results(good_path, bad_path, iou_path):
    return (
        pd.read_csv(good_path),
        pd.read_csv(bad_path),
        pd.read_csv(iou_path),
    )


def _configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_good_bad_comparison(df_good, df_bad, output_path):
    plt = _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        df_good["percentile"] * 100,
        df_good["accuracy"] * 100,
        marker="o",
        linestyle="-",
        label="Good Neurons",
    )
    ax.plot(
        df_bad["percentile"] * 100,
        df_bad["accuracy"] * 100,
        marker="s",
        linestyle="--",
        label="Bad Neurons",
    )
    ax.set_title("Comparison: Good vs Bad Neuron Cumulative Ablation Effect")
    ax.set_xlabel("Percentile of Neurons Ablated (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_combined_comparison(df_good, df_bad, df_iou, output_path):
    plt = _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        df_good["percentile"] * 100,
        df_good["accuracy"] * 100,
        marker="o",
        linewidth=2.5,
        label="Good Neurons",
        color="green",
        markersize=8,
    )
    ax.plot(
        df_bad["percentile"] * 100,
        df_bad["accuracy"] * 100,
        marker="s",
        linewidth=2.5,
        label="Bad Neurons",
        color="red",
        markersize=8,
    )
    ax.plot(
        df_iou["percentile"] * 100,
        df_iou["accuracy"] * 100,
        marker="^",
        linewidth=2.5,
        label="IoU Ranked (All)",
        color="blue",
        markersize=8,
    )
    ax.set_xlabel("Percentile of Neurons Ablated (%)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Ablation Analysis: Good vs Bad vs IoU Ranked")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_ablation_results(
    good_path,
    bad_path,
    iou_path,
    output_dir,
    good_bad_name="ablation_good_vs_bad.png",
    combined_name="ablation_good_bad_iou.png",
):
    os.makedirs(output_dir, exist_ok=True)
    df_good, df_bad, df_iou = load_ablation_results(good_path, bad_path, iou_path)

    good_bad_path = os.path.join(output_dir, good_bad_name)
    combined_path = os.path.join(output_dir, combined_name)

    plot_good_bad_comparison(df_good, df_bad, good_bad_path)
    plot_combined_comparison(df_good, df_bad, df_iou, combined_path)

    return {
        "good_bad": good_bad_path,
        "combined": combined_path,
    }
