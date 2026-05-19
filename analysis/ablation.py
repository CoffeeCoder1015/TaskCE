from dataclasses import dataclass
from math import ceil
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PERCENTILES = tuple(step / 10 for step in range(1, 11))


@dataclass(frozen=True)
class AblationRunConfig:
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES


@dataclass(frozen=True)
class AblationRunResult:
    output_dir: Path
    result_paths: dict[str, Path]
    plot_paths: dict[str, Path]
    baseline_accuracy: float


def accuracy_from_stats(stats):
    total = stats["success"] + stats["fail"] + stats["reject"]
    return stats["success"] / total


def plot_ablation_results(good_csv_path, bad_csv_path, iou_csv_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    good = pd.read_csv(good_csv_path)
    bad = pd.read_csv(bad_csv_path)
    iou = pd.read_csv(iou_csv_path)

    good_bad_path = output_dir / "ablation_good_vs_bad.png"
    plt.figure(figsize=(8, 5))
    plt.plot(good["percentile"] * 100, good["accuracy"] * 100, marker="o", label="Good")
    plt.plot(bad["percentile"] * 100, bad["accuracy"] * 100, marker="s", label="Bad")
    plt.xlabel("Cumulative neurons ablated (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Good vs Bad Neuron Ablation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(good_bad_path)
    plt.close()

    combined_path = output_dir / "ablation_combined.png"
    plt.figure(figsize=(8, 5))
    plt.plot(good["percentile"] * 100, good["accuracy"] * 100, marker="o", label="Good")
    plt.plot(bad["percentile"] * 100, bad["accuracy"] * 100, marker="s", label="Bad")
    plt.plot(iou["percentile"] * 100, iou["accuracy"] * 100, marker="^", label="IoU ranked")
    plt.xlabel("Cumulative neurons ablated (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Neuron Ablation Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(combined_path)
    plt.close()

    return {
        "good_bad": good_bad_path,
        "combined": combined_path,
    }


def run_ablation(analysis_result, inference_engine, output_dir=None, config=None):
    config = config or AblationRunConfig()
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else analysis_result.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: consume the dataframe breakdown returned by run_ablation_analysis().
    groups = {
        "good": analysis_result.good_neurons["neuron"].astype(int).tolist(),
        "bad": analysis_result.bad_neurons["neuron"].astype(int).tolist(),
        "iou_ranked": analysis_result.iou_ranked_neurons["neuron"].astype(int).tolist(),
    }

    # Stage 2: run the unablated baseline once.
    baseline_stats = inference_engine(None)
    baseline_accuracy = accuracy_from_stats(baseline_stats)

    # Stage 3: run cumulative ablations for each precomputed group.
    result_paths = {}
    for group_name, neurons in groups.items():
        records = []
        last_count = 0
        for percentile in config.percentiles:
            count = max(1, ceil(len(neurons) * percentile))
            if count == last_count:
                continue
            neuron_set = neurons[:count]
            stats = inference_engine(neuron_set)
            accuracy = accuracy_from_stats(stats)
            records.append(
                {
                    "group": group_name,
                    "percentile": percentile,
                    "n_neurons": len(neuron_set),
                    "accuracy": accuracy,
                    "accuracy_delta": accuracy - baseline_accuracy,
                    "success": stats["success"],
                    "fail": stats["fail"],
                    "reject": stats["reject"],
                }
            )
            last_count = count

        output_name = "iou" if group_name == "iou_ranked" else group_name
        result_path = output_dir / f"ablation_cumulative_{output_name}.csv"
        pd.DataFrame(records).to_csv(result_path, index=False)
        result_paths[group_name] = result_path

    # Stage 4: plot the ablation outputs.
    plot_paths = plot_ablation_results(
        result_paths["good"],
        result_paths["bad"],
        result_paths["iou_ranked"],
        output_dir,
    )

    return AblationRunResult(
        output_dir=output_dir,
        result_paths=result_paths,
        plot_paths=plot_paths,
        baseline_accuracy=baseline_accuracy,
    )
