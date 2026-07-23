from dataclasses import dataclass
from math import ceil
import os
from pathlib import Path
import re

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PERCENTILES = tuple(step / 10 for step in range(1, 11))
DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"
DEFAULT_LORA_REPOSITORY = "Heroi/multitune-lora-backup"
DEFAULT_LAYER = "model.layers.8.feed_forward"
DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
COMPOSITIONAL_DATA_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "compositional_explanations"
    / "data"
)


def compositional_result_path(task_name):
    """Return the upstream compositional data file consumed by this analysis."""
    return COMPOSITIONAL_DATA_DIRECTORY / f"{task_name}_beam_results.csv"


@dataclass(frozen=True)
class AblationRunConfig:
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES


@dataclass(frozen=True)
class AblationRunResult:
    output_dir: Path
    result_paths: dict[str, Path]
    plot_paths: dict[str, Path]
    baseline_accuracy: float


@dataclass(frozen=True)
class AblationStudy:
    name: str
    dataset_name: str
    split: str
    labels: tuple[str, ...]
    class_token_ids: dict[str, int]

    @property
    def weight_column_names(self):
        return tuple(
            f"weight_{label.replace(' ', '_')}"
            for label in self.labels
        )


STUDIES = {
    "snli": AblationStudy(
        name="snli",
        dataset_name="snli",
        split="validation",
        labels=("entailment", "neutral", "contradiction"),
        class_token_ids={
            "entailment": 806,
            "neutral": 25919,
            "contradiction": 10913,
        },
    ),
    "claim": AblationStudy(
        name="claim",
        dataset_name="tals/vitaminc",
        split="validation[:10_000]",
        labels=("supports", "refutes", "not enough info"),
        class_token_ids={
            "supports": 56744,
            "refutes": 1891,
            "not enough info": 2897,
        },
    ),
}


def format_snli(example):
    example["prompt"] = [
        {
            "role": "system",
            "content": (
                "Determine the relationship between the `premise` and "
                "`hypothesis` and respond with `entailment`, `neutral`, "
                "or `contradiction`."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Premise: {example['premise']}\n"
                f"Hypothesis: {example['hypothesis']}"
            ),
        },
    ]
    example["answer"] = STUDIES["snli"].labels[int(example["label"])]
    return example


def format_claim(example):
    example["prompt"] = [
        {
            "role": "system",
            "content": (
                "Determine the relationship between the `claim` and "
                "`evidence` and respond with `supports`, `refutes`, or "
                "`not enough info`."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Evidence: {example['evidence']}\nClaim: {example['claim']}"
            ),
        },
    ]
    example["answer"] = str(example["label"]).strip().lower()
    return example


FORMATTERS = {
    "snli": format_snli,
    "claim": format_claim,
}


def latest_task_checkpoint(repository, task_name, *, token):
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=repository,
            repo_type="model",
            token=token,
        )
    )
    checkpoints = [
        path
        for path in (root / task_name).iterdir()
        if path.is_dir()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No LoRA checkpoint found for {task_name}.")

    def sort_key(path):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        return (int(match.group(1)) if match else -1, path.name)

    return max(checkpoints, key=sort_key)


def _load_dataset(name, *, split):
    from datasets import load_dataset

    return load_dataset(name, split=split)


def _select_neurons(formula_path, *, output_directory, weight_column_names):
    from theoretical.ablation.selection import run_ablation_analysis

    return run_ablation_analysis(
        formula_path,
        output_dir=output_directory,
        weight_column_names=weight_column_names,
    )


def _build_inference_engine(
    *,
    model_id,
    task_name,
    dataset,
    formatter,
    layer,
    class_token_ids,
    lora_path,
):
    from theoretical.ablation.inference import (
        AblationInferenceEngine,
        AblationTaskConfig,
    )

    task = AblationTaskConfig(
        name=task_name,
        dataset=dataset,
        data_formatter=formatter,
    )
    return AblationInferenceEngine(
        model_id=model_id,
        task=task,
        layer=layer,
        class_token_ids=class_token_ids,
        lora_path=lora_path,
    )


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


def run(
    task_name,
    *,
    formula_path=None,
    output_directory=None,
    model_id=DEFAULT_MODEL_ID,
    lora_repository=DEFAULT_LORA_REPOSITORY,
    lora_token=None,
    layer=DEFAULT_LAYER,
):
    study = STUDIES[task_name]
    formula_path = (
        Path(formula_path)
        if formula_path is not None
        else compositional_result_path(task_name)
    )
    output_directory = (
        Path(output_directory)
        if output_directory is not None
        else DATA_DIRECTORY
    )
    analysis_result = _select_neurons(
        formula_path,
        output_directory=output_directory,
        weight_column_names=study.weight_column_names,
    )

    dataset = _load_dataset(study.dataset_name, split=study.split)
    checkpoint = latest_task_checkpoint(
        lora_repository,
        task_name,
        token=(
            os.environ.get("HF_TOKEN")
            if lora_token is None
            else lora_token
        ),
    )
    inference_engine = _build_inference_engine(
        model_id=model_id,
        task_name=task_name,
        dataset=dataset,
        formatter=FORMATTERS[task_name],
        layer=layer,
        class_token_ids=study.class_token_ids,
        lora_path=checkpoint,
    )
    try:
        return run_ablation(
            analysis_result,
            inference_engine,
            output_dir=output_directory,
        )
    finally:
        inference_engine.close()
