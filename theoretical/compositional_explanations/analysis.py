"""Run the compositional explanation analysis."""

import gc
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .construction.tokenizer.orchestration import (
    get_tokenizer,
)
from .construction.vectors import (
    construct_feature_vectors,
)
from .postprocessing import (
    prune_min_acts,
    threshold,
)
from .search.algorithm import (
    searchConfig,
    search_all,
)


DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"
DEFAULT_LORA_REPOSITORY = "Heroi/multitune-lora-backup"
ACTIVATION_DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data"
DATA_DIRECTORY = Path(__file__).resolve().parent / "data"


def activation_path(
    model_id: str,
    dataset_name: str,
    layer: str,
    *,
    task_name: str | None = None,
    checkpoint_name: str | None = None,
) -> Path:
    """Return one generated activation file."""
    model_directory = ACTIVATION_DATA_DIRECTORY.joinpath(*model_id.split("/"))

    if task_name is None:
        # Select the base capture directory.
        capture_directory = model_directory / "base"
    elif checkpoint_name is not None:
        # Select the requested task checkpoint directory.
        capture_directory = model_directory / task_name / checkpoint_name
    else:
        # Select the numerically latest task checkpoint directory.
        task_directory = model_directory / task_name
        checkpoint_directories = []

        for checkpoint_directory in task_directory.iterdir():
            if checkpoint_directory.is_dir():
                checkpoint_directories.append(checkpoint_directory)

        capture_directory = max(
            checkpoint_directories,
            key=lambda checkpoint: int(
                checkpoint.name.removeprefix("checkpoint-")
            ),
        )

    activation_name = f"{dataset_name}_activation.pt"

    exact_layer_directory = capture_directory / layer
    if exact_layer_directory.is_dir():
        return exact_layer_directory / activation_name

    for saved_layer_directory in capture_directory.iterdir():
        if saved_layer_directory.name.endswith(f".{layer}"):
            return saved_layer_directory / activation_name

    raise KeyError(layer)


def result_path(task_name: str) -> Path:
    """Return the compositional data file owned by this analysis."""
    return DATA_DIRECTORY / f"{task_name}_beam_results.csv"


@dataclass(frozen=True)
class CompositionalTask:
    name: str
    dataset_name: str
    split: str
    feature_columns: tuple[str, ...]
    tokenizer_name: str
    tokenizer_uses_pos: bool
    alpha: float
    min_acts: int
    class_token_ids: dict[str, int]

    @property
    def weight_column_names(self) -> tuple[str, ...]:
        return tuple(
            f"weight_{label.replace(' ', '_')}"
            for label in self.class_token_ids
        )


TASKS = {
    "snli": CompositionalTask(
        name="snli",
        dataset_name="snli",
        split="validation",
        feature_columns=("premise", "hypothesis"),
        tokenizer_name="spacy-pos-snli-features",
        tokenizer_uses_pos=False,
        alpha=0.055,
        min_acts=500,
        class_token_ids={
            "entailment": 806,
            "neutral": 25919,
            "contradiction": 10913,
        },
    ),
    "claim": CompositionalTask(
        name="claim",
        dataset_name="tals/vitaminc",
        split="validation[:10_000]",
        feature_columns=("claim", "evidence"),
        tokenizer_name="spacy-pos-claim-features",
        tokenizer_uses_pos=False,
        alpha=0.052,
        min_acts=500,
        class_token_ids={
            "supports": 56744,
            "refutes": 1891,
            "not enough info": 2897,
        },
    ),
}


def load_activation(path: str | Path) -> torch.Tensor:
    """Load one raw experimental activation matrix from disk."""
    activation = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(activation, torch.Tensor):
        raise TypeError("Experimental activation data must be a tensor.")
    if activation.ndim != 2:
        raise ValueError(
            "Experimental activation data must have shape [examples, neurons], "
            f"got {tuple(activation.shape)}."
        )
    return activation.detach().cpu().to(torch.float32)


def classification_weights_from_model(model, class_token_ids):
    """Select analysis-supporting class weights from a loaded model."""
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Model does not expose weighted output embeddings.")
    class_ids = list(class_token_ids.values())
    return (
        output_embeddings.weight.detach()
        .to(device="cpu", dtype=torch.float32)[class_ids]
        .T
    )


def latest_task_checkpoint(
    lora_repository: str,
    task_name: str,
    *,
    token: str | bool | None,
) -> Path:
    lora_root = Path(
        snapshot_download(
            repo_id=lora_repository,
            repo_type="model",
            token=token,
        )
    )
    task_directory = lora_root / task_name
    checkpoints = [path for path in task_directory.iterdir() if path.is_dir()]
    if not checkpoints:
        raise FileNotFoundError(f"No LoRA checkpoint found for {task_name}.")

    def sort_key(path):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        return (int(match.group(1)) if match else -1, path.name)

    return max(checkpoints, key=sort_key)


def load_classification_weights(
    task: CompositionalTask,
    *,
    model_id: str,
    lora_repository: str,
    lora_token: str | bool | None,
) -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    checkpoint = latest_task_checkpoint(
        lora_repository,
        task.name,
        token=lora_token,
    )
    model = PeftModel.from_pretrained(model, checkpoint)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

    try:
        return classification_weights_from_model(model, task.class_token_ids)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run(
    task_name: str,
    activation_path: str | Path,
    *,
    output_path: str | Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    lora_repository: str = DEFAULT_LORA_REPOSITORY,
    lora_token: str | bool | None = None,
    num_workers: int = 1,
    config: searchConfig | None = None,
) -> Path:
    """Build and save compositional explanations for one configured task."""
    task = TASKS[task_name]
    activations = load_activation(activation_path)
    dataset = load_dataset(task.dataset_name, split=task.split)
    if len(dataset) != activations.shape[0]:
        raise ValueError(
            f"{task.name} dataset rows do not match activation rows: "
            f"{len(dataset)} != {activations.shape[0]}."
        )

    feature_tokenizer = get_tokenizer(
        task.tokenizer_name,
        dataset,
        task.feature_columns,
        enable_pos=task.tokenizer_uses_pos,
    )
    features = construct_feature_vectors(
        dataset,
        feature_tokenizer,
        task.feature_columns,
    )

    binary_activations = threshold(activations, alpha=task.alpha)
    searchable_activations = prune_min_acts(
        binary_activations,
        min_acts=task.min_acts,
    )
    search_results = search_all(
        searchable_activations.matrix,
        features,
        num_workers=num_workers,
        config=config or searchConfig(
            formula_length=5,
            pruned_queue_size=10,
            max_iterations=10,
            length_penalty=0.0,
        ),
    )

    classification_weights = load_classification_weights(
        task,
        model_id=model_id,
        lora_repository=lora_repository,
        lora_token=(
            os.environ.get("HF_TOKEN")
            if lora_token is None
            else lora_token
        ),
    )
    if classification_weights.shape[0] != activations.shape[1]:
        raise ValueError(
            "Classification weight and activation neuron counts differ: "
            f"{classification_weights.shape[0]} != {activations.shape[1]}."
        )

    results = pd.DataFrame(
        classification_weights.tolist(),
        columns=task.weight_column_names,
    )
    results.insert(0, "iou", 0.0)
    results.insert(0, "formula", "LOW_ACTS_PRUNED")
    results.insert(0, "neuron", range(len(results)))
    results = results.set_index("neuron", drop=False)

    for result in search_results:
        neuron_id = searchable_activations.neuron_ids[
            result.activation_index
        ]
        results.at[neuron_id, "formula"] = result.best_formula
        results.at[neuron_id, "iou"] = result.best_score

    results = results.sort_values(
        "iou",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    output_path = (
        Path(output_path)
        if output_path is not None
        else result_path(task_name)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return output_path
