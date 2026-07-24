"""Inspect whether task labels organize one activation representation."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_DIRECTORY = Path("data")
ACTIVATION_DATA_DIRECTORY = Path(__file__).resolve().parents[3] / "data"
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")
SNLI_LABELS = ("entailment", "neutral", "contradiction")


def activation_path(
    model_id: str,
    dataset_name: str,
    layer: str,
    *,
    task_name: str | None = None,
    checkpoint_name: str | None = None,
) -> Path:
    """Return one existing activation tensor path."""
    model_directory = ACTIVATION_DATA_DIRECTORY.joinpath(*model_id.split("/"))

    if task_name is None:
        capture_directory = model_directory / "base"
    elif checkpoint_name is not None:
        capture_directory = model_directory / task_name / checkpoint_name
    else:
        task_directory = model_directory / task_name
        checkpoints = [
            (int(match.group(1)), path)
            for path in task_directory.iterdir()
            if path.is_dir()
            and (match := CHECKPOINT_PATTERN.fullmatch(path.name)) is not None
        ]
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint-N directories found under {task_directory}."
            )
        capture_directory = max(checkpoints, key=lambda item: item[0])[1]

    activation_name = f"{dataset_name}_activation.pt"
    exact_path = capture_directory / layer / activation_name
    if exact_path.is_file():
        return exact_path

    suffix_matches = [
        directory / activation_name
        for directory in capture_directory.iterdir()
        if directory.is_dir()
        and directory.name.endswith(f".{layer}")
        and (directory / activation_name).is_file()
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if suffix_matches:
        raise ValueError(
            f"Layer {layer!r} matched multiple saved activation directories: "
            f"{[str(path.parent) for path in suffix_matches]}"
        )
    raise KeyError(layer)


def load_activation(path):
    """Load one capture as a CPU float tensor shaped [examples, neurons]."""
    import torch

    activation = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(activation, torch.Tensor):
        raise TypeError("Experimental activation data must be a tensor.")
    if activation.ndim != 2:
        raise ValueError(
            "Experimental activation data must have shape [examples, neurons], "
            f"got {tuple(activation.shape)}."
        )
    return activation.detach().cpu().to(torch.float32)


@dataclass(frozen=True)
class TaskLabels:
    dataset_name: str
    split: str
    label_field: str
    label_names: tuple[str, ...] | None = None

    def from_dataset(self, dataset):
        values = dataset[self.label_field]
        if self.label_names is None:
            return [str(value).strip().lower() for value in values]
        return [self.label_names[int(value)] for value in values]


TASKS = {
    "snli": TaskLabels("snli", "validation", "label", SNLI_LABELS),
    "claim": TaskLabels("tals/vitaminc", "validation[:10_000]", "label"),
    "fallacy": TaskLabels(
        "tasksource/logical-fallacy",
        "dev",
        "logical_fallacies",
    ),
}


def run(task_name, activation_path):
    """Load one capture and its aligned labels, then return the PCA result."""
    task = TASKS[task_name]
    states = load_activation(activation_path)
    dataset = _load_dataset(task.dataset_name, task.split)
    labels = task.from_dataset(dataset)
    return analyze_task_separation(states, labels)


def analyze_task_separation(states, labels):
    """Return the complete two-component label-projection result."""
    from sklearn.decomposition import PCA

    states = np.asarray(states, dtype=float)
    labels = [str(label) for label in labels]
    if states.ndim != 2:
        raise ValueError(f"expected 2D activation matrix, got {states.shape}")
    if len(labels) != states.shape[0]:
        raise ValueError(
            "dataset rows do not match activation rows: "
            f"{len(labels)} != {states.shape[0]}"
        )
    if states.shape[0] < 2 or states.shape[1] < 2:
        raise ValueError(
            "task separation requires at least 2 examples and 2 activation dimensions"
        )

    pca = PCA(n_components=2)
    points = pca.fit_transform(states)
    return {
        "points": points,
        "labels": labels,
        "label_counts": dict(Counter(labels)),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_,
    }


def _load_dataset(dataset_name, split):
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split)
