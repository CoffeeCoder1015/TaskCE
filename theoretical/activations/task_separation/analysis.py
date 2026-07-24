"""Inspect whether task labels organize one activation representation."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from theoretical.activations.source import load_activation


SNLI_LABELS = ("entailment", "neutral", "contradiction")


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
