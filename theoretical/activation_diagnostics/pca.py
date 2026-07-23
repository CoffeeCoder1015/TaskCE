"""Plot PCA views from raw experimental activation tensor files."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA


SNLI_LABELS = ("entailment", "neutral", "contradiction")


@dataclass(frozen=True)
class PcaTask:
    dataset_name: str
    split: str
    label_field: str
    label_names: tuple[str, ...] | None = None

    def labels(self, dataset):
        labels = dataset[self.label_field]
        if self.label_names is None:
            return [str(label).strip().lower() for label in labels]
        return [self.label_names[int(label)] for label in labels]


TASKS = {
    "snli": PcaTask("snli", "validation", "label", SNLI_LABELS),
    "claim": PcaTask(
        "tals/vitaminc",
        "validation[:10_000]",
        "label",
    ),
    "fallacy": PcaTask(
        "tasksource/logical-fallacy",
        "dev",
        "logical_fallacies",
    ),
}


def load_activation(path: str | Path) -> torch.Tensor:
    activation = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(activation, torch.Tensor):
        raise TypeError("Experimental activation data must be a tensor.")
    if activation.ndim != 2:
        raise ValueError(
            "Experimental activation data must have shape [examples, neurons], "
            f"got {tuple(activation.shape)}."
        )
    return activation.detach().cpu().to(torch.float32)


def ordered_labels(labels):
    seen = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return seen


def plot_pca(states, labels, output_path, title):
    labels = list(labels)
    print(f"{title}: states={tuple(states.shape)}")
    print(f"{title}: label counts={Counter(labels)}")

    if states.shape[0] < 2 or states.shape[1] < 2:
        print(f"{title}: skipping PCA; need at least 2 examples and 2 activation dims.")
        return None

    pca = PCA(n_components=2)
    points = pca.fit_transform(states.numpy())

    plt.figure(figsize=(8, 6))
    for label in ordered_labels(labels):
        indices = [
            index
            for index, example_label in enumerate(labels)
            if example_label == label
        ]
        plt.scatter(
            points[indices, 0],
            points[indices, 1],
            label=label,
            alpha=0.7,
            s=18,
        )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"{title}: saved {output_path}")
    print(f"{title}: explained variance={pca.explained_variance_ratio_}")
    return output_path


def run(
    task_name: str,
    activation_path: str | Path,
    *,
    output_path: str | Path,
):
    task = TASKS[task_name]
    states = load_activation(activation_path)
    dataset = load_dataset(task.dataset_name, split=task.split)
    labels = task.labels(dataset)
    if len(labels) != states.shape[0]:
        raise ValueError(
            f"{task_name} dataset rows do not match activation rows: "
            f"{len(labels)} != {states.shape[0]}."
        )

    return plot_pca(
        states,
        labels,
        Path(output_path),
        f"{task_name} activation PCA",
    )
