import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from capture import load_captured_activations


def ordered_labels(labels):
    seen = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return seen


def plot_pca(capture_result, output_path, title):
    states = capture_result.states
    labels = list(capture_result.labels)

    print(f"{title}: states={tuple(states.shape)} layer={capture_result.layer}")
    print(f"{title}: label counts={Counter(labels)}")

    if states.shape[0] < 2 or states.shape[1] < 2:
        print(f"{title}: skipping PCA; need at least 2 examples and 2 activation dims.")
        return None

    pca = PCA(n_components=2)
    points = pca.fit_transform(states.numpy())

    plt.figure(figsize=(8, 6))
    for label in ordered_labels(labels):
        x_values = [
            points[index, 0]
            for index, example_label in enumerate(labels)
            if example_label == label
        ]
        y_values = [
            points[index, 1]
            for index, example_label in enumerate(labels)
            if example_label == label
        ]
        plt.scatter(x_values, y_values, label=label, alpha=0.7, s=18)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"{title}: saved {output_path}")
    print(f"{title}: explained variance={pca.explained_variance_ratio_}")
    return output_path


def plot_loaded_capture_pcas(results_path, output_dir):
    captured_results = load_captured_activations(results_path)
    output_dir = Path(output_dir)
    plotted_paths = []

    for task_name, task_results in captured_results.items():
        for model_name, capture_result in task_results.items():
            output_path = output_dir / f"{task_name}_activation_pca_{model_name}.png"
            plotted_path = plot_pca(
                capture_result,
                output_path,
                f"{task_name} {model_name} activation PCA",
            )
            if plotted_path is not None:
                plotted_paths.append(plotted_path)

    return plotted_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load saved captured activations and plot PCA images.",
    )
    parser.add_argument(
        "results_path",
        nargs="?",
        default="results",
        help="Results dir or captured_results.pt path. Defaults to results.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/activation_pca",
        help="Directory for PCA PNGs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_loaded_capture_pcas(args.results_path, args.output_dir)
