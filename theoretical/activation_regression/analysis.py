"""Matrix-level base-to-fine-tuned activation regression analysis."""

from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTIVATION_DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data"


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


def load_activation(path):
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


def run(
    base_path,
    finetuned_path,
    *,
    task_name,
    output_directory,
    top_percent=1.0,
):
    return save_regression_artifacts(
        load_activation(base_path),
        load_activation(finetuned_path),
        Path(output_directory),
        task_name=task_name,
        top_percent=top_percent,
    )


def _paired_matrices(base_states, finetuned_states):
    base_states = np.asarray(base_states, dtype=float)
    finetuned_states = np.asarray(finetuned_states, dtype=float)
    if base_states.ndim != 2 or finetuned_states.ndim != 2:
        raise ValueError("base_states and finetuned_states must both be 2D")
    if base_states.shape != finetuned_states.shape:
        raise ValueError(
            "base_states and finetuned_states must have the same shape: "
            f"{base_states.shape} != {finetuned_states.shape}"
        )
    return base_states, finetuned_states


def matrix_regression(base_states, finetuned_states):
    base_states, finetuned_states = _paired_matrices(base_states, finetuned_states)
    base_means = np.mean(base_states, axis=0)
    finetuned_means = np.mean(finetuned_states, axis=0)
    centered_base = base_states - base_means
    centered_finetuned = finetuned_states - finetuned_means

    covariance = centered_base.T @ centered_finetuned / base_states.shape[0]
    base_variances = np.mean(centered_base**2, axis=0)
    finetuned_variances = np.mean(centered_finetuned**2, axis=0)
    variance_products = base_variances[:, None] * finetuned_variances[None, :]
    r_squared = np.divide(
        covariance**2,
        variance_products,
        out=np.full_like(covariance, np.nan),
        where=variance_products != 0.0,
    )
    alpha = np.divide(
        covariance,
        base_variances[:, None],
        out=np.full_like(covariance, np.nan),
        where=base_variances[:, None] != 0.0,
    )
    beta = finetuned_means[None, :] - alpha * base_means[:, None]
    return {"r_squared": r_squared, "alpha": alpha, "beta": beta}


def identity_mse(base_states, finetuned_states):
    base_states, finetuned_states = _paired_matrices(base_states, finetuned_states)
    return np.mean((finetuned_states - base_states) ** 2, axis=0)


def matrix_ccc(base_states, finetuned_states):
    base_states, finetuned_states = _paired_matrices(base_states, finetuned_states)
    base_means = np.mean(base_states, axis=0)
    finetuned_means = np.mean(finetuned_states, axis=0)
    centered_base = base_states - base_means
    centered_finetuned = finetuned_states - finetuned_means

    covariance = centered_base.T @ centered_finetuned / base_states.shape[0]
    base_variances = np.mean(centered_base**2, axis=0)
    finetuned_variances = np.mean(centered_finetuned**2, axis=0)
    denominator = (
        base_variances[:, None]
        + finetuned_variances[None, :]
        + (base_means[:, None] - finetuned_means[None, :]) ** 2
    )
    return np.divide(
        2.0 * covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator != 0.0,
    )


def _save_heatmap(matrix, title, output_path, cmap, vmin=None, vmax=None):
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    axis.set_title(title)
    axis.set_xlabel("Fine-tuned neuron index")
    axis.set_ylabel("Base neuron index")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_regression_artifacts(
    base_states,
    finetuned_states,
    output_dir,
    *,
    task_name,
    top_percent=1.0,
):
    """Write regression conclusion-support plots to the selected project directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regression = matrix_regression(base_states, finetuned_states)
    paths = {}

    for metric, cmap in (("r_squared", "viridis"), ("alpha", "coolwarm"), ("beta", "coolwarm")):
        path = output_dir / f"{task_name}_{metric}_heatmap.png"
        _save_heatmap(regression[metric], f"{task_name} pairwise {metric}", path, cmap)
        paths[metric] = path

    cutoff = np.nanpercentile(regression["r_squared"], 100.0 - top_percent)
    top_r_squared = np.where(
        regression["r_squared"] >= cutoff,
        regression["r_squared"],
        np.nan,
    )
    paths["top_r_squared"] = output_dir / f"{task_name}_top_r_squared_heatmap.png"
    _save_heatmap(
        top_r_squared,
        f"{task_name} top {top_percent:g}% pairwise R^2",
        paths["top_r_squared"],
        "viridis",
    )

    cross_ccc = matrix_ccc(base_states, finetuned_states)
    base_ccc = matrix_ccc(base_states, base_states)
    paths["ccc"] = output_dir / f"{task_name}_ccc_heatmap.png"
    _save_heatmap(cross_ccc, f"{task_name} pairwise CCC", paths["ccc"], "coolwarm", -1.0, 1.0)
    paths["cross_minus_base_ccc"] = output_dir / f"{task_name}_cross_minus_base_ccc_heatmap.png"
    _save_heatmap(
        cross_ccc - base_ccc,
        f"{task_name} cross-minus-base CCC",
        paths["cross_minus_base_ccc"],
        "coolwarm",
        -2.0,
        2.0,
    )

    mse = identity_mse(base_states, finetuned_states)
    paths["identity_mse"] = output_dir / f"{task_name}_identity_mse_by_neuron.png"
    figure, axis = plt.subplots(figsize=(12, 5))
    indices = np.arange(mse.size)
    axis.plot(indices, mse, color="tab:blue", linewidth=0.8)
    axis.scatter(indices, mse, color="tab:blue", s=5)
    axis.set_title(f"{task_name} same-neuron identity MSE")
    axis.set_xlabel("Neuron index")
    axis.set_ylabel("Identity MSE")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths["identity_mse"], dpi=200)
    plt.close(figure)
    return paths
