import argparse
from pathlib import Path

import matplotlib
import numpy as np

from capture.saving import load_captured_activations


matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOP_PERCENT = 1.0


parser = argparse.ArgumentParser(
    description="Calculate matrix-level activation regressions for one layer."
)
parser.add_argument(
    "--layer",
    default="14th",
    help="Capture results directory or captured_results.pt file. Defaults to 14th.",
)


def matrix_regression(base_states, finetuned_states):
    base_states = base_states.astype(float, copy=False)
    finetuned_states = finetuned_states.astype(float, copy=False)

    base_means = np.mean(base_states, axis=0)
    finetuned_means = np.mean(finetuned_states, axis=0)
    # X_c[e, i] = X[e, i] - mu_X[i]
    centered_base = base_states - base_means
    # Y_c[e, j] = Y[e, j] - mu_Y[j]
    centered_finetuned = finetuned_states - finetuned_means

    example_count = base_states.shape[0]
    # C[i, j] = (1 / N) sum_e X_c[e, i] Y_c[e, j]
    covariance = centered_base.T @ centered_finetuned / example_count
    base_variances = np.mean(centered_base**2, axis=0)
    finetuned_variances = np.mean(centered_finetuned**2, axis=0)

    # V[i, j] = Var(X[:, i]) Var(Y[:, j])
    variance_products = base_variances[:, None] * finetuned_variances[None, :]
    # R^2[i, j] = C[i, j]^2 / V[i, j]
    r_squared = np.divide(
        covariance**2,
        variance_products,
        out=np.full_like(covariance, np.nan),
        where=variance_products != 0.0,
    )

    # alpha[i, j] = C[i, j] / Var(X[:, i])
    alpha = np.divide(
        covariance,
        base_variances[:, None],
        out=np.full_like(covariance, np.nan),
        where=base_variances[:, None] != 0.0,
    )
    # beta[i, j] = mu_Y[j] - alpha[i, j] mu_X[i]
    beta = finetuned_means[None, :] - alpha * base_means[:, None]

    return {
        "r_squared": r_squared,
        "alpha": alpha,
        "beta": beta,
    }


def identity_mse(base_states, finetuned_states):
    # mse[i] = (1 / N) sum_e (Y[e, i] - X[e, i])^2
    return np.mean((finetuned_states - base_states) ** 2, axis=0)


def matrix_ccc(base_states, finetuned_states):
    base_states = base_states.astype(float, copy=False)
    finetuned_states = finetuned_states.astype(float, copy=False)

    base_means = np.mean(base_states, axis=0)
    finetuned_means = np.mean(finetuned_states, axis=0)
    # X_c[e, i] = X[e, i] - mu_X[i]
    centered_base = base_states - base_means
    # Y_c[e, j] = Y[e, j] - mu_Y[j]
    centered_finetuned = finetuned_states - finetuned_means

    example_count = base_states.shape[0]
    # C[i, j] = (1 / N) sum_e X_c[e, i] Y_c[e, j]
    covariance = centered_base.T @ centered_finetuned / example_count
    base_variances = np.mean(centered_base**2, axis=0)
    finetuned_variances = np.mean(centered_finetuned**2, axis=0)

    # D[i, j] = Var(X[:, i]) + Var(Y[:, j]) + (mu_X[i] - mu_Y[j])^2
    denominator = (
        base_variances[:, None]
        + finetuned_variances[None, :]
        + (base_means[:, None] - finetuned_means[None, :]) ** 2
    )
    # CCC[i, j] = 2 C[i, j] / D[i, j]
    return np.divide(
        2.0 * covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator != 0.0,
    )


def plot_heatmap(
    matrix,
    title,
    x_label,
    y_label,
    colorbar_label,
    output_path,
    cmap,
    vmin=None,
    vmax=None,
):
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
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_identity_mse(mse_by_neuron, title, output_path):
    neuron_indices = np.arange(mse_by_neuron.size)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(neuron_indices, mse_by_neuron, color="tab:blue", linewidth=0.8)
    axis.scatter(neuron_indices, mse_by_neuron, color="tab:blue", s=5)
    axis.set_title(title)
    axis.set_xlabel("Neuron index")
    axis.set_ylabel("Identity MSE")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    args = parser.parse_args()
    captured_results = load_captured_activations(args.layer)
    output_dir = Path(args.layer) / "activation_regression"
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_name, task_results in captured_results.items():
        base_states = task_results["base"].states.detach().cpu().numpy()
        finetuned_states = task_results["finetuned"].states.detach().cpu().numpy()
        result = matrix_regression(
            base_states,
            finetuned_states,
        )

        plot_heatmap(
            result["r_squared"],
            f"{task_name} pairwise R^2",
            "Fine-tuned neuron index",
            "Base neuron index",
            "R^2",
            output_dir / f"{task_name}_r_squared_heatmap.png",
            "viridis",
        )
        plot_heatmap(
            result["alpha"],
            f"{task_name} pairwise alpha",
            "Fine-tuned neuron index",
            "Base neuron index",
            "alpha",
            output_dir / f"{task_name}_alpha_heatmap.png",
            "coolwarm",
        )
        plot_heatmap(
            result["beta"],
            f"{task_name} pairwise beta",
            "Fine-tuned neuron index",
            "Base neuron index",
            "beta",
            output_dir / f"{task_name}_beta_heatmap.png",
            "coolwarm",
        )

        cutoff = np.nanpercentile(
            result["r_squared"],
            100.0 - TOP_PERCENT,
        )
        top_r_squared = np.where(
            result["r_squared"] >= cutoff,
            result["r_squared"],
            np.nan,
        )
        plot_heatmap(
            top_r_squared,
            f"{task_name} top {TOP_PERCENT:g}% pairwise R^2",
            "Fine-tuned neuron index",
            "Base neuron index",
            "R^2",
            output_dir / f"{task_name}_top_r_squared_heatmap.png",
            "viridis",
        )

        mse_by_neuron = identity_mse(base_states, finetuned_states)
        plot_identity_mse(
            mse_by_neuron,
            f"{task_name} same-neuron identity MSE",
            output_dir / f"{task_name}_identity_mse_by_neuron.png",
        )

        cross_ccc = matrix_ccc(base_states, finetuned_states)
        base_ccc = matrix_ccc(base_states, base_states)
        cross_minus_base_ccc = cross_ccc - base_ccc
        plot_heatmap(
            cross_ccc,
            f"{task_name} pairwise CCC",
            "Fine-tuned neuron index",
            "Base neuron index",
            "CCC",
            output_dir / f"{task_name}_ccc_heatmap.png",
            "coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        plot_heatmap(
            cross_minus_base_ccc,
            f"{task_name} cross-minus-base CCC",
            "Fine-tuned neuron index",
            "Base neuron index",
            "CCC change",
            output_dir / f"{task_name}_cross_minus_base_ccc_heatmap.png",
            "coolwarm",
            vmin=-2.0,
            vmax=2.0,
        )
