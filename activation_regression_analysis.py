import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


DEFAULT_RESULTS_PATHS = ("14th", "8th")


def calc_diagonal_regression(base_states, finetuned_states):
    base_states = np.asarray(base_states)
    finetuned_states = np.asarray(finetuned_states)
    if base_states.ndim != 2 or finetuned_states.ndim != 2:
        raise ValueError("base_states and finetuned_states must both be 2D")
    if base_states.shape != finetuned_states.shape:
        raise ValueError(
            "base_states and finetuned_states must have matching neuron and "
            "example axes"
        )
    if base_states.shape[1] < 2:
        raise ValueError("diagonal regression requires at least two examples")

    working_dtype = np.result_type(
        base_states.dtype,
        finetuned_states.dtype,
        np.float32,
    )
    base_states = base_states.astype(working_dtype, copy=False)
    finetuned_states = finetuned_states.astype(working_dtype, copy=False)

    neuron_count = base_states.shape[0]
    alphas = np.empty(neuron_count, dtype=float)
    intercepts = np.empty(neuron_count, dtype=float)
    r_squared = np.full(neuron_count, np.nan, dtype=float)
    residuals = np.empty_like(finetuned_states)

    regression = LinearRegression()
    for neuron_index in range(neuron_count):
        base_history = base_states[neuron_index, :, None]
        finetuned_history = finetuned_states[neuron_index]
        regression.fit(base_history, finetuned_history)
        prediction = regression.predict(base_history)

        alphas[neuron_index] = regression.coef_[0]
        intercepts[neuron_index] = regression.intercept_
        residuals[neuron_index] = finetuned_history - prediction
        if np.ptp(finetuned_history) > 0.0:
            r_squared[neuron_index] = r2_score(
                finetuned_history,
                prediction,
            )

    return alphas, intercepts, r_squared, residuals


def calc_residual_self_similarity(residuals):
    residuals = np.asarray(residuals)
    if residuals.ndim != 2:
        raise ValueError("residuals must be 2D")
    if residuals.shape[0] == 0:
        return np.empty((0, 0), dtype=float)
    if residuals.shape[1] < 2:
        raise ValueError("residual correlation requires at least two examples")

    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.corrcoef(residuals)
    return np.clip(np.atleast_2d(similarity), -1.0, 1.0)


def plot_r_squared(task_name, r_squared, output_path):
    r_squared = np.asarray(r_squared, dtype=float)
    neuron_indices = np.arange(r_squared.size)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        neuron_indices,
        r_squared,
        color="tab:blue",
        linewidth=0.8,
    )
    axis.scatter(
        neuron_indices,
        r_squared,
        color="tab:blue",
        s=5,
        alpha=0.75,
    )
    axis.set_title(f"{task_name} same-neuron preservation")
    axis.set_xlabel("neuron index")
    axis.set_ylabel(r"$R^2$ from $Y_j = \beta_j + \alpha_j X_j + \varepsilon_j$")
    axis.set_xlim(-0.5, max(r_squared.size - 0.5, 0.5))
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.25)
    undefined_count = int(np.count_nonzero(~np.isfinite(r_squared)))
    if undefined_count:
        axis.text(
            0.01,
            0.02,
            f"{undefined_count} constant fine-tuned neurons have undefined R²",
            transform=axis.transAxes,
            fontsize=8,
        )
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def plot_residual_self_similarity(task_name, similarity, output_path):
    similarity = np.asarray(similarity, dtype=float)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("residual self-similarity must be a square matrix")

    color_map = plt.get_cmap("coolwarm").with_extremes(bad="lightgray")
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(
        np.ma.masked_invalid(similarity),
        cmap=color_map,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    axis.set_title(f"{task_name} residual self-similarity")
    axis.set_xlabel("neuron index")
    axis.set_ylabel("neuron index")
    figure.colorbar(
        image,
        ax=axis,
        label="Pearson correlation between residual histories",
    )
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def capture_states_as_neuron_rows(capture_result):
    return capture_result.states.detach().cpu().numpy().T


def default_output_dir(results_path):
    results_path = Path(results_path)
    if results_path.is_dir() or results_path.suffix == "":
        return results_path / "activation_regression"
    if results_path.parent.name == "captured_activations":
        return results_path.parent.parent / "activation_regression"
    return results_path.parent / "activation_regression"


def run_activation_regression_analysis(results_path, output_dir=None):
    from capture.saving import load_captured_activations

    captured_results = load_captured_activations(results_path)
    output_dir = (
        default_output_dir(results_path)
        if output_dir is None
        else Path(output_dir)
    )
    written_paths = {}

    for task_name, task_results in captured_results.items():
        if "base" not in task_results or "finetuned" not in task_results:
            print(f"{task_name}: skipping; both base and finetuned captures are required")
            continue

        base_states = capture_states_as_neuron_rows(task_results["base"])
        finetuned_states = capture_states_as_neuron_rows(task_results["finetuned"])
        alphas, intercepts, r_squared, residuals = calc_diagonal_regression(
            base_states,
            finetuned_states,
        )
        residual_self_similarity = calc_residual_self_similarity(residuals)

        task_paths = {
            "r_squared": plot_r_squared(
                task_name,
                r_squared,
                output_dir / f"{task_name}_r_squared_by_neuron.png",
            ),
            "residual_self_similarity": plot_residual_self_similarity(
                task_name,
                residual_self_similarity,
                output_dir / f"{task_name}_residual_self_similarity.png",
            ),
        }
        written_paths[task_name] = task_paths

        finite_r_squared = r_squared[np.isfinite(r_squared)]
        mean_r_squared = (
            float(np.mean(finite_r_squared))
            if finite_r_squared.size
            else float("nan")
        )
        print(
            f"{task_name}: neurons={base_states.shape[0]} "
            f"examples={base_states.shape[1]} mean_R2={mean_r_squared:.6f}"
        )
        print(f"{task_name}: saved {task_paths['r_squared']}")
        print(f"{task_name}: saved {task_paths['residual_self_similarity']}")

    return written_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Measure same-neuron preservation with diagonal regression and plot "
            "residual self-similarity."
        )
    )
    parser.add_argument(
        "results_paths",
        nargs="*",
        default=list(DEFAULT_RESULTS_PATHS),
        help=(
            "Capture roots or captured_results.pt files. Defaults to 14th and 8th."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    for results_path in args.results_paths:
        run_activation_regression_analysis(results_path)
