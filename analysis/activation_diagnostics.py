import json
from pathlib import Path

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt


COUNT_PERCENTILES = (50, 75, 90, 95, 99)
DEFAULT_MAX_BINS = 200
DEFAULT_TRACE_NEURONS_PER_PLOT = 1
DEFAULT_TRACE_SUBPLOT_COLUMNS = 32


def save_raw_activation_alpha_diagnostics(
    raw_acts,
    alpha,
    min_acts,
    output_dir,
    *,
    alpha_candidates=None,
    trace_neurons_per_plot=DEFAULT_TRACE_NEURONS_PER_PLOT,
    trace_subplot_columns=DEFAULT_TRACE_SUBPLOT_COLUMNS,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_acts = to_cpu_float_tensor(raw_acts)
    alpha_candidates = tuple(
        alpha_candidates
        if alpha_candidates is not None
        else local_alpha_candidates(alpha)
    )

    sweep = [
        alpha_sweep_record(raw_acts, candidate_alpha, min_acts)
        for candidate_alpha in alpha_candidates
    ]
    summary_path = output_dir / "raw_activation_alpha_sweep.json"
    write_json(summary_path, {"min_acts": int(min_acts), "alpha_sweep": sweep})

    traces_path = output_dir / "raw_activation_traces.png"
    plot_raw_activation_traces(
        raw_acts,
        traces_path,
        neurons_per_plot=trace_neurons_per_plot,
        subplot_columns=trace_subplot_columns,
    )

    correlation_heatmap_path = output_dir / "raw_activation_correlation_heatmap.png"
    plot_raw_correlation_heatmap(raw_acts, correlation_heatmap_path)

    return {
        "alpha_sweep": summary_path,
        "activation_traces": traces_path,
        "correlation_heatmap": correlation_heatmap_path,
    }


def save_binary_activation_count_diagnostics(binary_acts, min_acts, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = activation_counts(binary_acts)

    summary_path = output_dir / "activation_counts_summary.json"
    write_json(summary_path, count_summary(counts, min_acts))

    full_hist_path = output_dir / "activation_counts_hist_full.png"
    plot_activation_count_histogram(
        counts.numpy(),
        min_acts=min_acts,
        output_path=full_hist_path,
        title="Post-binarization activation counts",
    )

    nonzero_hist_path = output_dir / "activation_counts_hist_nonzero.png"
    nonzero_counts = counts[counts > 0].numpy()
    plot_activation_count_histogram(
        nonzero_counts,
        min_acts=min_acts,
        output_path=nonzero_hist_path,
        title="Post-binarization nonzero activation counts",
    )

    return {
        "summary": summary_path,
        "hist_full": full_hist_path,
        "hist_nonzero": nonzero_hist_path,
    }


def local_alpha_candidates(alpha):
    candidates = []
    for offset in (-0.10, -0.05, 0.0, 0.05, 0.10):
        candidate = round(float(alpha) + offset, 6)
        if 0.0 < candidate < 1.0 and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def alpha_sweep_record(raw_acts, alpha, min_acts):
    if raw_acts.shape[0] == 0 or raw_acts.shape[1] == 0:
        counts = torch.zeros(raw_acts.shape[1], dtype=torch.float32)
        record = count_summary(counts, min_acts)
        record["alpha"] = float(alpha)
        return record

    thresholds = per_neuron_thresholds(raw_acts, alpha)
    binary_acts = raw_acts > thresholds
    counts = activation_counts(binary_acts)
    record = count_summary(counts, min_acts)
    record["alpha"] = float(alpha)
    return record


def per_neuron_thresholds(raw_acts, alpha):
    # Thresholding for raw activations
    return torch.quantile(raw_acts, 1 - float(alpha), dim=0)


def activation_counts(binary_acts):
    binary_acts = to_cpu_tensor(binary_acts)
    if binary_acts.ndim != 2:
        raise ValueError(f"expected 2D activation matrix, got shape {tuple(binary_acts.shape)}")
    return binary_acts.sum(dim=0).to(torch.float32)


def count_summary(counts, min_acts):
    counts = counts.to(torch.float32)
    total_neurons = counts.numel()
    zero_count = int((counts == 0).sum().item())
    below_count = int((counts < min_acts).sum().item())
    kept_count = int((counts >= min_acts).sum().item())

    return {
        "min_acts": int(min_acts),
        "zero_activation_count": zero_count,
        "zero_activation_percent": percent(zero_count, total_neurons),
        "below_min_acts_count": below_count,
        "below_min_acts_percent": percent(below_count, total_neurons),
        "kept_count": kept_count,
        "kept_percent": percent(kept_count, total_neurons),
        "activation_count_percentiles": activation_count_percentiles(counts),
    }


def activation_count_percentiles(counts):
    if counts.numel() == 0:
        values = {f"p{percentile}": 0.0 for percentile in COUNT_PERCENTILES}
        values["max"] = 0.0
        return values

    quantiles = torch.tensor(
        [percentile / 100 for percentile in COUNT_PERCENTILES],
        dtype=torch.float32,
    )
    percentile_values = torch.quantile(counts.to(torch.float32), quantiles)
    values = {
        f"p{percentile}": float(value)
        for percentile, value in zip(COUNT_PERCENTILES, percentile_values.tolist(), strict=True)
    }
    values["max"] = float(counts.max().item())
    return values


def percent(count, total):
    if total == 0:
        return 0.0
    return float(count / total * 100)


def plot_activation_count_histogram(counts, min_acts, output_path, title):
    plt.figure(figsize=(10, 6))
    if counts.size == 0:
        plt.text(0.5, 0.5, "No activation counts", ha="center", va="center")
    else:
        plt.hist(counts, bins=adaptive_bins(counts), color="skyblue", edgecolor="black")
        plt.axvline(
            min_acts,
            color="crimson",
            linestyle="--",
            linewidth=2,
            label=f"min_acts = {min_acts}",
        )
        plt.legend()
    plt.title(title)
    plt.xlabel("Total activations")
    plt.ylabel("Number of neurons")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_raw_activation_traces(
    raw_acts,
    output_path,
    *,
    neurons_per_plot=DEFAULT_TRACE_NEURONS_PER_PLOT,
    subplot_columns=DEFAULT_TRACE_SUBPLOT_COLUMNS,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = raw_acts.numpy()
    total_neurons = values.shape[1]
    if neurons_per_plot <= 0:
        raise ValueError("neurons_per_plot must be positive")
    if subplot_columns <= 0:
        raise ValueError("subplot_columns must be positive")

    if total_neurons == 0:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No raw activations", ha="center", va="center")
        plt.title("Raw neuron activations over examples")
        plt.xlabel("Example index")
        plt.ylabel("Raw activation")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path

    x = np.arange(values.shape[0])
    chunks = list(range(0, total_neurons, neurons_per_plot))
    subplot_count = len(chunks)
    columns = min(subplot_columns, subplot_count)
    rows = int(np.ceil(subplot_count / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 5, rows * 3),
        squeeze=False,
        sharex=True,
    )
    axes_flat = axes.flatten()

    for axis, start in zip(axes_flat, chunks, strict=False):
        stop = min(start + neurons_per_plot, total_neurons)
        for neuron_index in range(start, stop):
            axis.plot(x, values[:, neuron_index], linewidth=0.45, alpha=0.45)
        axis.set_title(f"Neurons {start}-{stop - 1}", fontsize=9)
        axis.grid(True, linestyle="--", alpha=0.5)

    for axis in axes_flat[subplot_count:]:
        axis.axis("off")

    fig.suptitle("Raw neuron activations over examples", fontsize=14)
    fig.supxlabel("Example index")
    fig.supylabel("Raw activation")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_raw_correlation_heatmap(raw_acts, output_path):
    correlation = raw_activation_correlation_matrix(raw_acts)
    if correlation.size == 0:
        plot_empty_heatmap(output_path, "Raw activation correlation heatmap")
        return

    plt.figure(figsize=(10, 8))
    image = plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )
    plt.colorbar(image, label="Pearson correlation")
    plt.title("Raw activation correlation heatmap")
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron index")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_empty_heatmap(output_path, title):
    plt.figure(figsize=(10, 8))
    plt.text(0.5, 0.5, "No neuron columns", ha="center", va="center")
    plt.title(title)
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron index")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def raw_activation_correlation_matrix(raw_acts):
    values = raw_acts.numpy()
    if values.shape[1] == 0:
        return np.empty((0, 0))
    if values.shape[0] <= 1:
        return np.eye(values.shape[1])

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.atleast_2d(np.corrcoef(values, rowvar=False))
    constant_columns = values.std(axis=0) == 0
    correlation[constant_columns, :] = 0.0
    correlation[:, constant_columns] = 0.0
    np.fill_diagonal(correlation, 1.0)
    return correlation


def adaptive_bins(values, max_bins=DEFAULT_MAX_BINS):
    values = np.asarray(values)
    if values.size <= 1:
        return 1
    unique_count = np.unique(values).size
    return max(1, min(int(unique_count), int(np.sqrt(values.size) * 4), max_bins))


def to_cpu_float_tensor(value):
    value = to_cpu_tensor(value)
    if value.ndim != 2:
        raise ValueError(f"expected 2D activation matrix, got shape {tuple(value.shape)}")
    return value.to(torch.float32)


def to_cpu_tensor(value):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().cpu()


def write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
