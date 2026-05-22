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
HEATMAP_DPI = 300
DIAGNOSTICS_REPORT_FILENAME = "activation_diagnostics_report.txt"


def save_raw_activation_alpha_diagnostics(
    raw_acts,
    alpha,
    min_acts,
    output_dir,
    *,
    raw_acts_base=None,
    alpha_candidates=None,
    trace_neurons_per_plot=DEFAULT_TRACE_NEURONS_PER_PLOT,
    trace_subplot_columns=DEFAULT_TRACE_SUBPLOT_COLUMNS,
    top_k=10,
    report_path=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if raw_acts_base is None:
        raise ValueError("raw_acts_base is required for base/finetuned diagnostics")
    raw_acts_finetuned = to_cpu_float_tensor(raw_acts)
    raw_acts_base = to_cpu_float_tensor(raw_acts_base)
    if raw_acts_finetuned.shape != raw_acts_base.shape:
        raise ValueError(
            "base and finetuned activation matrices must have the same shape: "
            f"base={tuple(raw_acts_base.shape)}, "
            f"finetuned={tuple(raw_acts_finetuned.shape)}"
        )
    alpha_candidates = tuple(
        alpha_candidates
        if alpha_candidates is not None
        else local_alpha_candidates(alpha)
    )

    sweep = [
        alpha_sweep_record(raw_acts_finetuned, candidate_alpha, min_acts)
        for candidate_alpha in alpha_candidates
    ]
    report_path = Path(report_path) if report_path is not None else output_dir / DIAGNOSTICS_REPORT_FILENAME

    traces_path = output_dir / "raw_activation_traces.png"
    plot_raw_activation_traces(
        raw_acts_finetuned,
        traces_path,
        neurons_per_plot=trace_neurons_per_plot,
        subplot_columns=trace_subplot_columns,
    )

    correlation_base = raw_activation_correlation_matrix(raw_acts_base)
    correlation_finetuned = raw_activation_correlation_matrix(raw_acts_finetuned)
    correlation_difference = correlation_finetuned - correlation_base
    cosine_base = raw_activation_cosine_similarity_matrix(raw_acts_base)
    cosine_finetuned = raw_activation_cosine_similarity_matrix(raw_acts_finetuned)
    cosine_difference = cosine_finetuned - cosine_base

    correlation_base_path = output_dir / "raw_activation_correlation_heatmap_base.png"
    correlation_finetuned_path = output_dir / "raw_activation_correlation_heatmap_finetuned.png"
    correlation_difference_path = output_dir / "raw_activation_correlation_heatmap_difference.png"
    cosine_base_path = output_dir / "raw_activation_cosine_similarity_heatmap_base.png"
    cosine_finetuned_path = output_dir / "raw_activation_cosine_similarity_heatmap_finetuned.png"
    cosine_difference_path = output_dir / "raw_activation_cosine_similarity_heatmap_difference.png"

    plot_similarity_heatmap(
        correlation_base,
        correlation_base_path,
        "Base raw activation Pearson correlation heatmap",
        "Pearson correlation",
    )
    plot_similarity_heatmap(
        correlation_finetuned,
        correlation_finetuned_path,
        "Finetuned raw activation Pearson correlation heatmap",
        "Pearson correlation",
    )
    plot_difference_heatmap(
        correlation_difference,
        correlation_difference_path,
        "Finetuned minus base raw activation Pearson correlation heatmap",
        "Pearson correlation difference",
    )
    plot_similarity_heatmap(
        cosine_base,
        cosine_base_path,
        "Base raw activation cosine similarity heatmap",
        "Cosine similarity",
    )
    plot_similarity_heatmap(
        cosine_finetuned,
        cosine_finetuned_path,
        "Finetuned raw activation cosine similarity heatmap",
        "Cosine similarity",
    )
    plot_difference_heatmap(
        cosine_difference,
        cosine_difference_path,
        "Finetuned minus base raw activation cosine similarity heatmap",
        "Cosine similarity difference",
    )

    write_raw_diagnostics_report(
        report_path,
        min_acts,
        sweep,
        (
            ("Pearson correlation base", correlation_base, False),
            ("Pearson correlation finetuned", correlation_finetuned, False),
            ("Pearson correlation difference", correlation_difference, True),
            ("Cosine similarity base", cosine_base, False),
            ("Cosine similarity finetuned", cosine_finetuned, False),
            ("Cosine similarity difference", cosine_difference, True),
        ),
        top_k,
    )

    return {
        "diagnostics_report": report_path,
        "activation_traces": traces_path,
        "correlation_heatmap_base": correlation_base_path,
        "correlation_heatmap_finetuned": correlation_finetuned_path,
        "correlation_heatmap_difference": correlation_difference_path,
        "cosine_similarity_heatmap_base": cosine_base_path,
        "cosine_similarity_heatmap_finetuned": cosine_finetuned_path,
        "cosine_similarity_heatmap_difference": cosine_difference_path,
    }


def save_binary_activation_count_diagnostics(binary_acts, min_acts, output_dir, *, report_path=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = activation_counts(binary_acts)
    report_path = Path(report_path) if report_path is not None else output_dir / DIAGNOSTICS_REPORT_FILENAME
    summary = count_summary(counts, min_acts)

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

    jaccard_heatmap_path = output_dir / "binarized_activation_jaccard_heatmap.png"
    plot_jaccard_similarity_heatmap(binary_acts, jaccard_heatmap_path)
    append_binary_diagnostics_report(report_path, summary)

    return {
        "diagnostics_report": report_path,
        "hist_full": full_hist_path,
        "hist_nonzero": nonzero_hist_path,
        "jaccard_heatmap": jaccard_heatmap_path,
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


def plot_similarity_heatmap(matrix, output_path, title, colorbar_label):
    if matrix.size == 0:
        plot_empty_heatmap(output_path, title)
        return

    plt.figure(figsize=(10, 8))
    image = plt.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(image, label=colorbar_label)
    plt.title(title)
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=HEATMAP_DPI)
    plt.close()


def plot_difference_heatmap(matrix, output_path, title, colorbar_label):
    if matrix.size == 0:
        plot_empty_heatmap(output_path, title)
        return

    max_abs_difference = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    color_limit = max_abs_difference if max_abs_difference > 0 else 1.0

    plt.figure(figsize=(10, 8))
    image = plt.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(image, label=colorbar_label)
    plt.title(title)
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=HEATMAP_DPI)
    plt.close()


def plot_jaccard_similarity_heatmap(binary_acts, output_path):
    jaccard = jaccard_similarity_matrix(binary_acts)
    if jaccard.size == 0:
        plot_empty_heatmap(output_path, "Binarized activation Jaccard similarity heatmap")
        return

    plt.figure(figsize=(10, 8))
    image = plt.imshow(
        jaccard,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(image, label="Jaccard similarity / IoU")
    plt.title("Binarized activation Jaccard similarity heatmap")
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=HEATMAP_DPI)
    plt.close()


def plot_empty_heatmap(output_path, title):
    plt.figure(figsize=(10, 8))
    plt.text(0.5, 0.5, "No neuron columns", ha="center", va="center")
    plt.title(title)
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=HEATMAP_DPI)
    plt.close()


def raw_activation_correlation_matrix(raw_acts):
    values = to_cpu_float_tensor(raw_acts).numpy()
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


def raw_activation_cosine_similarity_matrix(raw_acts):
    values = to_cpu_float_tensor(raw_acts).numpy()
    if values.shape[1] == 0:
        return np.empty((0, 0))

    dot_products = values.T @ values
    norms = np.linalg.norm(values, axis=0)
    norm_products = norms[:, None] * norms[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.divide(
            dot_products,
            norm_products,
            out=np.zeros_like(dot_products, dtype=np.float32),
            where=norm_products != 0,
        )
    np.fill_diagonal(cosine, 1.0)
    return cosine


def upper_triangle_pairs(matrix):
    if matrix.shape[0] < 2:
        return []
    row_indices, column_indices = np.triu_indices(matrix.shape[0], k=1)
    return [
        (int(row), int(column), float(matrix[row, column]))
        for row, column in zip(row_indices, column_indices, strict=True)
    ]


def sorted_pairs(matrix, *, reverse):
    pairs = upper_triangle_pairs(matrix)
    if reverse:
        return sorted(pairs, key=lambda pair: (-pair[2], pair[0], pair[1]))
    return sorted(pairs, key=lambda pair: (pair[2], pair[0], pair[1]))


def pair_lines(matrix, *, top_k, is_difference):
    pairs = upper_triangle_pairs(matrix)
    if not pairs:
        return ["No neuron pairs available."]

    high_label = "Top increased pairs" if is_difference else "Top positive pairs"
    low_label = "Top decreased pairs" if is_difference else "Top negative pairs"
    lines = [high_label]
    lines.extend(format_pairs(sorted_pairs(matrix, reverse=True)[:top_k]))
    lines.append(low_label)
    lines.extend(format_pairs(sorted_pairs(matrix, reverse=False)[:top_k]))
    return lines


def format_pairs(pairs):
    return [
        f"  neuron {first} <-> neuron {second}: {score:.6f}"
        for first, second, score in pairs
    ]


def write_raw_diagnostics_report(report_path, min_acts, sweep, matrix_sections, top_k):
    lines = [
        "ACTIVATION DIAGNOSTICS REPORT",
        "",
        "RAW ACTIVATION ALPHA SWEEP",
        f"min_acts: {int(min_acts)}",
    ]
    for record in sweep:
        lines.extend(format_count_summary(record, prefix=f"alpha={record['alpha']:.6g}"))

    lines.extend(["", "TOP NEURON PAIRS"])
    for name, matrix, is_difference in matrix_sections:
        lines.extend(["", name])
        lines.extend(pair_lines(matrix, top_k=top_k, is_difference=is_difference))

    write_text_report(report_path, lines, mode="w")


def append_binary_diagnostics_report(report_path, summary):
    lines = [
        "",
        "POST-BINARIZATION ACTIVATION COUNT SUMMARY",
    ]
    lines.extend(format_count_summary(summary))
    write_text_report(report_path, lines, mode="a")


def format_count_summary(summary, prefix=None):
    lines = []
    if prefix is not None:
        lines.append(prefix)
    lines.extend(
        [
            f"  zero_activation_count: {summary['zero_activation_count']}",
            f"  zero_activation_percent: {summary['zero_activation_percent']:.6f}",
            f"  below_min_acts_count: {summary['below_min_acts_count']}",
            f"  below_min_acts_percent: {summary['below_min_acts_percent']:.6f}",
            f"  kept_count: {summary['kept_count']}",
            f"  kept_percent: {summary['kept_percent']:.6f}",
            "  activation_count_percentiles:",
        ]
    )
    for name, value in summary["activation_count_percentiles"].items():
        lines.append(f"    {name}: {value:.6f}")
    return lines


def write_text_report(path, lines, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as file:
        file.write("\n".join(lines))
        file.write("\n")


def jaccard_similarity_matrix(binary_acts):
    binary_acts = to_cpu_tensor(binary_acts)
    if binary_acts.ndim != 2:
        raise ValueError(f"expected 2D activation matrix, got shape {tuple(binary_acts.shape)}")
    binary_acts = binary_acts.numpy()

    num_examples, num_neurons = binary_acts.shape
    if num_neurons == 0:
        return np.empty((0, 0))
    if num_examples == 0:
        return np.zeros((num_neurons, num_neurons))

    X = binary_acts.astype(np.float32)
    intersection = X.T @ X
    sums = X.sum(axis=0)
    union = sums[:, None] + sums[None, :] - intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)

    np.fill_diagonal(jaccard, 1.0)
    return jaccard


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
