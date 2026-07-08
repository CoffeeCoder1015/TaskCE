import json
from pathlib import Path

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt


COUNT_PERCENTILES = (50, 75, 90, 95, 99)
DEFAULT_MAX_BINS = 200
HEATMAP_DPI = 300


def raw_activation_analysis(
    captured_results,
    task_name,
    *,
    left_key="base",
    right_key="finetuned",
    output_dir=None,
):
    left_states = captured_results[task_name][left_key].states
    right_states = captured_results[task_name][right_key].states
    left_states = to_cpu_float_tensor(left_states)
    right_states = to_cpu_float_tensor(right_states)
    validate_same_activation_shape(left_states, right_states, left_key, right_key)

    pearson_left = raw_activation_correlation_matrix(left_states)
    pearson_right = raw_activation_correlation_matrix(right_states)
    cosine_left = raw_activation_cosine_similarity_matrix(left_states)
    cosine_right = raw_activation_cosine_similarity_matrix(right_states)

    result = {
        "left_label": left_key,
        "right_label": right_key,
        "pearson": {
            "left": pearson_left,
            "right": pearson_right,
            "difference": pearson_right - pearson_left,
        },
        "cosine": {
            "left": cosine_left,
            "right": cosine_right,
            "difference": cosine_right - cosine_left,
        },
    }

    if output_dir is not None:
        result["paths"] = save_raw_activation_analysis(result, output_dir)

    return result


def validate_same_activation_shape(left_states, right_states, left_label, right_label):
    if left_states.shape != right_states.shape:
        raise ValueError(
            f"{left_label} and {right_label} activation matrices must have the same shape: "
            f"{left_label}={tuple(left_states.shape)}, "
            f"{right_label}={tuple(right_states.shape)}"
        )


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


def run_alpha_sweep(
    captured_results,
    task_name,
    *,
    alpha,
    min_acts,
    model_key="finetuned",
    alpha_candidates=None,
    output_dir=None,
):
    raw_states = to_cpu_float_tensor(captured_results[task_name][model_key].states)
    num_examples = int(raw_states.shape[0])
    num_neurons = int(raw_states.shape[1])
    if alpha_candidates is None:
        candidate_values = [alpha]
        if num_examples > 0:
            min_acts_alpha = min_acts / num_examples
            candidate_values.extend(
                min_acts_alpha * multiplier
                for multiplier in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
            )
        candidate_values.extend(float(alpha) + offset for offset in (-0.05, -0.025, 0.025, 0.05))
    else:
        candidate_values = alpha_candidates

    candidates = sorted(
        {
            round(float(candidate), 6)
            for candidate in candidate_values
            if 0.0 < float(candidate) < 1.0
        }
    )

    records = []
    for candidate_alpha in candidates:
        if raw_states.shape[0] == 0 or raw_states.shape[1] == 0:
            counts = torch.zeros(raw_states.shape[1], dtype=torch.float32)
        else:
            thresholds = torch.quantile(raw_states, 1 - float(candidate_alpha), dim=0)
            binary_acts = raw_states > thresholds
            counts = activation_counts(binary_acts)

        counts = counts.to(torch.float32)
        total_neurons = counts.numel()
        zero_count = int((counts == 0).sum().item())
        below_count = int((counts < min_acts).sum().item())
        kept_count = int((counts >= min_acts).sum().item())
        expected_activation_count = float(candidate_alpha * num_examples)
        records.append(
            {
                "alpha": float(candidate_alpha),
                "min_acts": int(min_acts),
                "expected_activation_count": expected_activation_count,
                "expected_minus_min_acts": expected_activation_count - int(min_acts),
                "zero_activation_count": zero_count,
                "zero_activation_percent": percent(zero_count, total_neurons),
                "below_min_acts_count": below_count,
                "below_min_acts_percent": percent(below_count, total_neurons),
                "kept_count": kept_count,
                "kept_percent": percent(kept_count, total_neurons),
                "activation_count_percentiles": activation_count_percentiles(counts),
            }
        )

    result = {
        "task_name": task_name,
        "alpha": float(alpha),
        "min_acts": int(min_acts),
        "model_key": model_key,
        "num_examples": num_examples,
        "num_neurons": num_neurons,
        "records": records,
    }

    if output_dir is not None:
        result["paths"] = save_alpha_sweep_results(result, output_dir)

    return result


def binary_activation_analysis(
    binary_acts,
    *,
    min_acts,
    output_dir=None,
):
    counts = activation_counts(binary_acts)
    result = {
        "counts": counts,
        "nonzero_counts": counts[counts > 0],
        "summary": count_summary(counts, min_acts),
    }

    if output_dir is not None:
        result["paths"] = save_binary_activation_analysis(result, output_dir)

    return result


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


def to_cpu_float_tensor(value):
    value = to_cpu_tensor(value)
    if value.ndim != 2:
        raise ValueError(f"expected 2D activation matrix, got shape {tuple(value.shape)}")
    return value.to(torch.float32)


def to_cpu_tensor(value):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().cpu()


def save_alpha_sweep_results(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_sweep_path = output_dir / "alpha_sweep.json"
    alpha_sweep_report_path = output_dir / "alpha_sweep.md"
    serializable_result = {
        "task_name": result["task_name"],
        "alpha": result["alpha"],
        "min_acts": result["min_acts"],
        "model_key": result["model_key"],
        "num_examples": result["num_examples"],
        "num_neurons": result["num_neurons"],
        "records": result["records"],
    }
    with alpha_sweep_path.open("w", encoding="utf-8") as file:
        json.dump(serializable_result, file, indent=2)
        file.write("\n")

    lines = [
        "# Alpha Sweep",
        "",
        f"task: {result['task_name']}",
        f"model: {result['model_key']}",
        f"selected alpha: {result['alpha']:.6g}",
        f"min_acts: {result['min_acts']}",
        f"examples: {result['num_examples']}",
        f"neurons: {result['num_neurons']}",
        "",
        "| alpha | expected acts/neuron | expected - min_acts | kept | kept % | below min | below min % | zero | zero % | p50 | p75 | p90 | p95 | p99 | max |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    lines.extend(alpha_sweep_record_rows(result["records"]))
    with alpha_sweep_report_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))
        file.write("\n")

    return {
        "alpha_sweep": alpha_sweep_path,
        "alpha_sweep_report": alpha_sweep_report_path,
    }


def alpha_sweep_record_rows(records):
    rows = []
    for record in records:
        percentiles = record["activation_count_percentiles"]
        values = [
            f"{record['alpha']:.6g}",
            f"{record['expected_activation_count']:.2f}",
            f"{record['expected_minus_min_acts']:.2f}",
            str(record["kept_count"]),
            f"{record['kept_percent']:.6f}",
            str(record["below_min_acts_count"]),
            f"{record['below_min_acts_percent']:.6f}",
            str(record["zero_activation_count"]),
            f"{record['zero_activation_percent']:.6f}",
            f"{percentiles['p50']:.6f}",
            f"{percentiles['p75']:.6f}",
            f"{percentiles['p90']:.6f}",
            f"{percentiles['p95']:.6f}",
            f"{percentiles['p99']:.6f}",
            f"{percentiles['max']:.6f}",
        ]
        rows.append("| " + " | ".join(values) + " |")
    return rows


def save_raw_activation_analysis(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left_label = result["left_label"].title()
    right_label = result["right_label"].title()

    metrics = [
        ("correlation", "pearson", "Pearson correlation"),
        ("cosine_similarity", "cosine", "cosine similarity"),
    ]
    comparisons = [
        ("base", "left", left_label, "", (-1, 1)),
        ("finetuned", "right", right_label, "", (-1, 1)),
        ("difference", "difference", f"{right_label} minus {left_label.lower()}", " difference", "symmetric"),
    ]

    paths = {}
    for filename_metric, result_metric, metric_label in metrics:
        colorbar_label = metric_label[:1].upper() + metric_label[1:]
        for suffix, result_side, title_label, colorbar_suffix, color_range in comparisons:
            key = f"{filename_metric}_heatmap_{suffix}"
            path = output_dir / f"raw_activation_{filename_metric}_heatmap_{suffix}.png"
            title = f"{title_label} raw activation {metric_label} heatmap"
            plot_heatmap(result[result_metric][result_side], path, title, colorbar_label + colorbar_suffix, color_range=color_range)
            paths[key] = path

    return paths


def save_binary_activation_analysis(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_hist_path = output_dir / "activation_counts_hist_full.png"
    nonzero_hist_path = output_dir / "activation_counts_hist_nonzero.png"

    plot_activation_count_histogram(
        result["counts"].numpy(),
        min_acts=result["summary"]["min_acts"],
        output_path=full_hist_path,
        title="Post-binarization activation counts",
    )
    plot_activation_count_histogram(
        result["nonzero_counts"].numpy(),
        min_acts=result["summary"]["min_acts"],
        output_path=nonzero_hist_path,
        title="Post-binarization nonzero activation counts",
    )

    paths = {
        "hist_full": full_hist_path,
        "hist_nonzero": nonzero_hist_path,
    }
    return paths


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


def plot_heatmap(matrix, output_path, title, colorbar_label, *, cmap="coolwarm", color_range=(-1, 1)):
    plt.figure(figsize=(10, 8))

    if matrix.size == 0:
        plt.text(0.5, 0.5, "No neuron columns", ha="center", va="center")
    else:
        if color_range == "symmetric":
            max_abs_difference = float(np.max(np.abs(matrix)))
            color_limit = max_abs_difference if max_abs_difference > 0 else 1.0
            vmin, vmax = -color_limit, color_limit
        else:
            vmin, vmax = color_range

        image = plt.imshow(
            matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
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


def adaptive_bins(values, max_bins=DEFAULT_MAX_BINS):
    values = np.asarray(values)
    if values.size <= 1:
        return 1
    unique_count = np.unique(values).size
    return max(1, min(int(unique_count), int(np.sqrt(values.size) * 4), max_bins))
