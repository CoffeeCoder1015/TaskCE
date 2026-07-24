"""Measure whether thresholded activations support downstream formula analysis."""

import torch

from theoretical.activations.source import load_activation


COUNT_PERCENTILES = (50, 75, 90, 95, 99)


def run(
    activation_path,
    *,
    task_name,
    alpha,
    min_acts,
    alpha_candidates=None,
):
    """Load one capture and return its complete threshold-coverage result."""
    return analyze_threshold_coverage(
        load_activation(activation_path),
        task_name=task_name,
        alpha=alpha,
        min_acts=min_acts,
        alpha_candidates=alpha_candidates,
    )


def analyze_threshold_coverage(
    raw_states,
    *,
    task_name,
    alpha,
    min_acts,
    alpha_candidates=None,
):
    """Construct selected-threshold counts and the surrounding alpha sweep."""
    raw_states = _activation_tensor(raw_states)
    alpha = _alpha(alpha)
    min_acts = int(min_acts)
    if min_acts < 0:
        raise ValueError(f"min_acts must be nonnegative, got {min_acts}")

    selected_counts = _counts_for_alpha(raw_states, alpha)
    sweep = alpha_sweep(
        raw_states,
        task_name=task_name,
        alpha=alpha,
        min_acts=min_acts,
        alpha_candidates=alpha_candidates,
    )
    integer_counts = [int(count) for count in selected_counts.tolist()]

    return {
        "selected": {
            "alpha": alpha,
            "min_acts": min_acts,
            "counts": integer_counts,
            "nonzero_counts": [count for count in integer_counts if count > 0],
            "summary": count_summary(selected_counts, min_acts),
        },
        "sweep": sweep,
    }


def alpha_sweep(
    raw_states,
    *,
    task_name,
    alpha,
    min_acts,
    alpha_candidates=None,
):
    """Measure coverage at the selected alpha and nearby candidates."""
    raw_states = _activation_tensor(raw_states)
    num_examples, num_neurons = map(int, raw_states.shape)
    candidates = _alpha_candidates(
        alpha,
        min_acts,
        num_examples,
        alpha_candidates,
    )
    records = []

    for candidate_alpha in candidates:
        counts = _counts_for_alpha(raw_states, candidate_alpha)
        summary = count_summary(counts, min_acts)
        expected_activation_count = candidate_alpha * num_examples
        records.append(
            {
                "alpha": candidate_alpha,
                "expected_activation_count": expected_activation_count,
                "expected_minus_min_acts": expected_activation_count - min_acts,
                **summary,
            }
        )

    return {
        "task_name": task_name,
        "selected_alpha": float(alpha),
        "min_acts": int(min_acts),
        "num_examples": num_examples,
        "num_neurons": num_neurons,
        "records": records,
    }


def count_summary(counts, min_acts):
    """Return JSON-ready evidence-coverage statistics."""
    counts = counts.to(torch.int64)
    total_neurons = counts.numel()
    zero_count = int((counts == 0).sum().item())
    below_count = int((counts < min_acts).sum().item())
    kept_count = int((counts >= min_acts).sum().item())

    return {
        "zero_activation_count": zero_count,
        "zero_activation_percent": _percent(zero_count, total_neurons),
        "below_min_acts_count": below_count,
        "below_min_acts_percent": _percent(below_count, total_neurons),
        "kept_count": kept_count,
        "kept_percent": _percent(kept_count, total_neurons),
        "activation_count_percentiles": _count_percentiles(counts),
    }


def _counts_for_alpha(raw_states, alpha):
    if raw_states.shape[0] == 0 or raw_states.shape[1] == 0:
        return torch.zeros(raw_states.shape[1], dtype=torch.int64)
    thresholds = torch.quantile(raw_states, 1.0 - alpha, dim=0)
    return (raw_states > thresholds).sum(dim=0).to(torch.int64)


def _alpha_candidates(alpha, min_acts, num_examples, candidates):
    if candidates is None:
        candidate_values = [alpha]
        if num_examples > 0:
            min_acts_alpha = min_acts / num_examples
            candidate_values.extend(
                min_acts_alpha * multiplier
                for multiplier in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
            )
        candidate_values.extend(
            alpha + offset
            for offset in (-0.05, -0.025, 0.025, 0.05)
        )
    else:
        candidate_values = candidates

    return sorted(
        {
            round(_alpha(candidate), 6)
            for candidate in candidate_values
            if 0.0 < float(candidate) < 1.0
        }
    )


def _count_percentiles(counts):
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
        for percentile, value in zip(
            COUNT_PERCENTILES,
            percentile_values.tolist(),
            strict=True,
        )
    }
    values["max"] = float(counts.max().item())
    return values


def _activation_tensor(value):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.detach().cpu()
    if value.ndim != 2:
        raise ValueError(
            f"expected 2D activation matrix, got shape {tuple(value.shape)}"
        )
    return value.to(torch.float32)


def _alpha(value):
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {value}")
    return value


def _percent(count, total):
    if total == 0:
        return 0.0
    return float(count / total * 100.0)
