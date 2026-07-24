"""Measure what happened to neuron activation behavior after fine-tuning."""

import numpy as np

from theoretical.activations.source import load_activation


def run(base_path, finetuned_path):
    """Load paired captures and return their complete neuron-fate measurements."""
    return analyze_neuron_fate(
        load_activation(base_path),
        load_activation(finetuned_path),
    )


def analyze_neuron_fate(base_states, finetuned_states):
    """Construct the complete pairwise measurement structure."""
    base_states, finetuned_states = _paired_matrices(
        base_states,
        finetuned_states,
        require_same_neurons=True,
    )
    affine = matrix_regression(base_states, finetuned_states)
    cross_ccc = matrix_ccc(base_states, finetuned_states)
    base_ccc = matrix_ccc(base_states, base_states)

    return {
        "affine": affine,
        "literal": {
            "ccc": cross_ccc,
            "identity_mse": identity_mse(base_states, finetuned_states),
        },
        "relative_to_base": {
            "base_ccc": base_ccc,
            "cross_minus_base_ccc": cross_ccc - base_ccc,
        },
    }


def matrix_regression(base_states, finetuned_states):
    """Fit every directed base-to-fine affine relationship."""
    base_states, finetuned_states = _paired_matrices(
        base_states,
        finetuned_states,
    )
    statistics = _pairwise_statistics(base_states, finetuned_states)
    covariance = statistics["covariance"]
    base_variances = statistics["base_variances"]
    finetuned_variances = statistics["finetuned_variances"]

    variance_products = np.multiply.outer(base_variances, finetuned_variances)
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
    beta = (
        statistics["finetuned_means"][None, :]
        - alpha * statistics["base_means"][:, None]
    )
    return {
        "r_squared": r_squared,
        "alpha": alpha,
        "beta": beta,
    }


def identity_mse(base_states, finetuned_states):
    """Return pairwise MSE against the fixed identity relationship y = x."""
    base_states, finetuned_states = _paired_matrices(
        base_states,
        finetuned_states,
    )
    base_mean_squares = np.mean(base_states**2, axis=0)
    finetuned_mean_squares = np.mean(finetuned_states**2, axis=0)
    cross_products = np.matmul(base_states.T, finetuned_states) / base_states.shape[0]
    mse = (
        base_mean_squares[:, None]
        + finetuned_mean_squares[None, :]
        - 2.0 * cross_products
    )
    return np.maximum(mse, 0.0)


def matrix_ccc(base_states, finetuned_states):
    """Return pairwise concordance with the literal identity relationship."""
    base_states, finetuned_states = _paired_matrices(
        base_states,
        finetuned_states,
    )
    statistics = _pairwise_statistics(base_states, finetuned_states)
    denominator = (
        statistics["base_variances"][:, None]
        + statistics["finetuned_variances"][None, :]
        + (
            statistics["base_means"][:, None]
            - statistics["finetuned_means"][None, :]
        )
        ** 2
    )
    return np.divide(
        2.0 * statistics["covariance"],
        denominator,
        out=np.full_like(statistics["covariance"], np.nan),
        where=denominator != 0.0,
    )


def _pairwise_statistics(base_states, finetuned_states):
    base_means = np.mean(base_states, axis=0)
    finetuned_means = np.mean(finetuned_states, axis=0)
    centered_base = base_states - base_means
    centered_finetuned = finetuned_states - finetuned_means
    example_count = base_states.shape[0]

    return {
        "base_means": base_means,
        "finetuned_means": finetuned_means,
        "base_variances": np.mean(centered_base**2, axis=0),
        "finetuned_variances": np.mean(centered_finetuned**2, axis=0),
        "covariance": np.matmul(centered_base.T, centered_finetuned)
        / example_count,
    }


def _paired_matrices(
    base_states,
    finetuned_states,
    *,
    require_same_neurons=False,
):
    base_states = np.asarray(base_states, dtype=float)
    finetuned_states = np.asarray(finetuned_states, dtype=float)
    if base_states.ndim != 2 or finetuned_states.ndim != 2:
        raise ValueError("base_states and finetuned_states must both be 2D")
    if base_states.shape[0] != finetuned_states.shape[0]:
        raise ValueError(
            "base_states and finetuned_states must share their example axis: "
            f"{base_states.shape} != {finetuned_states.shape}"
        )
    if base_states.shape[0] == 0:
        raise ValueError("activation matrices must contain at least one example")
    if require_same_neurons and base_states.shape[1] != finetuned_states.shape[1]:
        raise ValueError(
            "cross-minus-base comparisons require the same neuron count: "
            f"{base_states.shape[1]} != {finetuned_states.shape[1]}"
        )
    return base_states, finetuned_states


def calc_transport_latents(U, singular_values, Vt, rank):
    """Factor one fate matrix into paired low-rank neuron coordinates."""
    U = np.asarray(U, dtype=float)
    singular_values = np.asarray(singular_values, dtype=float)
    Vt = np.asarray(Vt, dtype=float)
    if U.ndim != 2 or singular_values.ndim != 1 or Vt.ndim != 2:
        raise ValueError("U, singular_values, and Vt must be an SVD decomposition")
    if U.shape[1] < singular_values.size or Vt.shape[0] < singular_values.size:
        raise ValueError("SVD factor shapes do not match singular_values")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")

    rank_used = min(int(rank), singular_values.size)
    sigma_sqrt = np.sqrt(singular_values[:rank_used])
    base = U[:, :rank_used] * sigma_sqrt
    finetuned = Vt[:rank_used].T * sigma_sqrt
    return base, finetuned


def calc_bidirectional_top_k_connections(affinity, k, relationship):
    """Select signed top-k fate relationships in both directions."""
    affinity = np.asarray(affinity, dtype=float)
    if affinity.ndim != 2:
        raise ValueError("affinity must be 2D")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if relationship not in {"positive", "negative"}:
        raise ValueError("relationship must be either 'positive' or 'negative'")
    return _top_k_connections(affinity, k, relationship), _top_k_connections(
        affinity.T,
        k,
        relationship,
    )


def _top_k_connections(affinity, k, relationship):
    _, candidate_count = affinity.shape
    k_used = min(int(k), candidate_count)
    query_indices = []
    candidate_indices = []
    scores = []

    for query_index, row in enumerate(affinity):
        if relationship == "positive":
            valid_candidates = np.flatnonzero(row > 0.0)
        else:
            valid_candidates = np.flatnonzero(row < 0.0)
        order = np.argsort(-np.abs(row[valid_candidates]), kind="stable")
        selected_candidates = valid_candidates[order[:k_used]]
        query_indices.extend([query_index] * selected_candidates.size)
        candidate_indices.extend(selected_candidates.tolist())
        scores.extend(row[selected_candidates].tolist())

    query_indices = np.asarray(query_indices, dtype=int)
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    scores = np.asarray(scores, dtype=float)
    candidate_degrees = np.bincount(
        candidate_indices,
        minlength=candidate_count,
    )
    return query_indices, candidate_indices, scores, candidate_degrees
