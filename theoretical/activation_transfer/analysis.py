"""Cross-model activation-transfer calculations.

Matrices use the notebook's established ``[neurons, examples]`` orientation.
"""

from pathlib import Path

import numpy as np


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
    return activation.detach().cpu().to(torch.float32).numpy().T


def run(
    base_path,
    finetuned_path,
    *,
    task_name,
    output_path,
):
    base_states = load_activation(base_path)
    finetuned_states = load_activation(finetuned_path)
    if base_states.shape != finetuned_states.shape:
        raise ValueError(
            "Base and fine-tuned activations must have the same shape: "
            f"{base_states.shape} != {finetuned_states.shape}."
        )

    cross_pearson, cross_cosine = calc_cross_similarity(
        base_states,
        finetuned_states,
    )
    base_pearson, base_cosine = calc_cross_similarity(
        base_states,
        base_states,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        cross_pearson=cross_pearson,
        cross_cosine=cross_cosine,
        base_pearson=base_pearson,
        base_cosine=base_cosine,
        cross_minus_base_pearson=cross_pearson - base_pearson,
        cross_minus_base_cosine=cross_cosine - base_cosine,
    )
    return output_path


def calc_transport_latents(U, singular_values, Vt, rank):
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
    affinity = np.asarray(affinity, dtype=float)
    if affinity.ndim != 2:
        raise ValueError("affinity must be 2D")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if relationship not in {"correlation", "anticorrelation"}:
        raise ValueError(
            "relationship must be either 'correlation' or 'anticorrelation'"
        )
    return _top_k_connections(affinity, k, relationship), _top_k_connections(
        affinity.T, k, relationship
    )


def _top_k_connections(affinity, k, relationship):
    _, candidate_count = affinity.shape
    k_used = min(int(k), candidate_count)
    if k_used == 0:
        return (
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty(0, dtype=float),
            np.zeros(candidate_count, dtype=int),
        )

    query_indices = []
    candidate_indices = []
    scores = []
    for query_index, row in enumerate(affinity):
        valid_candidates = np.flatnonzero(row > 0.0)
        if relationship == "anticorrelation":
            valid_candidates = np.flatnonzero(row < 0.0)
        order = np.argsort(-np.abs(row[valid_candidates]), kind="stable")
        selected_candidates = valid_candidates[order[:k_used]]
        query_indices.extend([query_index] * selected_candidates.size)
        candidate_indices.extend(selected_candidates.tolist())
        scores.extend(row[selected_candidates].tolist())

    query_indices = np.asarray(query_indices, dtype=int)
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    scores = np.asarray(scores, dtype=float)
    candidate_degrees = np.bincount(candidate_indices, minlength=candidate_count)
    return query_indices, candidate_indices, scores, candidate_degrees


def calc_cross_similarity(base_states, finetuned_states):
    base_states = np.asarray(base_states, dtype=float)
    finetuned_states = np.asarray(finetuned_states, dtype=float)
    if base_states.ndim != 2 or finetuned_states.ndim != 2:
        raise ValueError("base_states and finetuned_states must both be 2D")
    if base_states.shape[1] != finetuned_states.shape[1]:
        raise ValueError("base_states and finetuned_states must share their example axis")

    pearson = np.corrcoef(base_states, finetuned_states)[
        : base_states.shape[0], base_states.shape[0] :
    ]
    dot_products = np.matmul(base_states, finetuned_states.T)
    base_norms = np.linalg.norm(base_states, axis=1)
    finetuned_norms = np.linalg.norm(finetuned_states, axis=1)
    norm_products = base_norms[:, None] * finetuned_norms[None, :]
    cosine = np.divide(
        dot_products,
        norm_products,
        out=np.zeros_like(dot_products, dtype=float),
        where=norm_products != 0.0,
    )
    return pearson, cosine
