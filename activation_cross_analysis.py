import numpy as np


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


def calc_bidirectional_top_k_connections(affinity, k):
    affinity = np.asarray(affinity, dtype=float)
    if affinity.ndim != 2:
        raise ValueError("affinity must be 2D")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    return _top_k_connections(affinity, k), _top_k_connections(affinity.T, k)


def _top_k_connections(affinity, k):
    query_count, candidate_count = affinity.shape
    k_used = min(int(k), candidate_count)
    if k_used == 0:
        return (
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty(0, dtype=float),
            np.zeros(candidate_count, dtype=int),
        )

    nearest = np.argsort(-affinity, axis=1, kind="stable")[:, :k_used]
    query_indices = np.repeat(np.arange(query_count), k_used)
    candidate_indices = nearest.reshape(-1)
    scores = affinity[query_indices, candidate_indices]
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
        : base_states.shape[0],
        base_states.shape[0] :,
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
