import numpy as np


def calc_cross_similarity(task_data):
    base_states = task_data["base"]
    finetuned_states = task_data["finetuned"]

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

    return {
        "pearson": pearson,
        "cosine": cosine,
    }
