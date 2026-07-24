"""Construct binary dataset vectors for compositional search features."""

from collections.abc import Sequence
from itertools import batched

import numpy as np
from scipy.sparse import csr_matrix
from sympy import Symbol

from .vocabulary import (
    count_token_ids,
    select_feature_token_ids,
)


def build_token_presence_matrices(
    dataset,
    tokenizer,
    batch_size: int = 256,
) -> dict[str, csr_matrix]:
    """Build one example-by-vocabulary presence matrix per text column."""
    matrices = {}

    for column in dataset.column_names:
        row_indices = []
        token_indices = []

        for indexed_batch in batched(enumerate(dataset[column]), batch_size):
            indexed_batch = list(indexed_batch)
            texts = [text for _row_index, text in indexed_batch]
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
            )

            for (row_index, _text), token_ids in zip(
                indexed_batch,
                encoded["input_ids"],
                strict=True,
            ):
                unique_token_ids = set(token_ids)
                row_indices.extend([row_index] * len(unique_token_ids))
                token_indices.extend(unique_token_ids)

        matrices[column] = csr_matrix(
            (
                np.ones(len(row_indices), dtype=np.bool_),
                (row_indices, token_indices),
            ),
            shape=(len(dataset), len(tokenizer)),
            dtype=np.bool_,
        )

    return matrices


def construct_feature_vectors(
    dataset,
    tokenizer,
    columns: Sequence[str],
    *,
    batch_size: int = 512,
    top_k: int = 2_000,
) -> list[tuple[Symbol, np.ndarray]]:
    """Construct the complete atomic feature set for one analysis dataset."""
    feature_dataset = dataset.select_columns(list(columns))
    token_counts = count_token_ids(
        feature_dataset,
        tokenizer,
        batch_size=batch_size,
    )
    feature_token_ids = select_feature_token_ids(
        token_counts,
        tokenizer,
        top_k=top_k,
    )
    token_presence_matrices = build_token_presence_matrices(
        feature_dataset,
        tokenizer,
        batch_size=batch_size,
    )
    feature_vectors = []

    for column, matrix in token_presence_matrices.items():
        selected_vectors = matrix[:, feature_token_ids].toarray()

        for feature_index, token_id in enumerate(feature_token_ids):
            vector = selected_vectors[:, feature_index]
            if not vector.any():
                continue

            token = tokenizer.decode([token_id])
            feature_vectors.append(
                (
                    Symbol(f"{column}:{token}"),
                    vector,
                )
            )

    return feature_vectors
