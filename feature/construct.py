import numpy as np
from scipy.sparse import csr_matrix, hstack

from feature.batch import batched
from feature.find import count_model_token_ids, top_token_counts
from feature.formula import Leaf
from sympy import Symbol


def identity_column_selector(columns: list[str]):
    """Return a dataset row mapper that keeps only the requested columns."""
    return lambda example: {column: example[column] for column in columns}


def construct_label_vocab_matrices(
    dataset,
    tokenizer,
    feature_text_selector=None,
    batch_size=256,
):
    feature_dataset = (
        dataset.map(feature_text_selector, remove_columns=dataset.column_names)
        if feature_text_selector is not None
        else dataset
    )
    vocab_size = len(tokenizer) 
    matrices = {}

    for label in feature_dataset.column_names:
        # construct binary vectors for each label (premise, hypothesis, etc)
        rows = []
        cols = []

        for row_start, batch in batched_with_start(feature_dataset[label], batch_size):
            encoded = tokenizer(
                batch,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            for batch_offset, token_ids in enumerate(encoded["input_ids"]):
                row_idx = row_start + batch_offset
                for token_id in set(token_ids): # set compresses into uniquely activated tokens 
                    rows.append(row_idx) # <- token presence for the example axis
                    cols.append(token_id) # <- binary vectors axis

        data = np.ones(len(rows), dtype=np.int8)
        matrices[label] = csr_matrix(
            (data, (rows, cols)),
            shape=(len(feature_dataset), vocab_size),
            dtype=np.int8,
        )

    return matrices


def batched_with_start(iterable, batch_size):
    row_start = 0
    for batch in batched(iterable, batch_size):
        yield row_start, batch
        row_start += len(batch)


def construct_vectors(label_vocab_matrices, features, tokenizer):
    # Pick out from all binary vectors the most frequently activated
    sparse_feature_matrix = hstack(
        [matrix[:, features] for matrix in label_vocab_matrices.values()],
        format="csr",
        dtype=np.int8,
    )
    dense_feature_matrix = sparse_feature_matrix.toarray()
    feature_vectors = []
    feature_index = 0

    for label in label_vocab_matrices:
        for token_id in features:
            decoded_token = tokenizer.decode([token_id])
            feature_vectors.append(
                (
                    Symbol(f"{label}:{decoded_token}"),
                    dense_feature_matrix[:, feature_index],
                )
            )
            feature_index += 1

    return feature_vectors

def ConstructFeatures(
    dataset,
    tokenizer,
    batch_size=512,
    top_k=2000,
    feature_text_selector=None,
):
    feature_dataset = (
        dataset.map(feature_text_selector, remove_columns=dataset.column_names)
        if feature_text_selector is not None
        else dataset
    )
    token_counts = count_model_token_ids(
        dataset=feature_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size
    )
    top_token_ids = top_token_counts(token_counts, tokenizer, top_k=top_k)

    label_vocab_matrices = construct_label_vocab_matrices(
        feature_dataset,
        tokenizer,
        batch_size=batch_size,
    )
    feature_vectors = construct_vectors(
        label_vocab_matrices,
        top_token_ids,
        tokenizer,
    )
    return feature_vectors
