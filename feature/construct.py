import numpy as np
from scipy.sparse import csr_matrix, hstack

from feature.batch import batched


def construct_label_vocab_matrices(dataset, tokenizer, formatter=None, batch_size=256):
    formatted_dataset = (
        dataset.map(formatter, remove_columns=dataset.column_names)
        if formatter is not None
        else dataset
    )
    vocab_size = len(tokenizer) 
    matrices = {}

    for label in formatted_dataset.column_names:
        # construct binary vectors for each label (premise, hypothesis, etc)
        rows = []
        cols = []

        for row_start, batch in batched_with_start(formatted_dataset[label], batch_size):
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
            shape=(len(formatted_dataset), vocab_size),
            dtype=np.int8,
        )

    return matrices


def batched_with_start(iterable, batch_size):
    row_start = 0
    for batch in batched(iterable, batch_size):
        yield row_start, batch
        row_start += len(batch)


def construct_vectors(label_vocab_matrices, features, tokenizer):
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
                    {
                        "label": label,
                        "token_id": token_id,
                        "token": decoded_token,
                        "name": f"{label}:{decoded_token}",
                    },
                    dense_feature_matrix[:, feature_index],
                )
            )
            feature_index += 1

    return feature_vectors
