import numpy as np
from scipy.sparse import csr_matrix, hstack

from feature.batch import batched


def construct_label_vocab_matrices(dataset, tokenizer, formatter=None, batch_size=256):
    formatted_dataset = (
        dataset.map(formatter, remove_columns=dataset.column_names)
        if formatter is not None
        else dataset
    )
    vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else tokenizer.vocab_size
    matrices = {}

    for label in formatted_dataset.column_names:
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
                for token_id in set(token_ids):
                    rows.append(row_idx)
                    cols.append(token_id)

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


def construct_vectors(dataset, formatter, tokenizer, features, batch_size=256):
    matrices = construct_label_vocab_matrices(
        dataset,
        tokenizer,
        formatter=formatter,
        batch_size=batch_size,
    )
    return hstack(
        [matrix[:, features] for matrix in matrices.values()],
        format="csr",
        dtype=np.int8,
    )
