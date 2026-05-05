# This is to find features.

from collections.abc import Iterable, Iterator, Mapping
from itertools import chain

import numpy as np

from feature.batch import batched
from feature.skip import token_should_be_skipped


def normalize_dataset(dataset: Iterable[Mapping[str, str]]) -> Iterator[str]:
    for example in dataset:
        for sentence in example.values():
            yield sentence


def count_model_token_ids(dataset, tokenizer, batch_size=256):
    vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else tokenizer.vocab_size
    counts = np.zeros(vocab_size, dtype=np.int64)
    sentences = normalize_dataset(dataset)

    for batch in batched(sentences, batch_size):
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        input_ids = encoded["input_ids"]
        flat_ids = np.fromiter(chain.from_iterable(input_ids), dtype=np.int64)
        counts += np.bincount(flat_ids, minlength=vocab_size)[:vocab_size]

    return counts


def special_token_ids(tokenizer) -> set[int]:
    return set(tokenizer.all_special_ids or [])


def top_token_counts(
    counts: np.ndarray,
    tokenizer,
    top_k=2_000,
):
    sorted_token_ids = np.argsort(-counts).tolist()
    skipped_special_ids = special_token_ids(tokenizer)
    token_ids = []
    i = 0

    while (
        len(token_ids) < top_k
        and i < len(sorted_token_ids)
        and counts[sorted_token_ids[i]] > 0
    ):
        token_id = sorted_token_ids[i]
        i += 1

        if token_should_be_skipped(token_id, tokenizer, skipped_special_ids):
            continue

        token_ids.append(token_id)

    if len(token_ids) < top_k:
        print(f"Warning: top_k={top_k} includes zero-count tokens")

    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    return [(token, int(counts[token_id])) for token, token_id in zip(tokens, token_ids)]
