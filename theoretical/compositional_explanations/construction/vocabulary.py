"""Select the tokenizer vocabulary used as compositional search features."""

from collections.abc import Iterable, Iterator, Mapping
from itertools import batched, chain
import string

import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS


SKIP_TOKENS = STOP_WORDS | set(string.punctuation) | {""}


def normalize_dataset(
    dataset: Iterable[Mapping[str, str]],
) -> Iterator[str]:
    """Yield every selected text field as one shared feature corpus."""
    for example in dataset:
        yield from example.values()


def count_token_ids(dataset, tokenizer, batch_size: int = 256) -> np.ndarray:
    """Count tokenizer IDs across every selected dataset column."""
    counts = np.zeros(len(tokenizer), dtype=np.int64)

    for text_batch in batched(normalize_dataset(dataset), batch_size):
        encoded = tokenizer(
            list(text_batch),
            add_special_tokens=False,
            return_attention_mask=False,
        )
        token_ids = np.fromiter(
            chain.from_iterable(encoded["input_ids"]),
            dtype=np.int64,
        )
        counts += np.bincount(token_ids, minlength=len(tokenizer))[: len(tokenizer)]

    return counts


def token_is_searchable(
    token_id: int,
    tokenizer,
    special_token_ids: set[int],
) -> bool:
    """Return whether a token is meaningful as a lexical search feature."""
    if token_id in special_token_ids:
        return False

    token = tokenizer.decode([token_id]).strip().lower()
    return token not in SKIP_TOKENS


def select_feature_token_ids(
    counts: np.ndarray,
    tokenizer,
    top_k: int = 2_000,
) -> list[int]:
    """Select frequent lexical tokens, then append configured POS features."""
    ranked_token_ids = np.argsort(-counts, kind="stable")
    special_token_ids: set[int] = set(tokenizer.all_special_ids or ())
    lexical_token_ids = []

    for token_id in ranked_token_ids:
        token_id = int(token_id)
        if counts[token_id] == 0 or len(lexical_token_ids) == top_k:
            break
        if token_is_searchable(token_id, tokenizer, special_token_ids):
            lexical_token_ids.append(token_id)

    pos_token_ids = [
        token_id
        for token_id in tokenizer.additional_special_tokens_ids
        if (token := tokenizer.decode([token_id]).strip()).startswith("<")
        and token.endswith(">")
    ]
    return lexical_token_ids + pos_token_ids
