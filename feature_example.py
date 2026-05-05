from collections.abc import Iterable, Iterator, Mapping
from itertools import chain
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def format_snli_text(example: Mapping[str, Any]) -> dict[str, str]:
    return {"premise": example["premise"], "hypothesis": example["hypothesis"]}


def format_fallacy_text(example: Mapping[str, Any]) -> dict[str, str]:
    return {"text": example["source_article"]}


def format_vitaminc_text(example: Mapping[str, Any]) -> dict[str, str]:
    return {"evidence": example["evidence"], "claim": example["claim"]}


def normalize_dataset(
    dataset: Iterable[Mapping[str, Any]],
) -> Iterator[tuple[str, str]]:
    for example in dataset:
        for _, sentence in example.items():
            yield sentence


def batched(iterable: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def count_model_token_ids(
    dataset: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    batch_size: int = 256,
) -> np.ndarray:
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
        
        # For debugging (differnet HF Transformer versions have different batching behavior)
        # print(input_ids)
        # exit()

        flat_ids = np.fromiter(chain.from_iterable(input_ids), dtype=np.int64)
        counts += np.bincount(flat_ids, minlength=vocab_size)[:vocab_size]

    return counts


def top_token_counts(
    counts: np.ndarray,
    tokenizer: Any,
    top_k: int = 2_000,
) -> list[tuple[str, int]]:
    token_ids = np.argsort(-counts)[:top_k].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return [(token, int(counts[token_id])) for token, token_id in zip(tokens, token_ids)]



model_id = "LiquidAI/LFM2.5-1.2B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
datasets = {
    "snli": load_dataset("snli", split="validation", trust_remote_code=True),
    "fallacy": load_dataset(
        "tasksource/logical-fallacy",
        split="dev",
        trust_remote_code=True,
    ),
    "claim": load_dataset("tals/vitaminc", split="validation", trust_remote_code=True),
}
task_formatters = {
    "snli": format_snli_text,
    "fallacy": format_fallacy_text,
    "claim": format_vitaminc_text,
}

for task_name, dataset in datasets.items():
    formatted_dataset = dataset.map(
        task_formatters[task_name],
        remove_columns=dataset.column_names,
    )
    token_counts = count_model_token_ids(
        formatted_dataset,
        tokenizer,
        batch_size=512,
    )
    top_tokens = top_token_counts(token_counts, tokenizer, top_k=2_000)
    print(f"\nTask: {task_name}")
    print(f"Top token count: {len(top_tokens)}")
    print("Top 10 tokens:", top_tokens[:10])
