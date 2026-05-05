from collections.abc import Iterable
from itertools import chain

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def format_snli_text(example):
    return {"premise": example["premise"], "hypothesis": example["hypothesis"]}


def format_fallacy_text(example):
    return {"text": example["source_article"]}


def format_vitaminc_text(example):
    return {"evidence": example["evidence"], "claim": example["claim"]}


def normalize_dataset(dataset):
    for example in dataset:
        for _, sentence in example.items():
            yield sentence


def batched(iterable, batch_size):
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


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
        
        # For debugging (differnet HF Transformer versions have different batching behavior)
        # print(input_ids)
        # exit()

        flat_ids = np.fromiter(chain.from_iterable(input_ids), dtype=np.int64)
        counts += np.bincount(flat_ids, minlength=vocab_size)[:vocab_size]

    return counts


def top_token_counts(
    counts: np.ndarray,
    tokenizer,
    top_k=2_000,
):
    token_ids = np.argsort(-counts)[:top_k].tolist()
    if counts[token_ids[-1]] == 0:
        print(f"Warning: top_k={top_k} includes zero-count tokens")
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
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
