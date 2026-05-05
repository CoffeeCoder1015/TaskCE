from datasets import load_dataset
from transformers import AutoTokenizer

from feature import (
    count_model_token_ids,
    top_token_counts,
)
from feature.construct import construct_label_vocab_matrices, construct_vectors


def format_snli_text(example):
    return {"premise": example["premise"], "hypothesis": example["hypothesis"]}


def format_fallacy_text(example):
    return {"text": example["source_article"]}


def format_vitaminc_text(example):
    return {"evidence": example["evidence"], "claim": example["claim"]}


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


def token_outputs(token_ids, counts, tokenizer, limit=10):
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids[:limit]]
    return [
        (token, int(counts[token_id]))
        for token, token_id in zip(tokens, token_ids[:limit])
    ]


def decoded_tokens(token_ids, tokenizer, limit=10):
    return [tokenizer.decode([token_id]) for token_id in token_ids[:limit]]


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
    top_token_ids = top_token_counts(token_counts, tokenizer, top_k=2_000)

    label_vocab_matrices = construct_label_vocab_matrices(
        formatted_dataset,
        tokenizer,
        batch_size=512,
    )
    feature_matrix = construct_vectors(
        formatted_dataset,
        None,
        tokenizer,
        top_token_ids,
        batch_size=512,
    )

    print(f"\nTask: {task_name}")
    print(f"Rows: {len(formatted_dataset)}")
    print(f"Fields: {formatted_dataset.column_names}")
    print(f"Occurrence observations: {int(token_counts.sum())}")
    print(f"Top token count: {len(top_token_ids)}")
    print(f"Label vocab matrix shapes: { {k: v.shape for k, v in label_vocab_matrices.items()} }")
    print(f"Final binary feature matrix shape: {feature_matrix.shape}")
    print(f"Final binary feature matrix nonzeros: {feature_matrix.nnz}")
    print("Top 10 tokens:", token_outputs(top_token_ids, token_counts, tokenizer))
