from datasets import load_dataset
from transformers import AutoTokenizer

from feature import count_model_token_ids, top_token_counts


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
    tokens = [tokenizer.decode([token_id]) for token_id in top_token_ids]
    top_tokens = [
        (token, int(token_counts[token_id]))
        for token, token_id in zip(tokens, top_token_ids)
    ]
    print(f"\nTask: {task_name}")
    print(f"Top token count: {len(top_tokens)}")
    print("Top 10 tokens:", top_tokens[:10])
