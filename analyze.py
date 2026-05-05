from datasets import load_dataset
from transformers import AutoTokenizer

from feature import (
    count_model_token_ids,
    top_token_counts,
)
from feature.construct import construct_label_vocab_matrices, construct_vectors
from capture import Capture, CaptureConfig

def format_snli_text(example):
    return {"premise": example["premise"], "hypothesis": example["hypothesis"]}

def token_outputs(token_ids, counts, tokenizer, limit=10):
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids[:limit]]
    return [
        (token, int(counts[token_id]))
        for token, token_id in zip(tokens, token_ids[:limit])
    ]

dataset = load_dataset("snli", split="validation", trust_remote_code=True)
model_id = "LiquidAI/LFM2.5-1.2B-Thinking"
tokenizer = AutoTokenizer.from_pretrained(model_id)

formatted_dataset = dataset.map(
    format_snli_text,
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
feature_vectors = construct_vectors(
    label_vocab_matrices,
    top_token_ids,
    tokenizer,
)

print(f"Rows: {len(formatted_dataset)}")
print(f"Fields: {formatted_dataset.column_names}")
print(f"Occurrence observations: {int(token_counts.sum())}")
print(f"Top token count: {len(top_token_ids)}")
print(f"Label vocab matrix shapes: { {k: v.shape for k, v in label_vocab_matrices.items()} }")
print(f"Final binary feature count: {len(feature_vectors)}")
print(f"Final binary feature vector length: {len(feature_vectors[0][1])}")
print(f"Final binary feature nonzeros: {sum(vector.sum() for _, vector in feature_vectors)}")
print("First 10 feature names:", [formula.flatten() for formula, _ in feature_vectors[:10]])
print("Top 10 tokens:", token_outputs(top_token_ids, token_counts, tokenizer))


SNLI_LABELS = ["entailment", "neutral", "contradiction"]
SNLI_SYSTEM_PROMPT = {
    "role": "system",
    "content": """Determine the relationship between the `premise`and `hypothesis` and respond with an answer.
You must respond with an answer of `entailment`, `neutral` or `contradiction`""",
}
def format_snli_for_capture(example):
    test_example = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    example["prompt"] = [
        SNLI_SYSTEM_PROMPT,
        {"role": "user", "content": test_example},
    ]

    example["answer"] = SNLI_LABELS[example["label"]]
    return example

capture_results = Capture(
    model_id="LiquidAI/LFM2.5-1.2B-Thinking",
    lora_dir="../multitune/output",  # Assumed to be used in conjunction with https://github.com/CoffeeCoder1015/multitune
    tasks=[
        CaptureConfig(
            name="snli",
            dataset=dataset,
            data_formatter=format_snli_for_capture,
        )
    ],
    layer=-2,
    batch_size=256,
)