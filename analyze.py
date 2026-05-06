from datasets import load_dataset
from transformers import AutoTokenizer

from feature import (
    count_model_token_ids,
    top_token_counts,
)
from feature.beamsearch import beamsearch_all
from feature.construct import construct_label_vocab_matrices, construct_vectors
from capture import Capture, CaptureConfig
from capture.postprocessing import prune_min_acts, threshold

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
top_token_ids = top_token_counts(token_counts, tokenizer, top_k=4_000)

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

activations_base = capture_results["snli"]["base"]
activations_fine = capture_results["snli"]["finetuned"]
print(activations_base)
print(activations_fine)


ALPHA = 0.055
base_binary_acts = threshold(activations_base.states, alpha=ALPHA)
base_pruned_acts, base_neuron_ids = prune_min_acts(base_binary_acts)
fine_binary_acts = threshold(activations_fine.states, alpha=ALPHA)
fine_pruned_acts, fine_neuron_ids = prune_min_acts(fine_binary_acts)

def summarize_postprocessing(binary_acts, pruned_acts, neuron_ids):
    return {
        "binary_shape": tuple(binary_acts.shape),
        "pruned_shape": tuple(pruned_acts.shape),
        "kept_neuron_count": int(neuron_ids.numel()),
        "kept_neuron_preview": neuron_ids[:20].tolist(),
    }

print(
    "Base postprocessing:",
    summarize_postprocessing(base_binary_acts, base_pruned_acts, base_neuron_ids),
)
print(
    "Finetuned postprocessing:",
    summarize_postprocessing(fine_binary_acts, fine_pruned_acts, fine_neuron_ids),
)

BEAM_SIZE = 10
MAX_FORMULA_LENGTH = 6
COMPLEXITY_PENALTY = 1.00
SCORE_BATCH_SIZE = 128
beam_results = beamsearch_all(
    feature_vectors,
    fine_pruned_acts,
    beam_size=BEAM_SIZE,
    formula_length=MAX_FORMULA_LENGTH,
    complexity_penalty=COMPLEXITY_PENALTY,
    score_batch_size=SCORE_BATCH_SIZE,
)

print("Beam search results:")
for result in beam_results:
    original_neuron_id = int(fine_neuron_ids[result.activation_index])
    best_formula, best_iou = result.best
    best_noncomp_formula, best_noncomp_iou = result.best_noncomp

    print(
        f"Neuron {original_neuron_id}: "
        f"best_iou={best_iou:.4f} "
        f"best={best_formula}"
    )
    print(
        f"  best_noncomp_iou={best_noncomp_iou:.4f} "
        f"best_noncomp={best_noncomp_formula}"
    )
    print("  samples:")
    for sample_iou, sample_formula in result.samples:
        print(f"    {sample_iou:.4f} {sample_formula}")
