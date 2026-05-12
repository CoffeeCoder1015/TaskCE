import gc
import os

from datasets import load_dataset
import pandas as pd
from feature.search import search_all
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature import (
    count_model_token_ids,
    top_token_counts,
)
from feature.construct import construct_label_vocab_matrices, construct_vectors
from capture import Capture, CaptureConfig
from capture.postprocessing import prune_min_acts, threshold

MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"
LORA_DIR = "../multitune/output"
RESULT_DIR = "results"
BEAM_RESULTS_CSV = os.path.join(RESULT_DIR, "snli_beam_results.csv")

CLASS_TOKEN_IDS = {
    "entailment": 806,
    "neutral": 25919,
    "contradiction": 10913,
}
CLASS_NAMES = ["entailment", "neutral", "contradiction"]


def format_snli_text(example):
    return {"premise": example["premise"], "hypothesis": example["hypothesis"]}


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


def token_outputs(token_ids, counts, tokenizer, limit=10):
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids[:limit]]
    return [
        (token, int(counts[token_id]))
        for token, token_id in zip(tokens, token_ids[:limit])
    ]


def log_class_token_decodes(tokenizer):
    decoded = {}
    print("Target NLI class token ids:")
    for label in CLASS_NAMES:
        token_id = CLASS_TOKEN_IDS[label]
        token = tokenizer.decode([token_id])
        decoded[label] = token
        print(f"  {label} token_id={token_id} decoded={token!r}")
    return decoded


def resolve_latest_lora_checkpoint(lora_dir, task_name):
    task_dir = os.path.join(lora_dir, task_name)
    if not os.path.isdir(task_dir):
        return None

    checkpoints = [
        os.path.join(task_dir, path)
        for path in os.listdir(task_dir)
        if os.path.isdir(os.path.join(task_dir, path))
    ]
    if not checkpoints:
        return None
    return sorted(checkpoints)[-1]


def load_lm_head_model(model_id, lora_path=None, dtype=torch.bfloat16, device_map="auto"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
    )
    if lora_path is not None:
        print(f"Loading LoRA adapter for lm_head weights: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    model.eval()
    return model


def get_classification_weights(model):
    class_token_ids = [CLASS_TOKEN_IDS[label] for label in CLASS_NAMES]
    lm_head_weight = model.lm_head.weight.detach().cpu().to(torch.float32)
    weights = lm_head_weight[class_token_ids].T
    print(f"Classification weights shape: {tuple(weights.shape)}")
    return weights


os.makedirs(RESULT_DIR, exist_ok=True)
dataset = load_dataset("snli", split="validation", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
log_class_token_decodes(tokenizer)

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
print("First 10 feature names:", [formula for formula, _ in feature_vectors[:10]])
print("Top 10 tokens:", token_outputs(top_token_ids, token_counts, tokenizer))

capture_results = Capture(
    model_id=MODEL_ID,
    lora_dir=LORA_DIR,
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

alpha = 0.055
base_binary_acts = threshold(activations_base.states, alpha=alpha)
base_pruned_acts, base_neuron_ids = prune_min_acts(base_binary_acts)
fine_binary_acts = threshold(activations_fine.states, alpha=alpha)
fine_pruned_acts, fine_neuron_ids = prune_min_acts(fine_binary_acts)

def summarize_postprocessing(binary_acts, pruned_acts, neuron_ids):
    return {
        "binary_shape": tuple(binary_acts.shape),
        "pruned_shape": tuple(pruned_acts.shape),
        "kept_neuron_count": int(neuron_ids.numel()),
        "kept_neuron_preview": neuron_ids[:20].tolist(),
    }

print( "Base postprocessing:", summarize_postprocessing(base_binary_acts, base_pruned_acts, base_neuron_ids),)
print( "Finetuned postprocessing:", summarize_postprocessing(fine_binary_acts, fine_pruned_acts, fine_neuron_ids),)

search_results = search_all(fine_pruned_acts[:, :10], feature_vectors)
lora_path = resolve_latest_lora_checkpoint(LORA_DIR, "snli")
lm_head_model = load_lm_head_model(MODEL_ID, lora_path=lora_path)
try:
    classification_weights = get_classification_weights(lm_head_model)
finally:
    del lm_head_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

records = []
searched_neuron_ids = set()

def class_weight_record(classification_weights, neuron_id):
    weight_ent, weight_neut, weight_contr = classification_weights[neuron_id].tolist()
    return {
        "weight_ent": float(weight_ent),
        "weight_neut": float(weight_neut),
        "weight_contr": float(weight_contr),
    }


def search_result_record(neuron_id, formula, iou, classification_weights):
    return {
        "neuron": neuron_id,
        "formula": formula,
        "iou": float(iou),
        **class_weight_record(classification_weights, neuron_id),
    }

for result in search_results:
    neuron_id = int(fine_neuron_ids[result.activation_index])
    searched_neuron_ids.add(neuron_id)
    records.append(
        search_result_record(
            neuron_id,
            result.best_formula,
            result.best_score,
            classification_weights,
        )
    )

for neuron_id in range(int(fine_binary_acts.shape[1])):
    if neuron_id in searched_neuron_ids:
        continue
    records.append(
        search_result_record(
            neuron_id,
            "LOW_ACTS_PRUNED",
            0.0,
            classification_weights,
        )
    )

beam_df = pd.DataFrame(
    records,
    columns=[
        "neuron",
        "formula",
        "iou",
        "weight_ent",
        "weight_neut",
        "weight_contr",
    ],
).sort_values("iou", ascending=False, ignore_index=True)

print("\nSearch result dataframe stats:")
print(f"Rows: {len(beam_df)}")
print(f"Columns: {list(beam_df.columns)}")
print("\nNumeric summary:")
print(beam_df.describe(include="number").to_string())
print("\nTop 10 formulas:")
print(beam_df["formula"].value_counts().head(10).to_string())

os.makedirs(os.path.dirname(BEAM_RESULTS_CSV), exist_ok=True)
beam_df.to_csv(BEAM_RESULTS_CSV, index=False)
print(f"Saved search result dataframe: {BEAM_RESULTS_CSV}")
