import gc
import os

from datasets import load_dataset
import pandas as pd
from feature.search import LevelSearch, Search, search_worker
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature import (
    count_model_token_ids,
    top_token_counts,
)
from feature.beamsearch import beamsearch_all
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

def token_outputs(token_ids, counts, tokenizer, limit=10):
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids[:limit]]
    return [
        (token, int(counts[token_id]))
        for token, token_id in zip(tokens, token_ids[:limit])
    ]

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

def summarize_postprocessing(binary_acts, pruned_acts, neuron_ids):
    return {
        "binary_shape": tuple(binary_acts.shape),
        "pruned_shape": tuple(pruned_acts.shape),
        "kept_neuron_count": int(neuron_ids.numel()),
        "kept_neuron_preview": neuron_ids[:20].tolist(),
    }


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


def log_class_token_decodes(tokenizer):
    decoded = {}
    print("Target NLI class token ids:")
    for label in CLASS_NAMES:
        token_id = CLASS_TOKEN_IDS[label]
        token = tokenizer.decode([token_id])
        decoded[label] = token
        print(f"  {label} token_id={token_id} decoded={token!r}")
    return decoded


def get_classification_weights(model):
    class_token_ids = [CLASS_TOKEN_IDS[label] for label in CLASS_NAMES]
    lm_head_weight = model.lm_head.weight.detach().cpu().to(torch.float32)
    weights = lm_head_weight[class_token_ids].T
    print(f"Classification weights shape: {tuple(weights.shape)}")
    return weights


def build_beam_results_dataframe(
    beam_results,
    neuron_ids,
    classification_weights,
    total_neurons=None,
    activation_counts=None,
):
    records = []
    searched_neuron_ids = set()

    def activation_count_for(neuron_id):
        if activation_counts is None:
            return None
        return int(activation_counts[neuron_id].item())

    def weight_record(neuron_id):
        weight_ent, weight_neut, weight_contr = classification_weights[neuron_id].tolist()
        return {
            "weight_ent": float(weight_ent),
            "weight_neut": float(weight_neut),
            "weight_contr": float(weight_contr),
        }

    for result in beam_results:
        original_neuron_id = int(neuron_ids[result.activation_index])
        searched_neuron_ids.add(original_neuron_id)
        formula, iou = result.best
        records.append(
            {
                "neuron": original_neuron_id,
                "iou": float(iou),
                "formula": formula,
                **weight_record(original_neuron_id),
                "activation_count": activation_count_for(original_neuron_id),
                "was_pruned": False,
                "prune_reason": "",
            }
        )

    if total_neurons is not None:
        for neuron_id in range(int(total_neurons)):
            if neuron_id in searched_neuron_ids:
                continue
            records.append(
                {
                    "neuron": neuron_id,
                    "iou": 0.0,
                    "formula": "LOW_ACTS_PRUNED",
                    **weight_record(neuron_id),
                    "activation_count": activation_count_for(neuron_id),
                    "was_pruned": True,
                    "prune_reason": "low_acts",
                }
            )

    return pd.DataFrame(
        records,
        columns=[
            "neuron",
            "iou",
            "formula",
            "weight_ent",
            "weight_neut",
            "weight_contr",
            "activation_count",
            "was_pruned",
            "prune_reason",
        ],
    ).sort_values("iou", ascending=False, ignore_index=True)


def print_dataframe_stats(df):
    print("\nBeam result dataframe stats:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if df.empty:
        return
    print("\nNumeric summary:")
    print(df.describe(include="number").to_string())
    print("\nTop 10 formulas:")
    print(df["formula"].value_counts().head(10).to_string())


def save_beam_results_dataframe(df, output_path=BEAM_RESULTS_CSV):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print_dataframe_stats(df)
    df.to_csv(output_path, index=False)
    print(f"Saved beam result dataframe: {output_path}")
    return output_path


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
    lora_dir=LORA_DIR,  # Assumed to be used in conjunction with https://github.com/CoffeeCoder1015/multitune
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

print(
    "Base postprocessing:",
    summarize_postprocessing(base_binary_acts, base_pruned_acts, base_neuron_ids),
)
print(
    "Finetuned postprocessing:",
    summarize_postprocessing(fine_binary_acts, fine_pruned_acts, fine_neuron_ids),
)

# LevelSearch(fine_pruned_acts[:,0],feature_vectors)
search_worker(fine_pruned_acts,list(range(0,10)),feature_vectors)

# beam_size = 30
# max_formula_length = 6
# complexity_penalty = 1.00
# score_batch_size = 4096
# beamsearch_workers = 8
# beam_results = beamsearch_all(
#     feature_vectors,
#     fine_pruned_acts,
#     beam_size=beam_size,
#     formula_length=max_formula_length,
#     complexity_penalty=complexity_penalty,
#     score_batch_size=score_batch_size,
#     num_workers=beamsearch_workers,
# )


# print("Beam search results:")
# for result in beam_results:
#     original_neuron_id = int(fine_neuron_ids[result.activation_index])
#     best_formula, best_iou = result.best
#     best_noncomp_formula, best_noncomp_iou = result.best_noncomp

#     print(
#         f"Neuron {original_neuron_id}: "
#         f"best_iou={best_iou:.4f} "
#         f"best={best_formula}"
#     )
#     print(
#         f"  best_noncomp_iou={best_noncomp_iou:.4f} "
#         f"best_noncomp={best_noncomp_formula}"
#     )

# lora_path = resolve_latest_lora_checkpoint(LORA_DIR, "snli")
# lm_head_model = load_lm_head_model(MODEL_ID, lora_path=lora_path)
# try:
#     classification_weights = get_classification_weights(lm_head_model)
# finally:
#     del lm_head_model
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

# beam_df = build_beam_results_dataframe(
#     beam_results,
#     fine_neuron_ids,
#     classification_weights,
#     total_neurons=fine_binary_acts.shape[1],
#     activation_counts=fine_binary_acts.sum(dim=0),
# )
# beam_results_path = save_beam_results_dataframe(beam_df)