import os
from datasets import load_dataset
from analysis.saving import (
    build_neuron_search_results_dataframe,
    save_neuron_search_results_csv,
)
from transformers import AutoTokenizer

from capture.classification_weights import get_classification_weights
from capture.captureConfig import CaptureConfig
from capture.capturer import Capture
from capture.postprocessing import threshold, prune_min_acts

from feature.construct import ConstructFeatures, identity_column_selector
from feature.search import search_all, searchConfig


SNLI_LABELS = ("entailment", "neutral", "contradiction")
SNLI_CLASS_TOKEN_IDS = {
    "entailment": 806,
    "neutral": 25919,
    "contradiction": 10913,
}
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


VITAMINC_LABELS = ("supports", "refutes", "not enough info")
VITAMINC_CLASS_TOKEN_IDS = {
    "supports": 56744,
    "refutes": 1891,
    "not enough info": 2897,
}
VITAMINC_SYSTEM_PROMPT = {
    "role": "system",
    "content": """Determine the relationship between the `claim`and `evidence` and respond with an answer.
You must respond with an answer of `supports`, `refutes` or `not enough info`""",
}


def format_vitaminc_for_capture(example):
    test_example = f"Evidence: {example['evidence']}\nClaim: {example['claim']}"
    example["prompt"] = [
        VITAMINC_SYSTEM_PROMPT,
        {"role": "user", "content": test_example},
    ]
    example["answer"] = str(example["label"]).strip().lower()
    return example


def post_process_activations(activation, alpha, min_acts):
    binarized_activations = threshold(activation.states, alpha=alpha)
    kept_bin_activations, kept_neurons = prune_min_acts(
        binarized_activations,
        min_acts=min_acts,
    )
    return binarized_activations, kept_bin_activations, kept_neurons


if __name__ == "__main__":
    snli = load_dataset("snli", split="validation")
    vitaminc = load_dataset("tals/vitaminc", split="validation[:10_000]")

    model_id = "LiquidAI/LFM2.5-1.2B-Thinking"
    feature_tokenizer = AutoTokenizer.from_pretrained(model_id)
    snli_features = ConstructFeatures(
        snli,
        feature_tokenizer,
        feature_text_selector=identity_column_selector(["premise", "hypothesis"]),
    )
    claim_features = ConstructFeatures(
        vitaminc,
        feature_tokenizer,
        feature_text_selector=identity_column_selector(["claim", "evidence"]),
    )

    lora_dir = "../multitune/output"

    captured_results = Capture(
        model_id,
        lora_dir,
        tasks=[
            CaptureConfig("snli", snli, format_snli_for_capture),
            CaptureConfig("claim", vitaminc, format_vitaminc_for_capture),
        ],
        layer=-2,
    )

    output_dir = "results"
    search_config = searchConfig(
        formula_length=5,
        pruned_queue_size=10,
        max_iterations=10,
        length_penalty=0.0,
    )
    search_tasks = [
        {
            "name": "snli",
            "alpha": 0.53,
            "min_acts": 500,
            "features": snli_features,
            "class_token_ids": SNLI_CLASS_TOKEN_IDS,
            "labels": SNLI_LABELS,
        },
        {
            "name": "claim",
            "alpha": 0.5,
            "min_acts": 500,
            "features": claim_features,
            "class_token_ids": VITAMINC_CLASS_TOKEN_IDS,
            "labels": VITAMINC_LABELS,
        },
    ]

    for task in search_tasks:
        name = task["name"]
        alpha = task["alpha"]
        min_acts = task["min_acts"]
        features = task["features"]
        activations = captured_results[name]["finetuned"]
        binarized_activations, kept_activations, kept_neurons = post_process_activations(
            activations,
            alpha,
            min_acts,
        )

        search_results = search_all(
            kept_activations,
            features,
            num_workers=10,
            config=search_config,
        )

        classification_weights = get_classification_weights( model_id, lora_dir, name, task["class_token_ids"],)
        weight_column_names = tuple( f"weight_{label.replace(' ', '_')}" for label in task["labels"])

        dataframe = build_neuron_search_results_dataframe(
            search_results,
            kept_neurons,
            binarized_activations.shape[1],
            classification_weights,
            weight_column_names,
        )

        output_csv_path = os.path.join(output_dir, f"{name}_beam_results.csv")
        save_neuron_search_results_csv(dataframe, output_csv_path)
