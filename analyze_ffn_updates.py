import os

from datasets import load_dataset
from transformers import AutoTokenizer
from analysis import run_ablation, run_ablation_analysis
from analysis.activation_diagnostics import (
    save_activation_diagnostics as save_activation_diagnostics_artifacts,
)
from analysis.ablation_inference import AblationInferenceEngine, AblationTaskConfig
from analysis.saving import (
    build_neuron_search_results_dataframe,
    save_neuron_search_results_csv,
)

from capture.classification_weights import (
    get_classification_weights,
)
from capture import Capture, CaptureConfig, save_captured_activations
from capture.postprocessing import threshold, prune_min_acts

from feature.construct import ConstructFeatures, identity_column_selector
from feature.search import search_all, searchConfig
from WordTokenizer import get_tokenizer


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


def save_activation_diagnostics(captured_results, search_tasks, output_dir):
    for task in search_tasks:
        name = task["name"]
        alpha = task["alpha"]
        min_acts = task["min_acts"]
        activations_base = captured_results[name]["base"]
        activations_finetuned = captured_results[name]["finetuned"]
        task_output_dir = os.path.join(output_dir, name)
        binarized_activations = threshold(activations_finetuned.states, alpha=alpha)

        save_activation_diagnostics_artifacts(
            activations_finetuned.states,
            activations_base.states,
            binarized_activations,
            min_acts,
            task_output_dir,
            alpha=alpha,
        )


def select_tokenizer_training_text(columns: list[str]):
    def selector(example):
        example["text"] = "\n".join(str(example[column]) for column in columns)
        return {"text": example["text"]}

    return selector


if __name__ == "__main__":
    snli = load_dataset("snli", split="validation")
    vitaminc = load_dataset("tals/vitaminc", split="validation[:10_000]")

    model_id = "LiquidAI/LFM2.5-1.2B-Thinking"
    snli_feature_tokenizer = get_tokenizer(
        "spacy-pos-snli-features",
        snli.map(select_tokenizer_training_text(["premise", "hypothesis"]))
    )
    claim_feature_tokenizer = get_tokenizer(
        "spacy-pos-claim-features",
        vitaminc.map(select_tokenizer_training_text(["claim", "evidence"]))
    )
    model_tokenizer = AutoTokenizer.from_pretrained(model_id) # Not used, but for comparison
    snli_features = ConstructFeatures(
        snli,
        snli_feature_tokenizer,
        feature_text_selector=identity_column_selector(["premise", "hypothesis"]),
    )
    print(snli_features[:30])
    claim_features = ConstructFeatures(
        vitaminc,
        claim_feature_tokenizer,
        feature_text_selector=identity_column_selector(["claim", "evidence"]),
    )
    print(claim_features[:30])

    lora_dir = "Heroi/multitune-lora-backup"
    lora_remote = True
    lora_token = os.environ.get("HF_TOKEN")

    output_dir = "results_base"
    search_config = searchConfig(
        formula_length=5,
        pruned_queue_size=10,
        max_iterations=10,
        length_penalty=0.0,
    )
    search_tasks = [
        {
            "name": "snli",
            "alpha": 0.055,
            "min_acts": 500,
            "features": snli_features,
            "class_token_ids": SNLI_CLASS_TOKEN_IDS,
            "labels": SNLI_LABELS,
            "dataset": snli,
            "data_formatter": format_snli_for_capture,
        },
        {
            "name": "claim",
            "alpha": 0.052,
            "min_acts": 500,
            "features": claim_features,
            "class_token_ids": VITAMINC_CLASS_TOKEN_IDS,
            "labels": VITAMINC_LABELS,
            "dataset": vitaminc,
            "data_formatter": format_vitaminc_for_capture,
        },
    ]

    feedforward_layer = "model.layers.14.feed_forward"

    captured_results = Capture(
        model_id,
        lora_dir,
        tasks=[
            CaptureConfig("snli", snli, format_snli_for_capture),
            CaptureConfig("claim", vitaminc, format_vitaminc_for_capture),
        ],
        layer=feedforward_layer,
        lora_remote=lora_remote,
        lora_token=lora_token,
    )
    save_captured_activations(captured_results, output_dir)
    save_activation_diagnostics(captured_results, search_tasks, output_dir)

    for task in search_tasks:
        name = task["name"]
        alpha = task["alpha"]
        min_acts = task["min_acts"]
        features = task["features"]
        activations = captured_results[name]["base"]
        task_output_dir = os.path.join(output_dir, name)
        binarized_activations, kept_activations, kept_neurons = post_process_activations(
            activations,
            alpha,
            min_acts,
        )

        # Searches
        search_results = search_all(
            kept_activations,
            features,
            num_workers=8,
            config=search_config,
        )

        classification_weights = get_classification_weights(
            model_id,
            lora_dir,
            name,
            task["class_token_ids"],
            lora_remote=lora_remote,
            lora_token=lora_token,
        )
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

    print("--- Search complete, entering ablation stage ---")

    for task in search_tasks:
        name = task["name"]
        weight_column_names = tuple( f"weight_{label.replace(' ', '_')}" for label in task["labels"])
        task_output_dir = os.path.join(output_dir, name)
        output_csv_path = os.path.join(output_dir, f"{name}_beam_results.csv")

        # Ablations
        analysis_result = run_ablation_analysis(
            result_csv_path=output_csv_path,
            output_dir=task_output_dir,
            weight_column_names=weight_column_names,
        )
        ablation_task = AblationTaskConfig(
            name=name,
            dataset=task["dataset"],
            data_formatter=task["data_formatter"],
        )
        lora_path = None
        inference_engine = AblationInferenceEngine(
            model_id=model_id,
            task=ablation_task,
            layer=feedforward_layer,
            class_token_ids=task["class_token_ids"],
            lora_path=lora_path,
        )
        try:
            run_ablation(
                analysis_result=analysis_result,
                inference_engine=inference_engine,
                output_dir=task_output_dir,
            )
        finally:
            inference_engine.close()
