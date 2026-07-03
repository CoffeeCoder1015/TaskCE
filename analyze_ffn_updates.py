import os

import pandas as pd
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from capture import CaptureConfig, save_captured_activations
from capture.capturer import capture_task_activations
from capture.classification_weights import latest_task_lora_checkpoint
from capture.postprocessing import threshold, prune_min_acts

from feature.construct import ConstructFeatures, identity_column_selector
from feature.search import search_all, searchConfig
from WordTokenizer import get_tokenizer


SNLI_LABELS = ("entailment", "neutral", "contradiction")
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


def select_tokenizer_training_text(columns: list[str]):
    def selector(example):
        example["text"] = "\n".join(str(example[column]) for column in columns)
        return {"text": example["text"]}

    return selector


if __name__ == "__main__":
    snli = load_dataset("snli", split="validation")
    vitaminc = load_dataset("tals/vitaminc", split="validation[:10_000]")
    max_examples = None
    if max_examples is not None:
        snli = snli.select(range(min(max_examples, len(snli))))
        vitaminc = vitaminc.select(range(min(max_examples, len(vitaminc))))

    snli_feature_tokenizer = get_tokenizer(
        "spacy-pos-snli-features",
        snli.map(select_tokenizer_training_text(["premise", "hypothesis"]))
    )
    claim_feature_tokenizer = get_tokenizer(
        "spacy-pos-claim-features",
        vitaminc.map(select_tokenizer_training_text(["claim", "evidence"]))
    )

    model_id = "LiquidAI/LFM2.5-1.2B-Thinking"
    model_config = AutoConfig.from_pretrained(model_id)
    model_tokenizer = AutoTokenizer.from_pretrained(model_id)
    if model_tokenizer.pad_token is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token
    model_tokenizer.padding_side = "left"

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
            "dataset": snli,
            "capture_config": CaptureConfig("snli", snli, format_snli_for_capture),
            "batch_size": 256,
        },
        {
            "name": "claim",
            "alpha": 0.052,
            "min_acts": 500,
            "features": claim_features,
            "dataset": vitaminc,
            "capture_config": CaptureConfig("claim", vitaminc, format_vitaminc_for_capture),
            "batch_size": 64,
        },
    ]

    lora_dir = "../multitune/output"
    model_variants = {
        "base": None,
        "snli_finetuned": latest_task_lora_checkpoint(lora_dir, "snli"),
        "claim_finetuned": latest_task_lora_checkpoint(lora_dir, "claim"),
    }

    ffn_update_layer = "model.layers.8.feed_forward"
    output_dir = "results/ffn_update_formula_comparison"
    run_full_experiment = False
    print("FFN update formula comparison")
    print(f"model_id: {model_id}")
    print(f"layer: {ffn_update_layer}")
    print(f"output_dir: {output_dir}")
    print(f"run_full_experiment: {run_full_experiment}")
    for variant_name, lora_path in model_variants.items():
        print(f"{variant_name}: {lora_path}")
    if not run_full_experiment:
        raise SystemExit("Set run_full_experiment = True in this file to capture/search.")

    captured_results = {}
    for task in search_tasks:
        name = task["name"]
        captured_results[name] = {}
        for variant_name, lora_path in model_variants.items():
            print(
                f"[Capture] task={name} variant={variant_name} "
                f"layer={ffn_update_layer} lora={lora_path}"
            )
            captured_results[name][variant_name] = capture_task_activations(
                model_id=model_id,
                tokenizer=model_tokenizer,
                task=task["capture_config"],
                layer=ffn_update_layer,
                lora_path=lora_path,
                batch_size=task["batch_size"],
            )
            expected_shape = (len(task["dataset"]), model_config.hidden_size)
            actual_shape = tuple(captured_results[name][variant_name].states.shape)
            shape_status = "match" if actual_shape == expected_shape else "MISMATCH"
            print(
                f"[Shape] task={name} variant={variant_name} "
                f"expected={expected_shape} actual={actual_shape} {shape_status}"
            )
            if actual_shape != expected_shape:
                raise ValueError(
                    "Captured FFN update shape mismatch: "
                    f"task={name}, variant={variant_name}, "
                    f"expected={expected_shape}, actual={actual_shape}"
                )

    save_captured_activations(captured_results, output_dir)

    for task in search_tasks:
        name = task["name"]
        alpha = task["alpha"]
        min_acts = task["min_acts"]
        features = task["features"]
        task_output_dir = os.path.join(output_dir, name)
        os.makedirs(task_output_dir, exist_ok=True)

        formula_tables = {}
        for variant_name in model_variants:
            activations = captured_results[name][variant_name]
            binarized_activations = threshold(activations.states, alpha=alpha)
            kept_activations, kept_neurons = prune_min_acts(
                binarized_activations,
                min_acts=min_acts,
            )

            # Searches
            search_results = search_all(
                kept_activations,
                features,
                num_workers=8,
                config=search_config,
            )

            kept_neuron_ids = kept_neurons.detach().cpu().tolist()
            kept_neuron_id_set = set(kept_neuron_ids)
            rows = []
            for result in search_results:
                activation_id = int(kept_neuron_ids[result.activation_index])
                rows.append(
                    {
                        "activation": activation_id,
                        "formula": result.best_formula,
                        "iou": float(result.best_score),
                    }
                )
            for activation_id in range(int(binarized_activations.shape[1])):
                if activation_id in kept_neuron_id_set:
                    continue
                rows.append(
                    {
                        "activation": activation_id,
                        "formula": "LOW_ACTS_PRUNED",
                        "iou": 0.0,
                    }
                )

            dataframe = pd.DataFrame(rows).sort_values("activation", ignore_index=True)
            formula_tables[variant_name] = dataframe
            output_csv_path = os.path.join(
                task_output_dir,
                f"{variant_name}_ffn_update_formulas.csv",
            )
            dataframe.to_csv(output_csv_path, index=False)

        comparison = None
        for variant_name, dataframe in formula_tables.items():
            renamed = dataframe.rename(
                columns={
                    "formula": f"{variant_name}_formula",
                    "iou": f"{variant_name}_iou",
                }
            )
            comparison = (
                renamed
                if comparison is None
                else comparison.merge(renamed, on="activation", how="outer")
            )

        comparison["snli_matches_base"] = (
            comparison["snli_finetuned_formula"] == comparison["base_formula"]
        )
        comparison["claim_matches_base"] = (
            comparison["claim_finetuned_formula"] == comparison["base_formula"]
        )
        comparison["snli_matches_claim"] = (
            comparison["snli_finetuned_formula"] == comparison["claim_finetuned_formula"]
        )
        comparison_csv_path = os.path.join(
            task_output_dir,
            "ffn_update_formula_comparison.csv",
        )
        comparison.to_csv(comparison_csv_path, index=False)

    print("--- FFN update formula comparison complete ---")
