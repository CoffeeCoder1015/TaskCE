import argparse
import gc
import os
from dataclasses import dataclass
from typing import Callable

from datasets import load_dataset
from feature.search import searchConfig, search_all
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis import (
    build_neuron_search_results_dataframe,
    save_neuron_search_results_csv,
    run_ablation,
    run_ablation_analysis,
)
from analysis.ablation_inference import AblationInferenceEngine
from evaluation import EvalConfig
from feature.construct import ConstructFeatures
from capture import Capture, CaptureConfig
from capture.postprocessing import prune_min_acts, threshold

MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"
LORA_DIR = "../multitune/output"
RESULT_DIR = "results"
ACTIVATION_ALPHA = 0.055
MIN_ACTS = 500
SEARCH_NUM_WORKERS = 10
SEARCH_CONFIG = searchConfig(
    formula_length=5,
    pruned_queue_size=10,
    max_iterations=10,
    length_penalty=0.0,
)

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

# Backward-compatible aliases for existing imports/tests.
CLASS_NAMES = list(SNLI_LABELS)
CLASS_TOKEN_IDS = SNLI_CLASS_TOKEN_IDS
BEAM_RESULTS_CSV = os.path.join(RESULT_DIR, "snli_beam_results.csv")


@dataclass(frozen=True)
class AnalysisTarget:
    name: str
    dataset_name: str
    split: str
    labels: tuple[str, ...]
    class_token_ids: dict[str, int]
    feature_text_selector: Callable
    data_formatter: Callable
    model_task_name: str | None = None

    @property
    def beam_results_csv(self):
        return os.path.join(RESULT_DIR, f"{self.name}_beam_results.csv")

    @property
    def capture_name(self):
        return self.model_task_name or self.name

    @property
    def output_dir(self):
        return os.path.join(RESULT_DIR, self.name)

    @property
    def weight_column_names(self):
        return tuple(f"weight_{label.replace(' ', '_')}" for label in self.labels)


def select_snli_feature_text(example):
    return {"premise": example["premise"], "hypothesis": example["hypothesis"]}


def format_snli_for_capture(example):
    test_example = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    example["prompt"] = [
        SNLI_SYSTEM_PROMPT,
        {"role": "user", "content": test_example},
    ]
    example["answer"] = SNLI_LABELS[example["label"]]
    return example


def select_vitaminc_feature_text(example):
    return {"evidence": example["evidence"], "claim": example["claim"]}


def format_vitaminc_for_capture(example):
    test_example = f"Evidence: {example['evidence']}\nClaim: {example['claim']}"
    example["prompt"] = [
        VITAMINC_SYSTEM_PROMPT,
        {"role": "user", "content": test_example},
    ]
    example["answer"] = str(example["label"]).strip().lower()
    return example


ANALYSIS_TARGETS = {
    "snli": AnalysisTarget(
        name="snli",
        dataset_name="snli",
        split="validation",
        labels=SNLI_LABELS,
        class_token_ids=SNLI_CLASS_TOKEN_IDS,
        feature_text_selector=select_snli_feature_text,
        data_formatter=format_snli_for_capture,
    ),
    "claim": AnalysisTarget(
        name="claim",
        dataset_name="tals/vitaminc",
        split="validation[:10_000]",
        labels=VITAMINC_LABELS,
        class_token_ids=VITAMINC_CLASS_TOKEN_IDS,
        feature_text_selector=select_vitaminc_feature_text,
        data_formatter=format_vitaminc_for_capture,
    ),
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


def get_classification_weights(model, class_token_ids=None):
    class_token_ids = class_token_ids or CLASS_TOKEN_IDS
    class_ids = list(class_token_ids.values())
    lm_head_weight = model.lm_head.weight.detach().cpu().to(torch.float32)
    weights = lm_head_weight[class_ids].T
    print(f"Classification weights shape: {tuple(weights.shape)}")
    return weights


def extract_first_class(content, classes):
    content = content.lower()
    classifications = {label: content.find(label) for label in classes}

    return min(
        filter(lambda x: x[1] >= 0, classifications.items()),
        key=lambda kv: kv[1],
        default=(None, None),
    )[0]


def log_stage_task(stage, target):
    print( f"[{stage}] is operating on task {target.name}" )

def selected_targets(target_name):
    if target_name == "all":
        return list(ANALYSIS_TARGETS.values())
    return [ANALYSIS_TARGETS[target_name]]

def load_target_datasets(targets):
    datasets = {}
    for target in targets:
        log_stage_task("dataset loading", target)
        datasets[target.name] = load_dataset(
            target.dataset_name,
            split=target.split,
            trust_remote_code=True,
        )
    return datasets


def build_feature_vectors_for_target(target, dataset, tokenizer):
    log_stage_task("feature construction", target)
    return ConstructFeatures(
        dataset,
        tokenizer,
        feature_text_selector=target.feature_text_selector,
    )


def build_feature_vectors_by_target(targets, datasets, tokenizer):
    return {
        target.name: build_feature_vectors_for_target(
            target,
            datasets[target.name],
            tokenizer,
        )
        for target in targets
    }


def build_capture_configs(targets, datasets):
    return [
        CaptureConfig(
            name=target.capture_name,
            dataset=datasets[target.name],
            data_formatter=target.data_formatter,
        )
        for target in targets
    ]


def capture_analysis_targets(targets, datasets):
    for target in targets:
        log_stage_task("capture", target)
    return Capture(
        model_id=MODEL_ID,
        lora_dir=LORA_DIR,
        tasks=build_capture_configs(targets, datasets),
        layer=-2,
        batch_size=256,
    )


def load_classification_weights_for_target(target):
    log_stage_task("classification weight loading", target)
    lora_path = resolve_latest_lora_checkpoint(LORA_DIR, target.capture_name)
    lm_head_model = load_lm_head_model(MODEL_ID, lora_path=lora_path)
    try:
        classification_weights = get_classification_weights(
            lm_head_model,
            class_token_ids=target.class_token_ids,
        )
    finally:
        del lm_head_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return classification_weights, lora_path


def save_target_search_results(target, search_results, fine_binary_acts, fine_neuron_ids):
    log_stage_task("search result saving", target)
    classification_weights, lora_path = load_classification_weights_for_target(target)
    beam_df = build_neuron_search_results_dataframe(
        search_results=search_results,
        kept_neuron_ids=fine_neuron_ids,
        total_neuron_count=fine_binary_acts.shape[1],
        classification_weights=classification_weights,
        weight_column_names=target.weight_column_names,
    )
    save_neuron_search_results_csv(
        dataframe=beam_df,
        output_csv_path=target.beam_results_csv,
    )
    print(f"Saved search result dataframe: {target.beam_results_csv}")
    return lora_path


def run_target_ablation(target, dataset, lora_path):
    log_stage_task("ablation analysis", target)
    print("Running ablation analysis...")
    analysis_result = run_ablation_analysis(
        result_csv_path=target.beam_results_csv,
        output_dir=target.output_dir,
        weight_column_names=target.weight_column_names,
    )

    task = EvalConfig(
        name=target.capture_name,
        dataset=dataset,
        data_formatter=target.data_formatter,
        eval_fn=lambda content: extract_first_class(content, target.labels),
    )

    log_stage_task("ablation inference engine", target)
    print("Building ablation inference engine...")
    inference_engine = AblationInferenceEngine(
        model_id=MODEL_ID,
        task=task,
        layer=-2,
        class_token_ids=target.class_token_ids,
        lora_path=lora_path,
    )

    log_stage_task("ablation run", target)
    print("Running ablation...")
    try:
        run_ablation(
            analysis_result=analysis_result,
            inference_engine=inference_engine,
            output_dir=target.output_dir,
        )
    finally:
        inference_engine.close()


def run_search_and_ablation_for_target(
    target,
    dataset,
    feature_vectors,
    capture_results,
):
    os.makedirs(target.output_dir, exist_ok=True)

    activations_base = capture_results[target.capture_name]["base"]
    activations_fine = capture_results[target.capture_name]["finetuned"]
    print(activations_base)
    print(activations_fine)

    base_binary_acts = threshold(activations_base.states, alpha=ACTIVATION_ALPHA)
    _, _ = prune_min_acts(base_binary_acts, min_acts=MIN_ACTS)
    fine_binary_acts = threshold(activations_fine.states, alpha=ACTIVATION_ALPHA)
    fine_pruned_acts, fine_neuron_ids = prune_min_acts(fine_binary_acts, min_acts=MIN_ACTS)

    print(f"Analysis target: {target.name}")
    print(f"Activation alpha: {ACTIVATION_ALPHA}")
    print(f"Minimum activations: {MIN_ACTS}")
    print(f"Search workers: {SEARCH_NUM_WORKERS}")
    print(f"Search config: {SEARCH_CONFIG}")
    log_stage_task("feature search", target)
    search_results = search_all(
        fine_pruned_acts,
        feature_vectors,
        num_workers=SEARCH_NUM_WORKERS,
        config=SEARCH_CONFIG,
    )
    lora_path = save_target_search_results(
        target=target,
        search_results=search_results,
        fine_binary_acts=fine_binary_acts,
        fine_neuron_ids=fine_neuron_ids,
    )
    run_target_ablation(target, dataset, lora_path)


def run_analysis_targets(targets):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    datasets = load_target_datasets(targets)
    feature_vectors_by_target = build_feature_vectors_by_target(
        targets,
        datasets,
        tokenizer,
    )
    capture_results = capture_analysis_targets(targets, datasets)

    for target in targets:
        run_search_and_ablation_for_target(
            target=target,
            dataset=datasets[target.name],
            feature_vectors=feature_vectors_by_target[target.name],
            capture_results=capture_results,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run neuron search and ablation analysis.")
    parser.add_argument(
        "--target",
        choices=[*ANALYSIS_TARGETS.keys(), "all"],
        default="snli",
        help="Dataset analysis target to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_analysis_targets(selected_targets(args.target))


if __name__ == "__main__":
    main()
