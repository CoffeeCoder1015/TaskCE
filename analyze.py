import gc
import os

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
BEAM_RESULTS_CSV = os.path.join(RESULT_DIR, "snli_beam_results.csv")
ACTIVATION_ALPHA = 0.055
MIN_ACTS = 500
SEARCH_NUM_WORKERS = 10
SEARCH_CONFIG = searchConfig(
    formula_length=5,
    pruned_queue_size=10,
    max_iterations=10,
    length_penalty=0.0,
)

CLASS_TOKEN_IDS = {
    "entailment": 806,
    "neutral": 25919,
    "contradiction": 10913,
}
CLASS_NAMES = ["entailment", "neutral", "contradiction"]


def select_snli_feature_text(example):
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


if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)
    dataset = load_dataset("snli", split="validation", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    feature_vectors = ConstructFeatures(
        dataset,
        tokenizer,
        feature_text_selector=select_snli_feature_text,
    )

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

    base_binary_acts = threshold(activations_base.states, alpha=ACTIVATION_ALPHA)
    base_pruned_acts, base_neuron_ids = prune_min_acts(base_binary_acts, min_acts=MIN_ACTS)
    fine_binary_acts = threshold(activations_fine.states, alpha=ACTIVATION_ALPHA)
    fine_pruned_acts, fine_neuron_ids = prune_min_acts(fine_binary_acts, min_acts=MIN_ACTS)

    print(f"Activation alpha: {ACTIVATION_ALPHA}")
    print(f"Minimum activations: {MIN_ACTS}")
    print(f"Search workers: {SEARCH_NUM_WORKERS}")
    print(f"Search config: {SEARCH_CONFIG}")
    search_results = search_all(
        fine_pruned_acts,
        feature_vectors,
        num_workers=SEARCH_NUM_WORKERS,
        config=SEARCH_CONFIG,
    )
    lora_path = resolve_latest_lora_checkpoint(LORA_DIR, "snli")
    lm_head_model = load_lm_head_model(MODEL_ID, lora_path=lora_path)
    try:
        classification_weights = get_classification_weights(lm_head_model)
    finally:
        del lm_head_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    beam_df = build_neuron_search_results_dataframe(
        search_results=search_results,
        kept_neuron_ids=fine_neuron_ids,
        total_neuron_count=fine_binary_acts.shape[1],
        classification_weights=classification_weights,
    )
    save_neuron_search_results_csv(
        dataframe=beam_df,
        output_csv_path=BEAM_RESULTS_CSV,
    )

    print(f"Saved search result dataframe: {BEAM_RESULTS_CSV}")

    # Hook up the ablation pipeline
    print("Running ablation analysis...")
    analysis_result = run_ablation_analysis(
        result_csv_path=BEAM_RESULTS_CSV,
        output_dir=RESULT_DIR,
    )

    def extract_first_class(content, classes):
        content = content.lower()
        classifications = {label: content.find(label) for label in classes}

        return min(
            filter(lambda x: x[1] >= 0, classifications.items()),
            key=lambda kv: kv[1],
            default=(None, None),
        )[0]

    def extract_snli(content):
        return extract_first_class(content, SNLI_LABELS)

    snli_task = EvalConfig(
        name="snli",
        dataset=dataset,
        data_formatter=format_snli_for_capture,
        eval_fn=extract_snli,
    )

    print("Building ablation inference engine...")
    inference_engine = AblationInferenceEngine(
        model_id=MODEL_ID,
        task=snli_task,
        layer=-2,
        class_token_ids=CLASS_TOKEN_IDS,
        lora_path=lora_path,
    )

    print("Running ablation...")
    try:
        run_ablation(
            analysis_result=analysis_result,
            inference_engine=inference_engine,
            output_dir=RESULT_DIR,
        )
    finally:
        inference_engine.close()
