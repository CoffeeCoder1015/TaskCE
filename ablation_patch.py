import os
from analysis.ablation_inference import AblationInferenceEngine, AblationTaskConfig
from analysis import run_ablation, run_ablation_analysis
from datasets import load_dataset

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
vitaminc = load_dataset("tals/vitaminc", split="validation[:10_000]")
task = {
            "name": "claim",
            "alpha": 0.052,
            "min_acts": 500,
            "class_token_ids": VITAMINC_CLASS_TOKEN_IDS,
            "labels": VITAMINC_LABELS,
            "dataset": vitaminc,
            "data_formatter": format_vitaminc_for_capture,
        }

output_dir = "results_base"
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

model_id = "LiquidAI/LFM2.5-1.2B-Thinking"
feedforward_layer = "model.layers.14.feed_forward"

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
