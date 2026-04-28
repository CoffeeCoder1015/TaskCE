from collections import Counter

from datasets import load_dataset

from capture import Capture, CaptureConfig


MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"
SNLI_LABELS = ["entailment", "neutral", "contradiction"]

SNLI_SYSTEM_PROMPT = {
    "role": "system",
    "content": """Determine the relationship between the `premise`and `hypothesis` and respond with an answer.
You must respond with an answer of `entailment`, `neutral` or `contradiction`""",
}


def format_snli_for_capture(example):
    # The capture package needs the prompt for inference and the answer for tagging.
    test_example = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    example["prompt"] = [
        SNLI_SYSTEM_PROMPT,
        {"role": "user", "content": test_example},
    ]

    # This is the label that will stay aligned with the returned activation vector.
    example["answer"] = SNLI_LABELS[example["label"]]
    return example


# Keep this tiny for the minimal example. Increase the slice when the capture path is verified.
snli_examples = load_dataset("snli", split="validation[:8]")

capture_results = Capture(
    model_id=MODEL_ID,
    lora_dir=None,
    tasks=[
        CaptureConfig(
            name="snli",
            dataset=snli_examples,
            data_formatter=format_snli_for_capture,
        )
    ],
    # -2 means the second-to-last named module exposed by the model.
    layer=-2,
    batch_size=1,
)

snli_base = capture_results["snli"]["base"]

# Analysis belongs outside capture. This is where PCA or other analysis will consume states.
states = snli_base["states"]
labels = snli_base["labels"]

print("Captured states shape:", tuple(states.shape))
print("Captured label counts:", Counter(labels))
print("Captured layer:", snli_base["layer"])
