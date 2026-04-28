from collections import Counter

from datasets import load_dataset

from capture import Capture, CaptureConfig


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


snli_examples = load_dataset("snli", split="validation")

capture_results = Capture(
    model_id="LiquidAI/LFM2.5-1.2B-Thinking",
    lora_dir="../multitune/output", # Assumed to be used in conjunction with https://github.com/CoffeeCoder1015/multitune
    tasks=[
        CaptureConfig(
            name="snli",
            dataset=snli_examples,
            data_formatter=format_snli_for_capture,
        )
    ],
    layer=-2,
    batch_size=512,
)

snli_base = capture_results["snli"]["base"]

# Analysis belongs outside capture. This is where PCA or other analysis will consume states.
states = snli_base["states"]
labels = snli_base["labels"]

print("Captured states shape:", tuple(states.shape))
print("Captured label counts:", Counter(labels))
print("Captured layer:", snli_base["layer"])

