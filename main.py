import os
from datasets import load_dataset
import random

# Assumed to be used in conjunction with https://github.com/CoffeeCoder1015/multitune
lora_dir = "../multitune/output"

tasks = os.listdir(lora_dir)

snli_val = load_dataset("snli", split="validation")
logic_fallacy_val = load_dataset("tasksource/logical-fallacy",split="dev")

FALLACY_PROMPT_VARIATIONS = [
    lambda p : f"What logical fallacy is: {p}",
    lambda p : f"Examine \"{p}\" and identify what logical fallacy is it. ",
    lambda p : f"{p}, the logical fallacy is:",
    lambda p : f"Analyze the following statement for logical inconsistencies: \"{p}\". Provide the name of the fallacy and a brief justification.",
    lambda p : f"Instruction: Categorize the logical error in the text below.\nText: {p}\nFallacy Category:"
]

def fallacy_formatter(example):
    prompt_fn = random.choice(FALLACY_PROMPT_VARIATIONS)
    base_prompt = example["source_article"]
    answer = example["logical_fallacies"]

    prompt = [ {"role":"user","content":prompt_fn(base_prompt)} ]
    completion = [ {"role":"assistant","content":answer} ]
    return {
        "prompt":prompt,
        "completion":completion
    }

NLI_PROMPT_VARIATIONS = [
    lambda p, h: f"Is the hypothesis entailed, neutral, or contradictory to the premise? Premise: {p} Hypothesis: {h}",
    lambda p, h: f"What is the relationship between premise and hypothesis? Premise: {p} Hypothesis: {h}",
    lambda p, h: f"Inference the relationship between the Premise: {p} and Hypothesis: {h}",
    lambda p, h: f"Classify as entailment, neutral, or contradiction.\nPremise: {p}\nHypothesis: {h}",
    lambda p, h: f"Premise: {p}\nHypothesis: {h}\nRelationship:",
]

CLASSIFICATION_MAP = ["entailment", "neutral", "contradiction"]

def snli_formatter(example):
    prompt_fn = random.choice(NLI_PROMPT_VARIATIONS)
    prompt = [ {"role":"user","content":prompt_fn(example["premise"], example["hypothesis"])} ]

    label = CLASSIFICATION_MAP[example["label"]]
    completion = [ {"role":"assistant", "content":label}]

    example["prompt"] = prompt
    example["completion"] = completion
    return example
