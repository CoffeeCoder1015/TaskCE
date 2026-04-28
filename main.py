import os
from datasets import load_dataset
import random

from evaluation import Evaluate
from evaluation.evalConfig import EvalConfig

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

system_prompt = {
    "role": "system",
    "content": """Determine the relationship between the `Premise`and `Hypothesis` and respond with an answer. 
You must respond with an answer of `Entailment`, `Neutral` or `Contradiction`
You need to respond in the format shown in the following by chosing one of thosw answers:
<my_answer>[place your answer here]</my_answer>
You are lay out the steps to your final answer before responding with your final answer, but you must respond in this format or else your answer will be rejected."""
}

CLASSIFICATION_MAP = ["entailment", "neutral", "contradiction"]

def snli_formatter(example):
    test_example = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    prompt = [system_prompt, {"role": "user", "content": test_example}]

    label = CLASSIFICATION_MAP[example["label"]]
    completion = [{"role": "assistant", "content": f"<my_answer>{label}</my_answer>"}]

    example["prompt"] = prompt
    example["completion"] = completion
    return example


def extract_first_snli(response_chat,answer):
    assistant_response = response_chat[2]
    assert assistant_response["role"] == "assistant"

    content = assistant_response.get("content", "").lower()

    print(f"\n\033[92mAssistant's Response:\033[0m {content}")

    classifications = ["entailment", "neutral", "contradiction"]
    classifications = {k: content.find(k) for k in classifications}

    return min(
        filter(lambda x: x[1] >= 0, classifications.items()),
        key=lambda kv: kv[1],
        default=(None, None)
    )[0]


def extract_first_fallacy(response_chat,answer):
    pass

Evaluate(
    model="LiquidAI/LFM2.5-1.2B-Thinking",
    tasks=[
        EvalConfig(
            name="snli",
            dataset=snli_val,
            data_formatter=snli_formatter,
            eval_fn=extract_first_snli,
        ),
        EvalConfig(
            name="fallacy",
            dataset=logic_fallacy_val,
            data_formatter=fallacy_formatter,
            eval_fn=extract_first_fallacy,
        ),
    ]
)
