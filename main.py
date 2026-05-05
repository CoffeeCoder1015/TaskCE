from datasets import load_dataset
from evaluation import Evaluate, EvalConfig

snli_val = load_dataset("snli", split="validation")
logic_fallacy_val = load_dataset("tasksource/logical-fallacy",split="dev")
vitaminc_val = load_dataset("tals/vitaminc",split="validation[:10_000]")

SNLI_LABELS = ["entailment", "neutral", "contradiction"]
FALLACY_LABELS = [
    "appeal to emotion",
    "false causality",
    "ad populum",
    "circular reasoning",
    "fallacy of relevance",
    "faulty generalization",
    "ad hominem",
    "fallacy of extension",
    "equivocation",
    "fallacy of logic",
    "fallacy of credibility",
    "intentional",
    "false dilemma",
]
VITAMINC_LABELS=[
    "supports",
    "refutes",
    "not enough info"
]

SNLI_SYSTEM_PROMPT = {
    "role": "system",
    "content": """Determine the relationship between the `premise`and `hypothesis` and respond with an answer.
You must respond with an answer of `entailment`, `neutral` or `contradiction`"""
}

FALLACY_SYSTEM_PROMPT = {
    "role": "system",
    "content": "Identify the logical fallacy in the text and respond with an answer."
}

VITAMINC_SYSTEM_PROMPT = {
    "role": "system",
    "content": """Determine the relationship between the `claim`and `evidence` and respond with an answer.
You must respond with an answer of `supports`, `refutes` or `not enough info`"""
}

def format_snli(example):
    test_example = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    example["prompt"] = [
        SNLI_SYSTEM_PROMPT,
        {"role": "user", "content": test_example},
    ]
    example["answer"] = SNLI_LABELS[example["label"]]
    return example


def format_fallacy(example):
    example["prompt"] = [
        FALLACY_SYSTEM_PROMPT,
        {"role": "user", "content": f"Text: {example['source_article']}"},
    ]
    example["answer"] = str(example["logical_fallacies"]).strip().lower()
    return example

def format_vitaminc(example):
    test_example = f"Evidence: {example["evidence"]}\nClaim: {example["claim"]}"
    example["prompt"] = [
        VITAMINC_SYSTEM_PROMPT,
        {"role": "user", "content": test_example},
    ]
    example["answer"] = str( example["label"] ).lower()
    return example


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


def extract_fallacy(content):
    return extract_first_class(content, FALLACY_LABELS)

def extract_vitaminc(content):
    return extract_first_class(content, VITAMINC_LABELS)


Evaluate(
    model_id="LiquidAI/LFM2.5-1.2B-Thinking",
    lora_dir="../multitune/output", # Assumed to be used in conjunction with https://github.com/CoffeeCoder1015/multitune
    tasks=[
        EvalConfig(
            name="snli",
            dataset=snli_val,
            data_formatter=format_snli,
            eval_fn=extract_snli,
        ),
        EvalConfig(
            name="fallacy",
            dataset=logic_fallacy_val,
            data_formatter=format_fallacy,
            eval_fn=extract_fallacy,
        ), 
        EvalConfig(
            name="claim",
            dataset=vitaminc_val,
            data_formatter=format_vitaminc,
            eval_fn=extract_vitaminc
        )
    ]
)