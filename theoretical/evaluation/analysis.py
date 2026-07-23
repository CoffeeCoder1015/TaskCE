from collections import Counter
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import re
from typing import Callable

from huggingface_hub import snapshot_download
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
from peft import PeftModel


DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"
DEFAULT_LORA_REPOSITORY = "Heroi/multitune-lora-backup"

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
VITAMINC_LABELS = ["supports", "refutes", "not enough info"]

SNLI_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Determine the relationship between the `premise` and `hypothesis` "
        "and respond with `entailment`, `neutral`, or `contradiction`."
    ),
}
FALLACY_SYSTEM_PROMPT = {
    "role": "system",
    "content": "Identify the logical fallacy in the text and respond with an answer.",
}
VITAMINC_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Determine the relationship between the `claim` and `evidence` and "
        "respond with `supports`, `refutes`, or `not enough info`."
    ),
}


@dataclass(frozen=True)
class EvalConfig:
    name: str
    dataset: object
    data_formatter: Callable
    eval_fn: Callable

def evaluate_predictions(prediction, answer):
    if prediction is None:
        return "reject"
    if prediction == answer:
        return "success"
    return "fail"


def print_results(task_name, stats):
    total = sum(stats.values())
    accuracy = stats["success"] / total if total > 0 else 0

    print(f"\n\033[94m{task_name} results\033[0m")
    print("\033[94mSuccess:\033[0m", stats["success"])
    print("\033[94mFail:\033[0m", stats["fail"])
    print("\033[94mReject:\033[0m", stats["reject"])
    print("\033[94mAccuracy:\033[0m", accuracy)


def eval_task(model_id, tokenizer, task, lora_path=None, batch_size=128):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)

    tg_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    formatted_dataset = task.dataset.map(task.data_formatter)
    prompts = formatted_dataset["prompt"]
    answers = formatted_dataset["answer"]

    responses_raw = []

    print(f"\nStarting {task.name} inference.")
    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"Generating {task.name}",
    ):
        batch = prompts[i:i + batch_size]
        out = tg_pipeline(
            batch,
            batch_size=batch_size,
            return_full_text=False,
        )
        responses_raw.extend(out)
    print(f"{task.name} inference finished.")

    response_chats = [resp[0]["generated_text"] for resp in responses_raw]
    predictions = [task.eval_fn(response_chat) for response_chat in response_chats]
    results = [
        evaluate_predictions(prediction, answer)
        for prediction, answer in zip(predictions, answers)
    ]

    stats = Counter(results)

    del tg_pipeline, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


def latest_task_checkpoints(
    lora_dir,
    *,
    remote=False,
    token=None,
):
    root = Path(
        snapshot_download(
            repo_id=lora_dir,
            repo_type="model",
            token=token,
        )
        if remote
        else lora_dir
    )
    checkpoints = {}
    for task_directory in root.iterdir():
        if not task_directory.is_dir():
            continue
        candidates = [path for path in task_directory.iterdir() if path.is_dir()]
        if not candidates:
            continue

        def sort_key(path):
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            return (int(match.group(1)) if match else -1, path.name)

        checkpoints[task_directory.name] = max(candidates, key=sort_key)
    return checkpoints


def Evaluate(
    model_id: str,
    lora_dir: str,
    tasks: list[EvalConfig],
    *,
    lora_remote: bool = False,
    lora_token: str | bool | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    results = []
    print("[Base model evaluation]")
    for task in tasks:
        stats = eval_task(model_id, tokenizer, task)
        print_results(task.name, stats)
        results.append(
            {
                "task": task.name,
                "variant": "base",
                "success": stats["success"],
                "fail": stats["fail"],
                "reject": stats["reject"],
            }
        )

    print("[Finetuned model evaluation]")
    latest_checkpoints = latest_task_checkpoints(
        lora_dir,
        remote=lora_remote,
        token=lora_token,
    )

    for task in tasks:
        if task.name not in latest_checkpoints:
            print(f"Skipping {task.name}: no LoRA checkpoint found.")
            continue

        stats = eval_task(model_id, tokenizer, task, lora_path=latest_checkpoints[task.name])
        print_results(task.name,stats)
        results.append(
            {
                "task": task.name,
                "variant": "finetuned",
                "success": stats["success"],
                "fail": stats["fail"],
                "reject": stats["reject"],
            }
        )
    return results


def format_snli(example):
    example["prompt"] = [
        SNLI_SYSTEM_PROMPT,
        {
            "role": "user",
            "content": (
                f"Premise: {example['premise']}\n"
                f"Hypothesis: {example['hypothesis']}"
            ),
        },
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
    example["prompt"] = [
        VITAMINC_SYSTEM_PROMPT,
        {
            "role": "user",
            "content": (
                f"Evidence: {example['evidence']}\nClaim: {example['claim']}"
            ),
        },
    ]
    example["answer"] = str(example["label"]).strip().lower()
    return example


def extract_first_class(content, classes):
    content = content.lower()
    classifications = {label: content.find(label) for label in classes}
    return min(
        (
            item
            for item in classifications.items()
            if item[1] >= 0
        ),
        key=lambda item: item[1],
        default=(None, None),
    )[0]


def extract_snli(content):
    return extract_first_class(content, SNLI_LABELS)


def extract_fallacy(content):
    return extract_first_class(content, FALLACY_LABELS)


def extract_vitaminc(content):
    return extract_first_class(content, VITAMINC_LABELS)


def _load_dataset(name, *, split):
    from datasets import load_dataset

    return load_dataset(name, split=split)


def run(
    *,
    output_path,
    model_id=DEFAULT_MODEL_ID,
    lora_repository=DEFAULT_LORA_REPOSITORY,
    lora_token=None,
):
    tasks = [
        EvalConfig(
            "snli",
            _load_dataset("snli", split="validation"),
            format_snli,
            extract_snli,
        ),
        EvalConfig(
            "fallacy",
            _load_dataset("tasksource/logical-fallacy", split="dev"),
            format_fallacy,
            extract_fallacy,
        ),
        EvalConfig(
            "claim",
            _load_dataset("tals/vitaminc", split="validation[:10_000]"),
            format_vitaminc,
            extract_vitaminc,
        ),
    ]
    results = Evaluate(
        model_id=model_id,
        lora_dir=lora_repository,
        lora_remote=True,
        lora_token=(
            os.environ.get("HF_TOKEN")
            if lora_token is None
            else lora_token
        ),
        tasks=tasks,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    return output_path
