"""Generate final-token activations from base and task-specific LoRA models."""

import gc
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from experimental.generators.activations import generate_final_token_activations
from experimental.lora_checkpoints import latest_task_lora_checkpoints
from experimental.model import CaptureIdentity, WrappedModel


def format_snli(example):
    example["prompt"] = [
        {
            "role": "system",
            "content": (
                "Determine the relationship between the `premise` and "
                "`hypothesis` and respond with `entailment`, `neutral`, "
                "or `contradiction`."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Premise: {example['premise']}\n"
                f"Hypothesis: {example['hypothesis']}"
            ),
        },
    ]
    return example


def format_vitaminc(example):
    example["prompt"] = [
        {
            "role": "system",
            "content": (
                "Determine the relationship between the `claim` and "
                "`evidence` and respond with `supports`, `refutes`, or "
                "`not enough info`."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Evidence: {example['evidence']}\nClaim: {example['claim']}"
            ),
        },
    ]
    return example


def capture_identity(
    model_id: str,
    dataset: str,
    lora_path: str | None,
) -> CaptureIdentity:
    prefix = tuple(model_id.split("/"))
    if lora_path is None:
        prefix += ("base",)
    else:
        checkpoint = Path(lora_path)
        prefix += (checkpoint.parent.name or dataset, checkpoint.name)
    return CaptureIdentity(prefix=prefix, dataset=dataset)


model_id = "LiquidAI/LFM2.5-1.2B-Thinking"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "left"

checkpoints = latest_task_lora_checkpoints(
    "Heroi/multitune-lora-backup",
    remote=True,
    token=os.environ.get("HF_TOKEN"),
)

snli = load_dataset("snli", split="validation")
snli = snli.map(format_snli)

vitaminc = load_dataset(
    "tals/vitaminc",
    split="validation[:10_000]",
)
vitaminc = vitaminc.map(format_vitaminc)

for dataset_name, dataset in (
    ("snli", snli),
    ("claim", vitaminc),
):
    dataset = dataset.map(
        lambda example: {
            "input_ids": tokenizer.apply_chat_template(
                example["prompt"],
                add_generation_prompt=True,
                tokenize=True,
            )
        }
    )

    variants = [None]
    if dataset_name in checkpoints:
        variants.append(checkpoints[dataset_name])

    for checkpoint in variants:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        if checkpoint is not None:
            model = PeftModel.from_pretrained(model, checkpoint)

        model.eval()
        wrapped = WrappedModel(model)

        try:
            generate_final_token_activations(
                model=wrapped,
                tokenizer=tokenizer,
                data={"input_ids": dataset["input_ids"]},
                layers=("model.layers.8.feed_forward",),
                identity=capture_identity(
                    model_id=model_id,
                    dataset=dataset_name,
                    lora_path=checkpoint,
                ),
                batch_size=32,
            )
        finally:
            wrapped.remove_hooks()
            del wrapped
            del model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
