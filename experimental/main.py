"""Run SNLI capture and save final-token matrices separately from capture."""

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_lora import get_lora_checkpoints
from prepare_snli import get_prepared_snli

import torch

from capture import capture


MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"
LORA_REPO = "Heroi/multitune-lora-backup"
DATASET_SPLIT = "validation[:128]"
LAYERS = ["model.layers.8.feed_forward", "model.layers.14.feed_forward"]
DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "left"
dataset = get_prepared_snli(tokenizer, split=DATASET_SPLIT)
snli_checkpoint = get_lora_checkpoints(LORA_REPO)["snli"]

for checkpoint in (None, snli_checkpoint):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    variant = Path("base")
    if checkpoint is not None:
        model = PeftModel.from_pretrained(model, str(checkpoint))
        variant = Path("snli") / checkpoint.name

    with capture(
        model,
        dataset,
        tokenizer=tokenizer,
        layers=LAYERS,
        batch_size=32,
    ) as result:
        for layer in {layer for layer, _ in result}:
            activations = torch.stack([
                result[layer, example][0][-1] for example in range(len(dataset))
            ]).float()
            path = DATA_DIRECTORY / MODEL_ID / variant / layer / "snli_activation.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(activations, path)

    del result, model
