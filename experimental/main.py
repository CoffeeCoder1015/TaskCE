"""Run SNLI capture and save final-token matrices separately from capture."""

from pathlib import Path

from peft import PeftModel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.tree import Tree
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_lora import get_lora_checkpoints
from prepare_snli import get_prepared_snli

import torch

from capture import capture


MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"
LORA_REPO = "Heroi/multitune-lora-backup"
DATASET_SPLIT = "validation"
LAYERS = ["model.layers.8.feed_forward", "model.layers.14.feed_forward"]
DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "left"
dataset = get_prepared_snli(tokenizer, split=DATASET_SPLIT)
snli_checkpoint = get_lora_checkpoints(LORA_REPO)["snli"]

architecture = Tree(Text(f"Model architecture: {MODEL_ID}", style="bold"), guide_style="dim")

with Progress(
    TextColumn("[cyan]{task.description}"),
    TextColumn("{task.fields[stage]}"),
    BarColumn(bar_width=None),
    TaskProgressColumn(),
    MofNCompleteColumn(),
    TextColumn("examples"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
) as progress:
    for checkpoint in (None, snli_checkpoint):
        variant = Path("base") if checkpoint is None else Path("snli") / checkpoint.name
        task = progress.add_task(
            variant.as_posix(), total=len(dataset), start=False, stage="[yellow]loading"
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        if checkpoint is not None:
            model = PeftModel.from_pretrained(model, str(checkpoint))

        progress.start_task(task)
        progress.update(task, stage="inference")
        with capture(
            model,
            dataset,
            tokenizer=tokenizer,
            layers=LAYERS,
            batch_size=32,
            on_progress=lambda completed: progress.update(task, completed=completed),
        ) as result:
            progress.update(task, stage="[yellow]saving")
            captured_layers = {layer for layer, _ in result}
            for layer in captured_layers:
                activations = torch.stack([
                    result[layer, example][0][-1] for example in range(len(dataset))
                ]).float()
                path = DATA_DIRECTORY / MODEL_ID / variant / layer / "snli_activation.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(activations, path)

        progress.update(task, stage="[green]saved")
        nodes = {"": architecture.add(Text(variant.as_posix(), style="cyan"))}
        for module_path in dict(model.named_modules()):
            if module_path:
                parent, _, name = module_path.rpartition(".")
                nodes[module_path] = nodes[parent].add(
                    Text(name, style="green" if module_path in captured_layers else "")
                )

        del result, model

progress.console.print(architecture)
