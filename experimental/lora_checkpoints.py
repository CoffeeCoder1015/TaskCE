"""Locate and apply LoRA checkpoints outside the model wrapper."""

import re
from pathlib import Path

from huggingface_hub import snapshot_download
from peft import PeftModel


def load_lora(model, checkpoint):
    """Apply one caller-selected LoRA checkpoint to an existing model."""
    return PeftModel.from_pretrained(model, str(checkpoint))


def resolve_lora_root(
    lora_dir: str,
    *,
    remote: bool = False,
    token: str | bool | None = None,
) -> Path:
    if remote:
        lora_dir = snapshot_download(
            repo_id=lora_dir,
            repo_type="model",
            token=token,
        )
    return Path(lora_dir)


def latest_checkpoint(task_dir: Path) -> Path | None:
    if not task_dir.is_dir():
        return None
    checkpoints = [path for path in task_dir.iterdir() if path.is_dir()]
    if not checkpoints:
        return None

    def sort_key(path: Path):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match:
            return 1, int(match.group(1)), path.name
        return 0, 0, path.name

    return max(checkpoints, key=sort_key)


def latest_task_lora_checkpoint(
    lora_dir: str,
    task_name: str,
    *,
    remote: bool = False,
    token: str | bool | None = None,
) -> str | None:
    lora_root = resolve_lora_root(lora_dir, remote=remote, token=token)
    checkpoint = latest_checkpoint(lora_root / task_name)
    return str(checkpoint) if checkpoint is not None else None


def latest_task_lora_checkpoints(
    lora_dir: str,
    *,
    remote: bool = False,
    token: str | bool | None = None,
) -> dict[str, str]:
    lora_root = resolve_lora_root(lora_dir, remote=remote, token=token)
    checkpoints = {}
    for task_dir in lora_root.iterdir():
        checkpoint = latest_checkpoint(task_dir)
        if checkpoint is not None:
            checkpoints[task_dir.name] = str(checkpoint)
    return checkpoints
