"""Locate LoRA checkpoints outside the model wrapper."""

import re
from pathlib import Path

from huggingface_hub import snapshot_download


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


def _latest_checkpoint(task_dir: Path) -> Path:
    checkpoints = [path for path in task_dir.iterdir() if path.is_dir()]

    def sort_key(path: Path):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match:
            return 1, int(match.group(1)), path.name
        return 0, 0, path.name

    return max(checkpoints, key=sort_key)


def latest_task_lora_checkpoints(
    lora_dir: str,
    *,
    remote: bool = False,
    token: str | bool | None = None,
) -> dict[str, str]:
    lora_root = resolve_lora_root(lora_dir, remote=remote, token=token)
    checkpoints = {}
    for task_dir in lora_root.iterdir():
        if task_dir.is_dir():
            checkpoints[task_dir.name] = str(_latest_checkpoint(task_dir))
    return checkpoints
