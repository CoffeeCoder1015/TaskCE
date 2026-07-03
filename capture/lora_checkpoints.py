import os
import re

from huggingface_hub import snapshot_download


def resolve_lora_root(
    lora_dir: str,
    *,
    remote: bool = False,
    token: str | bool | None = None,
) -> str:
    lora_dir = os.fspath(lora_dir)
    if remote:
        return snapshot_download(repo_id=lora_dir, repo_type="model", token=token)
    return lora_dir


def task_checkpoint_dirs(lora_root: str, task_name: str):
    task_dir = os.path.join(lora_root, task_name)
    if not os.path.isdir(task_dir):
        return []

    return [
        os.path.join(task_dir, path)
        for path in os.listdir(task_dir)
        if os.path.isdir(os.path.join(task_dir, path))
    ]


def checkpoint_sort_key(path):
    name = os.path.basename(path.rstrip(os.sep))
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    if match:
        return 1, int(match.group(1)), name
    return 0, name


def latest_checkpoint(checkpoints):
    if not checkpoints:
        return None
    return sorted(checkpoints, key=checkpoint_sort_key)[-1]


def latest_task_lora_checkpoint(
    lora_dir: str,
    task_name: str,
    *,
    remote: bool = False,
    token: str | bool | None = None,
):
    lora_root = resolve_lora_root(lora_dir, remote=remote, token=token)
    return latest_checkpoint(task_checkpoint_dirs(lora_root, task_name))


def latest_task_lora_checkpoints(
    lora_dir: str,
    *,
    remote: bool = False,
    token: str | bool | None = None,
):
    lora_root = resolve_lora_root(lora_dir, remote=remote, token=token)
    task_names = [
        path
        for path in os.listdir(lora_root)
        if os.path.isdir(os.path.join(lora_root, path))
    ]
    checkpoints = {
        task_name: latest_checkpoint(task_checkpoint_dirs(lora_root, task_name))
        for task_name in task_names
    }
    return {
        task_name: checkpoint
        for task_name, checkpoint in checkpoints.items()
        if checkpoint is not None
    }
