from pathlib import Path

from huggingface_hub import snapshot_download


def get_lora_checkpoints(repo_id: str) -> dict[str, Path]:
    """Download adapters and select the highest checkpoint number for each task."""
    root = Path(snapshot_download(repo_id))
    latest = {}
    for task in root.iterdir():
        configs = list(task.glob("checkpoint-*/adapter_config.json"))
        if configs:
            config = max(
                configs,
                key=lambda path: int(path.parent.name.removeprefix("checkpoint-")),
            )
            latest[task.name] = config.parent
    return latest
