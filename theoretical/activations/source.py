"""Locate and load raw activation captures produced by the experimental pipeline."""

from pathlib import Path
import re


ACTIVATION_DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data"
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")


def activation_path(
    model_id: str,
    dataset_name: str,
    layer: str,
    *,
    task_name: str | None = None,
    checkpoint_name: str | None = None,
) -> Path:
    """Return one existing activation tensor path."""
    model_directory = ACTIVATION_DATA_DIRECTORY.joinpath(*model_id.split("/"))

    if task_name is None:
        capture_directory = model_directory / "base"
    elif checkpoint_name is not None:
        capture_directory = model_directory / task_name / checkpoint_name
    else:
        task_directory = model_directory / task_name
        checkpoints = [
            (int(match.group(1)), path)
            for path in task_directory.iterdir()
            if path.is_dir()
            and (match := CHECKPOINT_PATTERN.fullmatch(path.name)) is not None
        ]
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint-N directories found under {task_directory}."
            )
        capture_directory = max(checkpoints, key=lambda item: item[0])[1]

    activation_name = f"{dataset_name}_activation.pt"
    exact_path = capture_directory / layer / activation_name
    if exact_path.is_file():
        return exact_path

    suffix_matches = [
        directory / activation_name
        for directory in capture_directory.iterdir()
        if directory.is_dir()
        and directory.name.endswith(f".{layer}")
        and (directory / activation_name).is_file()
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if suffix_matches:
        raise ValueError(
            f"Layer {layer!r} matched multiple saved activation directories: "
            f"{[str(path.parent) for path in suffix_matches]}"
        )
    raise KeyError(layer)


def load_activation(path):
    """Load one capture as a CPU float tensor shaped [examples, neurons]."""
    import torch

    activation = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(activation, torch.Tensor):
        raise TypeError("Experimental activation data must be a tensor.")
    if activation.ndim != 2:
        raise ValueError(
            "Experimental activation data must have shape [examples, neurons], "
            f"got {tuple(activation.shape)}."
        )
    return activation.detach().cpu().to(torch.float32)
