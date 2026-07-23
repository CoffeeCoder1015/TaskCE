"""Resolve activation files already generated under repository-level data."""

from pathlib import Path, PureWindowsPath
import re


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


def _validate_component(value: str, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    if not value or value in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty path component.")
    if "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field} must be one path component.")
    if Path(value).is_absolute() or PureWindowsPath(value).drive:
        raise ValueError(f"{field} must be one path component.")


def _model_directory(model_id: str) -> Path:
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string.")
    components = tuple(model_id.split("/"))
    if not components:
        raise ValueError("model_id must contain at least one path component.")
    for index, component in enumerate(components):
        _validate_component(component, f"model_id component {index}")
    return DATA_DIRECTORY.joinpath(*components)


def resolve_latest_checkpoint(model_id: str, task_name: str) -> Path:
    """Return the latest checkpoint directory already generated for a task."""
    _validate_component(task_name, "task_name")
    task_directory = _model_directory(model_id) / task_name
    if not task_directory.is_dir():
        raise FileNotFoundError(
            f"No generated checkpoint directory exists for task {task_name!r}: "
            f"{task_directory}"
        )

    checkpoints = [path for path in task_directory.iterdir() if path.is_dir()]
    if not checkpoints:
        raise FileNotFoundError(
            f"No generated checkpoints exist for task {task_name!r}: "
            f"{task_directory}"
        )

    def sort_key(checkpoint: Path):
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
        if match:
            return 1, int(match.group(1)), checkpoint.name
        return 0, 0, checkpoint.name

    return max(checkpoints, key=sort_key)


def resolve_activation_path(
    model_id: str,
    dataset: str,
    layer: str,
    *,
    checkpoint: str | Path | None = None,
) -> Path:
    """Return one existing activation file for the base or selected checkpoint."""
    _validate_component(dataset, "dataset")
    _validate_component(layer, "layer")
    model_directory = _model_directory(model_id)

    if checkpoint is None:
        variant_directory = model_directory / "base"
    else:
        variant_directory = Path(checkpoint)
        if not variant_directory.is_absolute():
            variant_directory = model_directory / variant_directory
        try:
            relative_checkpoint = variant_directory.resolve().relative_to(
                model_directory.resolve()
            )
        except ValueError as error:
            raise ValueError(
                f"checkpoint must be inside the generated model directory: "
                f"{model_directory}"
            ) from error
        if len(relative_checkpoint.parts) != 2:
            raise ValueError(
                "checkpoint must identify one generated task/checkpoint directory, "
                f"got {relative_checkpoint}."
            )

    if not variant_directory.is_dir():
        raise FileNotFoundError(
            f"Generated activation variant does not exist: {variant_directory}"
        )

    activation_name = f"{dataset}_activation.pt"
    exact_path = variant_directory / layer / activation_name
    if exact_path.is_file():
        return exact_path

    suffix = f".{layer}"
    candidates = [
        layer_directory / activation_name
        for layer_directory in variant_directory.iterdir()
        if (
            layer_directory.is_dir()
            and layer_directory.name.endswith(suffix)
            and (layer_directory / activation_name).is_file()
        )
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No generated activation for dataset {dataset!r} and layer "
            f"{layer!r} exists beneath {variant_directory}."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Layer {layer!r} matched multiple generated activation files: "
            f"{candidates}"
        )
    return candidates[0]
