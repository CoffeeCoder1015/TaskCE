from pathlib import Path

import torch


DEFAULT_CAPTURE_ACTIVATION_SUBDIR = "captured_activations"
CAPTURED_RESULTS_FILENAME = "captured_results.pt"


def save_captured_activations(
    captured_results,
    output_dir,
    subdir=DEFAULT_CAPTURE_ACTIVATION_SUBDIR,
):
    output_dir = Path(output_dir)
    activation_dir = output_dir / subdir
    activation_dir.mkdir(parents=True, exist_ok=True)
    output_path = activation_dir / CAPTURED_RESULTS_FILENAME

    torch.save(captured_results, output_path)
    return output_path


def load_captured_activations(results_path, subdir=DEFAULT_CAPTURE_ACTIVATION_SUBDIR):
    cache_path = captured_results_path(results_path, subdir=subdir)
    return torch.load(cache_path, map_location="cpu", weights_only=False)


def captured_results_path(path, subdir=DEFAULT_CAPTURE_ACTIVATION_SUBDIR):
    path = Path(path)
    if path.is_dir():
        return path / subdir / CAPTURED_RESULTS_FILENAME
    return path
