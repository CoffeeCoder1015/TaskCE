from __future__ import annotations

from pathlib import Path


RESULTS_DIR = Path("results")
CAPTURED_RESULTS_PATH = RESULTS_DIR / "captured_activations" / "captured_results.pt"
GRAPH_DIR = RESULTS_DIR / "graph"
OUTPUT_TENSOR_FILENAME = "activation_cross_movements.pt"


def load_captured_results():
    import torch

    return torch.load(CAPTURED_RESULTS_PATH, map_location="cpu", weights_only=False)


def activation_cross_task_matrices(captured_results) -> dict[str, dict[str, object]]:
    import torch

    from analysis.activation_diagnostics import (
        raw_activation_correlation_matrix,
        raw_activation_cosine_similarity_matrix,
        validate_same_activation_shape,
    )

    tasks: dict[str, dict[str, object]] = {}
    for task_name, task_results in captured_results.items():
        if "base" not in task_results or "finetuned" not in task_results:
            continue

        base_states = task_results["base"].states
        finetuned_states = task_results["finetuned"].states
        validate_same_activation_shape(base_states, finetuned_states, "base", "finetuned")

        neuron_count = int(base_states.shape[1])
        vectors = torch.cat((base_states, finetuned_states), dim=1)
        pearson_full = raw_activation_correlation_matrix(vectors)
        cosine_full = raw_activation_cosine_similarity_matrix(vectors)

        tasks[str(task_name)] = {
            "neuron_count": neuron_count,
            "pearson": pearson_full[:neuron_count, neuron_count:],
            "cosine": cosine_full[:neuron_count, neuron_count:],
        }

    return tasks


def write_activation_cross_tensors(
    task_matrices: dict[str, dict[str, object]],
    tensor_path: Path,
) -> None:
    import torch

    payload = {
        "format": "activation_cross_movements",
        "dtype": "float32",
        "tasks": {},
    }

    for task_name in sorted(task_matrices):
        task_payload = task_matrices[task_name]
        payload["tasks"][task_name] = {
            "neuron_count": int(task_payload["neuron_count"]),
            "pearson": torch.as_tensor(task_payload["pearson"], dtype=torch.float32),
            "cosine": torch.as_tensor(task_payload["cosine"], dtype=torch.float32),
        }

    torch.save(payload, tensor_path)


def build_viewers() -> Path:
    captured_results = load_captured_results()
    task_matrices = activation_cross_task_matrices(captured_results)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    output_tensor_path = GRAPH_DIR / OUTPUT_TENSOR_FILENAME
    write_activation_cross_tensors(task_matrices, output_tensor_path)

    return output_tensor_path


def main() -> int:
    output_tensor_path = build_viewers()
    print(f"Wrote {output_tensor_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
