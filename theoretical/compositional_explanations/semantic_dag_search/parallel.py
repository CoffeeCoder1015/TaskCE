"""Multiprocessing coordination for semantic-DAG search."""

from collections.abc import Sequence
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np
import torch

from .algorithm import SearchConfig, search_batch
from .features import prepare_feature_matrix, prepare_initial_features


@dataclass(frozen=True)
class NeuronSearchResult:
    neuron_id: int
    formula: str
    iou: float


def resolve_search_device(device=None) -> torch.device:
    """Prefer CUDA for search unless the caller selects another device."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def numpy_bool(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.bool_, copy=False)
    return np.asarray(value, dtype=np.bool_)


def chunk_ranges(item_count: int, worker_count: int) -> list[tuple[int, int]]:
    if item_count == 0:
        return []

    worker_count = min(max(1, worker_count), item_count)
    chunk_size, remainder = divmod(item_count, worker_count)
    ranges = []
    start = 0
    for worker_index in range(worker_count):
        stop = start + chunk_size
        if worker_index < remainder:
            stop += 1
        ranges.append((start, stop))
        start = stop
    return ranges


def search_worker(
    activation_vectors,
    neuron_ids: Sequence[int],
    feature_vectors,
    device=None,
    config: SearchConfig | None = None,
    neuron_batch_size: int = 8,
) -> list[NeuronSearchResult]:
    """Search one neuron slice in semantic-DAG batches."""
    if not neuron_ids:
        return []
    if neuron_batch_size < 1:
        raise ValueError("neuron_batch_size must be at least one.")

    config = config or SearchConfig()
    worker_device = resolve_search_device(device)
    prepared_features = prepare_feature_matrix(
        feature_vectors,
        worker_device,
    )
    prepared_activations = torch.as_tensor(
        activation_vectors,
        dtype=torch.bool,
        device=worker_device,
    )

    results = []
    for start in range(
        0,
        len(neuron_ids),
        neuron_batch_size,
    ):
        stop = min(start + neuron_batch_size, len(neuron_ids))
        outcomes = search_batch(
            prepared_activations[:, start:stop].contiguous(),
            prepared_features,
            config,
        )
        results.extend(
            NeuronSearchResult(
                neuron_id=int(neuron_id),
                formula=outcome.formula,
                iou=outcome.iou,
            )
            for neuron_id, outcome in zip(
                neuron_ids[start:stop],
                outcomes,
                strict=True,
            )
        )

    return results


def search_all(
    activation_vectors,
    neuron_ids: Sequence[int],
    feature_vectors,
    *,
    num_workers: int = 1,
    device=None,
    config: SearchConfig | None = None,
    neuron_batch_size: int = 8,
) -> list[NeuronSearchResult]:
    """Search every retained neuron serially or across spawned processes."""
    neuron_ids = [int(neuron_id) for neuron_id in neuron_ids]
    if activation_vectors.shape[1] != len(neuron_ids):
        raise ValueError(
            "Activation columns must correspond one-to-one with neuron IDs: "
            f"{activation_vectors.shape[1]} != {len(neuron_ids)}."
        )
    if not neuron_ids:
        return []

    feature_vectors = prepare_initial_features(feature_vectors)

    if num_workers <= 1:
        return search_worker(
            activation_vectors,
            neuron_ids,
            feature_vectors,
            device,
            config,
            neuron_batch_size,
        )

    activation_vectors = numpy_bool(activation_vectors)
    feature_vectors = [
        (formula, numpy_bool(vector))
        for formula, vector in feature_vectors
    ]
    ranges = chunk_ranges(len(neuron_ids), num_workers)
    worker_arguments = [
        (
            activation_vectors[:, start:stop],
            neuron_ids[start:stop],
            feature_vectors,
            device,
            config,
            neuron_batch_size,
        )
        for start, stop in ranges
    ]

    context = mp.get_context("spawn")
    with context.Pool(processes=len(worker_arguments)) as pool:
        result_chunks = pool.starmap(
            search_worker,
            worker_arguments,
        )

    return [
        result
        for result_chunk in result_chunks
        for result in result_chunk
    ]
