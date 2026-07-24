"""Bounded semantic-DAG search over retained formula levels."""

from dataclasses import dataclass

import sympy
import torch

from .candidates import (
    SemanticState,
    materialize_destination_beam,
    score_candidate_source,
)
from .features import (
    PreparedFeatures,
    prepare_feature_matrix,
    prepare_initial_features,
    render_formula,
)
from .scoring import (
    intersection_over_union_from_counts,
)


@dataclass(frozen=True)
class SearchConfig:
    maximum_formula_length: int = 5
    beam_size: int = 10


@dataclass(frozen=True)
class SearchOutcome:
    formula: str
    iou: float


def _validate_config(config: SearchConfig) -> None:
    if config.maximum_formula_length < 1:
        raise ValueError("maximum_formula_length must be at least one.")
    if config.beam_size < 1:
        raise ValueError("beam_size must be at least one.")
def _initial_beams(
    features: PreparedFeatures,
    atomic_scores: torch.Tensor,
    beam_size: int,
) -> list[list[SemanticState]]:
    ordered_scores, ordered_indices = torch.sort(
        atomic_scores.T,
        dim=1,
        descending=True,
        stable=True,
    )
    ordered_scores = ordered_scores[:, :beam_size].detach().cpu()
    ordered_indices = ordered_indices[:, :beam_size].detach().cpu()
    beams = []

    for neuron_index in range(atomic_scores.shape[1]):
        beam = []
        for score, feature_index in zip(
            ordered_scores[neuron_index].tolist(),
            ordered_indices[neuron_index].tolist(),
            strict=True,
        ):
            if score <= 0:
                break
            beam.append(
                SemanticState(
                    formula=features.formulas[feature_index],
                    vector=features.vectors[feature_index],
                    iou=float(score),
                )
            )
        beams.append(beam)

    return beams


def _search_one(
    target: torch.Tensor,
    features: PreparedFeatures,
    initial_beam: list[SemanticState],
    feature_target_sizes: torch.Tensor,
    config: SearchConfig,
) -> SearchOutcome:
    if not initial_beam:
        return SearchOutcome(
            formula="LOW_ACTS_PRUNED",
            iou=0.0,
        )

    levels = {1: initial_beam}
    seen_semantics = set(features.semantic_keys)
    best_state = initial_beam[0]
    target_size = target.sum().float()

    for destination_length in range(
        2,
        config.maximum_formula_length + 1,
    ):
        if best_state.iou >= 1.0:
            break

        sources = []
        for right_length in range(1, destination_length // 2 + 1):
            left_length = destination_length - right_length
            left_states = levels.get(left_length, [])
            if not left_states:
                continue

            if right_length == 1:
                # Keep ordinary atomic expansion complete. The archived
                # length-one beam bounds reusable left states, while every
                # atomic feature remains available as a right-hand extension.
                sources.append(
                    score_candidate_source(
                        left_states,
                        features.formulas,
                        features.vectors,
                        features.sizes,
                        feature_target_sizes,
                        target,
                        target_size,
                        right_sparse_vectors=features.sparse_vectors,
                    )
                )
                continue

            right_states = levels.get(right_length, [])
            if not right_states:
                continue
            right_vectors = torch.stack(
                [state.vector for state in right_states]
            )
            sources.append(
                score_candidate_source(
                    left_states,
                    tuple(state.formula for state in right_states),
                    right_vectors,
                    right_vectors.sum(dim=1).float(),
                    (right_vectors & target).sum(dim=1).float(),
                    target,
                    target_size,
                )
            )

        destination_beam = materialize_destination_beam(
            sources,
            seen_semantics,
            config.beam_size,
        )
        levels[destination_length] = destination_beam
        for state in destination_beam:
            if state.iou > best_state.iou:
                best_state = state

    return SearchOutcome(
        formula=render_formula(best_state.formula),
        iou=best_state.iou,
    )


@torch.inference_mode()
def search_batch(
    neurons: torch.Tensor,
    features: PreparedFeatures,
    config: SearchConfig | None = None,
) -> list[SearchOutcome]:
    """Search a neuron batch using a bounded DAG of retained semantic states."""
    config = config or SearchConfig()
    _validate_config(config)
    neurons = torch.as_tensor(
        neurons,
        dtype=torch.bool,
        device=features.vectors.device,
    )
    if neurons.ndim != 2:
        raise ValueError("Neuron batches must have shape [examples, neurons].")
    if neurons.shape[0] != features.example_count:
        raise ValueError(
            "Feature and neuron vectors must contain the same examples: "
            f"{features.example_count} != {neurons.shape[0]}."
        )
    if neurons.shape[1] == 0:
        return []

    target_sizes = neurons.sum(dim=0).float()
    feature_target_sizes = torch.sparse.mm(
        features.sparse_vectors,
        neurons.float(),
    )
    atomic_scores = intersection_over_union_from_counts(
        feature_target_sizes,
        features.sizes[:, None],
        target_sizes[None, :],
    )
    initial_beams = _initial_beams(
        features,
        atomic_scores,
        config.beam_size,
    )

    return [
        _search_one(
            neurons[:, neuron_index],
            features,
            initial_beams[neuron_index],
            feature_target_sizes[:, neuron_index],
            config,
        )
        for neuron_index in range(neurons.shape[1])
    ]


def search(
    neuron: torch.Tensor,
    feature_vectors: list[tuple[sympy.Basic, object]],
    config: SearchConfig | None = None,
) -> SearchOutcome:
    """Search one neuron through the same semantic-DAG implementation."""
    config = config or SearchConfig()
    device = neuron.device
    feature_vectors = prepare_initial_features(feature_vectors)
    features = prepare_feature_matrix(feature_vectors, device)
    return search_batch(
        neuron[:, None],
        features,
        config,
    )[0]
