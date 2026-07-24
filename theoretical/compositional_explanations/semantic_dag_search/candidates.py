"""Candidate scoring and beam materialization for semantic-DAG search."""

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import cast

import sympy
from sympy.logic import And, Not, Or
from sympy.logic.boolalg import Boolean
import torch

from .features import semantic_key
from .scoring import (
    AND_OPERATION,
    LEFT_AND_NOT_RIGHT_OPERATION,
    OPERATION_COUNT,
    OR_OPERATION,
    RIGHT_AND_NOT_LEFT_OPERATION,
    pair_iou_scores,
)


@dataclass(frozen=True, eq=False)
class SemanticState:
    formula: sympy.Basic
    vector: torch.Tensor
    iou: float


@dataclass(frozen=True)
class CandidateSource:
    left_states: tuple[SemanticState, ...]
    right_formulas: tuple[sympy.Basic, ...]
    right_vectors: torch.Tensor
    scores: torch.Tensor


def score_candidate_source(
    left_states: list[SemanticState],
    right_formulas: tuple[sympy.Basic, ...],
    right_vectors: torch.Tensor,
    right_sizes: torch.Tensor,
    right_target_sizes: torch.Tensor,
    target: torch.Tensor,
    target_size: torch.Tensor,
    *,
    right_sparse_vectors: torch.Tensor | None = None,
) -> CandidateSource:
    """Score one retained-level pairing without materializing candidates."""
    left_vectors = torch.stack([state.vector for state in left_states])
    left_target_vectors = left_vectors & target
    combined_left = torch.cat(
        (left_vectors, left_target_vectors),
        dim=0,
    ).T.float()

    if right_sparse_vectors is None:
        pair_counts = right_vectors.float() @ combined_left
    else:
        pair_counts = torch.sparse.mm(
            right_sparse_vectors,
            combined_left,
        )

    left_count = left_vectors.shape[0]
    pair_sizes = pair_counts[:, :left_count].T
    pair_target_sizes = pair_counts[:, left_count:].T
    scores = pair_iou_scores(
        target_size=target_size,
        left_sizes=left_vectors.sum(dim=1).float(),
        left_target_sizes=left_target_vectors.sum(dim=1).float(),
        right_sizes=right_sizes,
        right_target_sizes=right_target_sizes,
        pair_sizes=pair_sizes,
        pair_target_sizes=pair_target_sizes,
    )
    return CandidateSource(
        left_states=tuple(left_states),
        right_formulas=right_formulas,
        right_vectors=right_vectors,
        scores=scores,
    )


def _compose_formula(
    left_formula: sympy.Basic,
    right_formula: sympy.Basic,
    operation: int,
) -> sympy.Basic:
    left = cast(Boolean, left_formula)
    right = cast(Boolean, right_formula)
    if operation == AND_OPERATION:
        return And(left, right)
    if operation == OR_OPERATION:
        return Or(left, right)
    if operation == LEFT_AND_NOT_RIGHT_OPERATION:
        return And(left, Not(right))
    if operation == RIGHT_AND_NOT_LEFT_OPERATION:
        return And(right, Not(left))
    raise ValueError(f"Unknown composition operation {operation}.")


def _compose_vector(
    left: torch.Tensor,
    right: torch.Tensor,
    operation: int,
) -> torch.Tensor:
    if operation == AND_OPERATION:
        return left & right
    if operation == OR_OPERATION:
        return left | right
    if operation == LEFT_AND_NOT_RIGHT_OPERATION:
        return left & ~right
    if operation == RIGHT_AND_NOT_LEFT_OPERATION:
        return right & ~left
    raise ValueError(f"Unknown composition operation {operation}.")


def materialize_destination_beam(
    sources: list[CandidateSource],
    seen_semantics: set[bytes],
    beam_size: int,
) -> list[SemanticState]:
    """Materialize the strongest globally new states for one formula length."""
    if not sources:
        return []

    flat_scores = [source.scores.flatten() for source in sources]
    source_ends = []
    candidate_count = 0
    for scores in flat_scores:
        candidate_count += scores.numel()
        source_ends.append(candidate_count)

    all_scores = torch.cat(flat_scores)
    ordered_scores, ordered_indices = torch.sort(
        all_scores,
        descending=True,
        stable=True,
    )
    destination_beam = []
    offset = 0
    selection_chunk_size = max(64, beam_size * 8)

    while (
        offset < ordered_indices.numel()
        and len(destination_beam) < beam_size
    ):
        stop = min(
            offset + selection_chunk_size,
            ordered_indices.numel(),
        )
        score_values = (
            ordered_scores[offset:stop]
            .detach()
            .cpu()
            .tolist()
        )
        index_values = (
            ordered_indices[offset:stop]
            .detach()
            .cpu()
            .tolist()
        )

        for score, global_index in zip(
            score_values,
            index_values,
            strict=True,
        ):
            if not math.isfinite(score):
                return destination_beam

            source_index = bisect_right(source_ends, global_index)
            source_start = (
                0 if source_index == 0 else source_ends[source_index - 1]
            )
            source = sources[source_index]
            source_candidate_index = global_index - source_start
            operation = source_candidate_index % OPERATION_COUNT
            pair_index = source_candidate_index // OPERATION_COUNT
            right_count = len(source.right_formulas)
            right_index = pair_index % right_count
            left_index = pair_index // right_count

            left_state = source.left_states[left_index]
            right_formula = source.right_formulas[right_index]
            right_vector = source.right_vectors[right_index]
            candidate_vector = _compose_vector(
                left_state.vector,
                right_vector,
                operation,
            )
            candidate_semantics = semantic_key(candidate_vector)
            if candidate_semantics in seen_semantics:
                continue

            seen_semantics.add(candidate_semantics)
            destination_beam.append(
                SemanticState(
                    formula=_compose_formula(
                        left_state.formula,
                        right_formula,
                        operation,
                    ),
                    vector=candidate_vector.clone(),
                    iou=float(score),
                )
            )
            if len(destination_beam) == beam_size:
                return destination_beam

        offset = stop

    return destination_beam
