"""Beam search over Boolean meanings, with ancestry kept separately from vectors."""

from dataclasses import dataclass

import numpy as np
from sympy import Symbol
from sympy.logic.boolalg import And, Not, Or
import torch

from .kernels import pack_vectors, score_atoms, score_compositions

# The final score axis, flat candidate identity, and ancestry share this order.
AND = 0
OR = 1
AND_NOT = 2
OPERATION_COUNT = 3


@dataclass(frozen=True)
class SearchConfig:
    maximum_formula_length: int = 5
    beam_size: int = 10
    neuron_batch_size: int = 8



@dataclass(frozen=True)
class SearchResult:
    activation_index: int
    best_formula: str
    best_score: float



@dataclass(frozen=True)
class Beam:
    """Per-neuron meanings [batch, beam, words] and their formula identities.

    A finite score identifies an occupied slot; unoccupied vectors are zero.
    """

    vectors: torch.Tensor
    scores: torch.Tensor
    formula_ids: torch.Tensor



@dataclass(frozen=True)
class FormulaHistory:
    """Append-only ancestry [batch, (maximum length - 1) * beam].

    Atomic IDs are feature indices. Composite ID F + j addresses column j.
    """

    operations: torch.Tensor
    parent_ids: torch.Tensor
    feature_ids: torch.Tensor



@dataclass(frozen=True)
class CandidateSelection:
    """Selected meanings, scores, and original flat candidate identities.

    Indices flatten [parent slot, retained feature slot, operation]. A finite
    score identifies a selected slot; filler indices are safe to gather.
    """

    vectors: torch.Tensor
    scores: torch.Tensor
    indices: torch.Tensor



def select_semantic_candidates(
    candidate_scores: torch.Tensor,
    parent_vectors: torch.Tensor,
    packed_features: torch.Tensor,
    retained_features: torch.Tensor,
    beam_size: int,
) -> CandidateSelection:
    """Keep the earliest score-ranked representative of each allowed meaning.

    Empty meanings, all original atomic meanings, and current parent meanings
    are excluded. Earlier levels do not contribute additional exclusions.
    """
    raise NotImplementedError("Semantic selection is added in the next stage.")


def search_batch(
    batch_neurons: torch.Tensor,
    packed_features: torch.Tensor,
    feature_formulas: list[Symbol],
    activation_start: int,
    config: SearchConfig,
) -> list[SearchResult]:
    """Search one packed neuron batch; keep best scores across all beam levels."""
    batch_size = batch_neurons.shape[0]
    feature_count = packed_features.shape[0]
    beam_size = config.beam_size
    device = batch_neurons.device
    atom_scores = score_atoms(batch_neurons, packed_features)
    retained_count = max(beam_size, int((atom_scores > 0).sum(dim=1).max().item()))
    retained_scores, retained_features = torch.topk(
        atom_scores.masked_fill(atom_scores <= 0, -torch.inf), retained_count, dim=1
    )
    retained_valid = torch.isfinite(retained_scores)
    initial_scores = retained_scores[:, :beam_size]
    initial_ids = retained_features[:, :beam_size]
    beam = Beam(
        vectors=packed_features[initial_ids].masked_fill(
            ~torch.isfinite(initial_scores)[:, :, None], 0
        ),
        scores=initial_scores,
        formula_ids=initial_ids,
    )
    best_scores = torch.full((batch_size,), -torch.inf, device=device)
    best_ids = torch.full((batch_size,), -1, dtype=torch.int64, device=device)
    history_capacity = (config.maximum_formula_length - 1) * beam_size
    history = FormulaHistory(
        operations=torch.empty((batch_size, history_capacity), dtype=torch.int8, device=device),
        parent_ids=torch.empty((batch_size, history_capacity), dtype=torch.int64, device=device),
        feature_ids=torch.empty((batch_size, history_capacity), dtype=torch.int64, device=device),
    )
    neuron_rows = torch.arange(batch_size, device=device)[:, None]
    for formula_length in range(1, config.maximum_formula_length + 1):
        improved = beam.scores[:, 0] > best_scores
        best_scores = torch.where(improved, beam.scores[:, 0], best_scores)
        best_ids = torch.where(improved, beam.formula_ids[:, 0], best_ids)
        if formula_length == config.maximum_formula_length:
            break
        parent_valid = torch.isfinite(beam.scores)
        active = parent_valid.any(dim=1)
        candidate_scores = score_compositions(
            batch_neurons, beam.vectors, packed_features, retained_features,
            parent_valid, retained_valid, active,
        )
        selection = select_semantic_candidates(
            candidate_scores, beam.vectors, packed_features, retained_features, beam_size
        )
        operations = selection.indices.remainder(OPERATION_COUNT)
        parent_feature_slots = torch.div(
            selection.indices, OPERATION_COUNT, rounding_mode="floor"
        )
        feature_slots = parent_feature_slots.remainder(retained_count)
        parent_slots = torch.div(parent_feature_slots, retained_count, rounding_mode="floor")
        node_start = (formula_length - 1) * beam_size
        node_stop = node_start + beam_size
        history.operations[:, node_start:node_stop] = operations
        history.parent_ids[:, node_start:node_stop] = beam.formula_ids[
            neuron_rows, parent_slots
        ]
        history.feature_ids[:, node_start:node_stop] = retained_features[
            neuron_rows, feature_slots
        ]
        composite_ids = (
            feature_count + node_start + torch.arange(beam_size, device=device)[None, :]
        ).expand(batch_size, -1)
        beam = Beam(
            vectors=selection.vectors, scores=selection.scores, formula_ids=composite_ids
        )

    raise NotImplementedError("Winner reconstruction is added in final assembly.")
