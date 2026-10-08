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
    raise NotImplementedError("Beam expansion has not been assembled yet.")
