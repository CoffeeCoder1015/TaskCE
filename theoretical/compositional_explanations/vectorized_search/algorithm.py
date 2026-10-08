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


def select_semantic_candidates(
    candidate_scores: torch.Tensor,
    parent_vectors: torch.Tensor,
    packed_features: torch.Tensor,
    retained_features: torch.Tensor,
    beam_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return selected vectors, scores, and flat origins for allowed meanings.

    Empty meanings, all original atomic meanings, and current parent meanings
    are excluded. Earlier levels do not contribute additional exclusions.
    """
    flat_scores = candidate_scores.flatten(start_dim=1)
    candidate_count = flat_scores.shape[1]
    word_count = parent_vectors.shape[2]
    empty_meaning = parent_vectors.new_zeros((1, word_count))
    selected_vectors, selected_scores, selected_indices = [], [], []
    for neuron in range(candidate_scores.shape[0]):
        ranked_scores, score_order = torch.sort(
            flat_scores[neuron], descending=True, stable=True
        )
        parents = parent_vectors[neuron, :, None, :]
        features = packed_features[retained_features[neuron]][None, :, :]
        candidate_vectors = torch.stack(
            (parents & features, parents | features, parents & ~features), dim=2
        ).reshape(candidate_count, word_count)
        excluded_vectors = torch.cat(
            (empty_meaning, packed_features, parent_vectors[neuron])
        )
        meanings, meaning_ids = torch.unique(
            torch.cat((excluded_vectors, candidate_vectors)), dim=0, return_inverse=True
        )
        excluded_count = excluded_vectors.shape[0]
        ranked_meaning_ids = meaning_ids[excluded_count:][score_order]
        excluded_meanings = torch.zeros(
            meanings.shape[0], dtype=torch.bool, device=candidate_scores.device
        )
        excluded_meanings[meaning_ids[:excluded_count]] = True
        ranks = torch.arange(candidate_count, device=candidate_scores.device)
        first_rank = torch.full(
            (meanings.shape[0],), candidate_count,
            dtype=torch.int64, device=candidate_scores.device,
        )
        first_rank.scatter_reduce_(0, ranked_meaning_ids, ranks, reduce="amin")
        eligible = (
            torch.isfinite(ranked_scores)
            & ~excluded_meanings[ranked_meaning_ids]
            & (ranks == first_rank[ranked_meaning_ids])
        )
        chosen_ranks = torch.topk(
            torch.where(eligible, ranks, candidate_count),
            beam_size, largest=False, sorted=True,
        ).values
        occupied = chosen_ranks < candidate_count
        gather_ranks = chosen_ranks.clamp_max(candidate_count - 1)
        chosen_indices = score_order[gather_ranks]
        selected_vectors.append(
            candidate_vectors[chosen_indices].masked_fill(~occupied[:, None], 0)
        )
        selected_scores.append(
            ranked_scores[gather_ranks].masked_fill(~occupied, -torch.inf)
        )
        selected_indices.append(chosen_indices)
    return (
        torch.stack(selected_vectors),
        torch.stack(selected_scores),
        torch.stack(selected_indices),
    )


@dataclass(frozen=True)
class FormulaHistory:
    """Append-only ancestry [batch, (maximum length - 1) * beam].

    Atomic IDs are feature indices. Composite ID F + j addresses column j.
    """

    operations: torch.Tensor
    parent_ids: torch.Tensor
    feature_ids: torch.Tensor


def reconstruct_formula(formula_id: int, feature_formulas: list[Symbol], history: FormulaHistory):
    """Follow one winning ancestry chain into SymPy Boolean expressions."""
    if formula_id < len(feature_formulas):
        return feature_formulas[formula_id]
    node = formula_id - len(feature_formulas)
    parent = reconstruct_formula(int(history.parent_ids[node]), feature_formulas, history)
    feature = feature_formulas[int(history.feature_ids[node])]
    operation = int(history.operations[node])
    if operation == AND:
        return And(parent, feature)
    if operation == OR:
        return Or(parent, feature)
    if operation == AND_NOT:
        return And(parent, Not(feature))
    raise ValueError(f"Unknown composition operation {operation}.")


def render_formula(formula) -> str:
    """Render SymPy's simplified expression using the established search syntax."""
    if isinstance(formula, Symbol):
        return str(formula)
    if formula.func is Not:
        return f"(NOT {render_formula(formula.args[0])})"
    if formula.func is And:
        return f"({' AND '.join(render_formula(arg) for arg in formula.args)})"
    if formula.func is Or:
        return f"({' OR '.join(render_formula(arg) for arg in formula.args)})"
    return str(formula)


@dataclass(frozen=True)
class Beam:
    """Per-neuron meanings [batch, beam, words] and their formula identities.

    A finite score identifies an occupied slot; unoccupied vectors are zero.
    """

    vectors: torch.Tensor
    scores: torch.Tensor
    formula_ids: torch.Tensor


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
    beam_size = min(config.beam_size, feature_count)
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
        active = parent_valid.any(dim=1) & (best_scores < 1.0)
        candidate_scores = score_compositions(
            batch_neurons, beam.vectors, packed_features, retained_features,
            parent_valid, retained_valid, active,
        )
        next_vectors, next_scores, selected_indices = select_semantic_candidates(
            candidate_scores, beam.vectors, packed_features, retained_features, beam_size
        )
        operations = selected_indices.remainder(OPERATION_COUNT)
        parent_feature_slots = torch.div(
            selected_indices, OPERATION_COUNT, rounding_mode="floor"
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
            vectors=next_vectors, scores=next_scores, formula_ids=composite_ids
        )

    best_scores, best_ids = best_scores.cpu(), best_ids.cpu()
    history = FormulaHistory(
        operations=history.operations.cpu(),
        parent_ids=history.parent_ids.cpu(),
        feature_ids=history.feature_ids.cpu(),
    )
    results = []
    for neuron in range(batch_size):
        if int(best_ids[neuron]) < 0:
            results.append(SearchResult(
                activation_index=activation_start + neuron,
                best_formula="LOW_ACTS_PRUNED",
                best_score=0.0,
            ))
            continue
        neuron_history = FormulaHistory(
            operations=history.operations[neuron],
            parent_ids=history.parent_ids[neuron],
            feature_ids=history.feature_ids[neuron],
        )
        formula = reconstruct_formula(int(best_ids[neuron]), feature_formulas, neuron_history)
        results.append(SearchResult(
            activation_index=activation_start + neuron,
            best_formula=render_formula(formula),
            best_score=float(best_scores[neuron]),
        ))
    return results


def search_all(
    activation_vectors: torch.Tensor,
    feature_vectors: list[tuple[Symbol, np.ndarray]],
    device: torch.device | str,
    config: SearchConfig | None = None,
) -> list[SearchResult]:
    """Pack inputs once on one CUDA device and process neuron batches in order."""
    config = SearchConfig() if config is None else config
    if not isinstance(config, SearchConfig):
        raise TypeError("config must be a SearchConfig.")
    for field in ("maximum_formula_length", "beam_size", "neuron_batch_size"):
        value = getattr(config, field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"config.{field} must be a positive native integer.")
    if not isinstance(activation_vectors, torch.Tensor):
        raise TypeError("activation_vectors must be a torch.Tensor.")
    if activation_vectors.device.type != "cpu":
        raise ValueError("activation_vectors must be on the CPU.")
    if activation_vectors.layout != torch.strided or activation_vectors.ndim != 2:
        raise ValueError("activation_vectors must be a dense [examples, neurons] matrix.")
    if activation_vectors.dtype != torch.bool:
        raise TypeError("activation_vectors must have Boolean dtype.")
    example_count, neuron_count = activation_vectors.shape
    if not isinstance(feature_vectors, list):
        raise TypeError("feature_vectors must be a list of (Symbol, Boolean array) tuples.")
    for feature_index, entry in enumerate(feature_vectors):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"Feature {feature_index} must be a (Symbol, array) tuple.")
        formula, vector = entry
        if not isinstance(formula, Symbol):
            raise TypeError(f"Feature {feature_index} must have a SymPy Symbol.")
        if not isinstance(vector, np.ndarray) or vector.dtype != np.bool_:
            raise TypeError(f"Feature {feature_index} must have a NumPy Boolean array.")
        if vector.ndim != 1 or vector.shape[0] != example_count:
            raise ValueError(f"Feature {feature_index} must have shape [examples].")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("device must select a CUDA device.")
    if neuron_count > 0 and (example_count == 0 or not feature_vectors):
        raise ValueError("Searching neurons requires examples and at least one feature.")
    if neuron_count == 0:
        return []
    feature_formulas = [formula for formula, vector in feature_vectors]
    packed_features = pack_vectors(
        np.stack([vector for formula, vector in feature_vectors]), device
    )
    packed_neurons = pack_vectors(activation_vectors.numpy().T, device)
    results = []
    for start in range(0, packed_neurons.shape[0], config.neuron_batch_size):
        results.extend(search_batch(
            packed_neurons[start:start + config.neuron_batch_size],
            packed_features, feature_formulas, start, config,
        ))
    return results
