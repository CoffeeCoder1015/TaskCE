"""Layered compositional explanation search."""

from dataclasses import dataclass
import multiprocessing as mp

import numpy as np
from sympy import Symbol
from sympy.logic.boolalg import And, Not, Or
import torch

from .. import NO_EXPLANATION_FORMULA
from .kernels import (
    atomic_iou_scores,
    composition_iou_scores,
    pack_vectors,
)


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
class Queue:
    vectors: torch.Tensor
    scores: torch.Tensor
    formula_ids: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class Best:
    scores: torch.Tensor
    formula_ids: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class FormulaTracking:
    operations: torch.Tensor
    parent_ids: torch.Tensor
    feature_ids: torch.Tensor


def select_unique_topk(
    candidate_scores: torch.Tensor,
    parent_vectors: torch.Tensor,
    packed_features: torch.Tensor,
    retained_ids: torch.Tensor,
    beam_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_scores = candidate_scores.flatten(start_dim=1)
    candidate_count = flat_scores.shape[1]
    packed_width = parent_vectors.shape[2]
    empty_vector = torch.zeros(
        (1, packed_width),
        dtype=parent_vectors.dtype,
        device=parent_vectors.device,
    )

    selected_scores = []
    selected_indices = []
    selected_valid = []
    for neuron_index in range(candidate_scores.shape[0]):
        scores, indices = torch.sort(
            flat_scores[neuron_index],
            descending=True,
            stable=True,
        )

        features = packed_features[retained_ids[neuron_index]]
        parents = parent_vectors[neuron_index, :, None, :]
        features = features[None, :, :]
        candidate_vectors = torch.stack(
            (
                parents & features,
                parents | features,
                parents & ~features,
            ),
            dim=2,
        ).reshape(candidate_count, packed_width)

        reference_count = (
            1
            + packed_features.shape[0]
            + parent_vectors.shape[1]
        )
        unique_vectors, inverse = torch.unique(
            torch.cat(
                (
                    empty_vector,
                    packed_features,
                    parent_vectors[neuron_index],
                    candidate_vectors,
                )
            ),
            dim=0,
            return_inverse=True,
        )
        reference_keys = inverse[:reference_count]
        candidate_keys = inverse[reference_count:]
        seen = torch.zeros(
            unique_vectors.shape[0],
            dtype=torch.bool,
            device=candidate_scores.device,
        )
        seen[reference_keys] = True

        ordered_keys = candidate_keys[indices]
        positions = torch.arange(
            candidate_count,
            device=candidate_scores.device,
        )
        first_positions = torch.full(
            (unique_vectors.shape[0],),
            candidate_count,
            dtype=torch.int64,
            device=candidate_scores.device,
        )
        first_positions.scatter_reduce_(
            0,
            ordered_keys,
            positions,
            reduce="amin",
        )
        eligible = (
            torch.isfinite(scores)
            & ~seen[ordered_keys]
            & (positions == first_positions[ordered_keys])
        )

        winner_positions = torch.topk(
            torch.where(
                eligible,
                positions,
                candidate_count,
            ),
            beam_size,
            largest=False,
            sorted=True,
        ).values
        valid = winner_positions < candidate_count
        winner_positions = winner_positions.clamp_max(
            candidate_count - 1
        )
        selected_scores.append(
            scores[winner_positions].masked_fill(
                ~valid,
                -torch.inf,
            )
        )
        selected_indices.append(indices[winner_positions])
        selected_valid.append(valid)

    return (
        torch.stack(selected_scores),
        torch.stack(selected_indices),
        torch.stack(selected_valid),
    )


def reconstruct_formula(
    formula_id: int,
    feature_count: int,
    feature_formulas: list[Symbol],
    operations: torch.Tensor,
    parent_ids: torch.Tensor,
    feature_ids: torch.Tensor,
):
    formulas = {}

    def reconstruct(current_id: int):
        if current_id < feature_count:
            return feature_formulas[current_id]
        if current_id in formulas:
            return formulas[current_id]

        node_index = current_id - feature_count
        parent = reconstruct(int(parent_ids[node_index]))
        feature = feature_formulas[int(feature_ids[node_index])]
        operation = int(operations[node_index])
        if operation == AND:
            formula = And(parent, feature)
        elif operation == OR:
            formula = Or(parent, feature)
        elif operation == AND_NOT:
            formula = And(parent, Not(feature))
        else:
            raise ValueError(
                f"Unknown composition operation {operation}."
            )

        formulas[current_id] = formula
        return formula

    return reconstruct(formula_id)


def render_formula(formula) -> str:
    if isinstance(formula, Symbol):
        return str(formula)
    if formula.func is Not:
        return f"(NOT {render_formula(formula.args[0])})"
    if formula.func is And:
        return f"({' AND '.join(render_formula(arg) for arg in formula.args)})"
    if formula.func is Or:
        return f"({' OR '.join(render_formula(arg) for arg in formula.args)})"
    return str(formula)


def filter_nonzero_features(
    neurons: torch.Tensor,
    packed_features: torch.Tensor,
    beam_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_scores = atomic_iou_scores(
        neurons,
        packed_features,
    )
    retained_count = max(
        beam_size,
        int((feature_scores > 0).sum(dim=1).max().item()),
    )
    retained_scores, retained_ids = torch.topk(
        feature_scores.masked_fill(
            feature_scores <= 0,
            -torch.inf,
        ),
        retained_count,
        dim=1,
    )
    return (
        retained_scores,
        retained_ids,
        torch.isfinite(retained_scores),
    )


def setup_search(
    retained_scores: torch.Tensor,
    retained_ids: torch.Tensor,
    retained_valid: torch.Tensor,
    packed_features: torch.Tensor,
    config: SearchConfig,
) -> tuple[Queue, Best, FormulaTracking]:
    queue_scores = retained_scores[:, : config.beam_size]
    queue_formula_ids = retained_ids[:, : config.beam_size]
    queue = Queue(
        vectors=packed_features[queue_formula_ids],
        scores=queue_scores,
        formula_ids=queue_formula_ids,
        valid=retained_valid[:, : config.beam_size],
    )

    neuron_count = retained_scores.shape[0]
    composite_capacity = (
        config.maximum_formula_length - 1
    ) * config.beam_size
    best = Best(
        scores=torch.full(
            (neuron_count,),
            -torch.inf,
            device=retained_scores.device,
        ),
        formula_ids=torch.full(
            (neuron_count,),
            -1,
            dtype=torch.int64,
            device=retained_scores.device,
        ),
        valid=torch.zeros(
            neuron_count,
            dtype=torch.bool,
            device=retained_scores.device,
        ),
    )
    formulas = FormulaTracking(
        operations=torch.empty(
            (neuron_count, composite_capacity),
            dtype=torch.int8,
            device=retained_scores.device,
        ),
        parent_ids=torch.empty(
            (neuron_count, composite_capacity),
            dtype=torch.int64,
            device=retained_scores.device,
        ),
        feature_ids=torch.empty(
            (neuron_count, composite_capacity),
            dtype=torch.int64,
            device=retained_scores.device,
        ),
    )
    return queue, best, formulas


def update_best(
    queue: Queue,
    best: Best,
) -> tuple[Best, torch.Tensor]:
    current_valid = queue.valid[:, 0]
    replace = (
        current_valid
        & (
            ~best.valid
            | (queue.scores[:, 0] > best.scores)
        )
    )
    best = Best(
        scores=torch.where(
            replace,
            queue.scores[:, 0],
            best.scores,
        ),
        formula_ids=torch.where(
            replace,
            queue.formula_ids[:, 0],
            best.formula_ids,
        ),
        valid=best.valid | current_valid,
    )
    active = queue.valid.any(dim=1) & (best.scores < 1.0)
    return best, active


def score_compositions(
    neurons: torch.Tensor,
    queue: Queue,
    packed_features: torch.Tensor,
    retained_ids: torch.Tensor,
    retained_valid: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    return composition_iou_scores(
        neurons,
        queue.vectors,
        packed_features,
        retained_ids,
        queue.valid,
        retained_valid,
        active,
    )


def prune_level(
    candidate_scores: torch.Tensor,
    queue: Queue,
    packed_features: torch.Tensor,
    retained_ids: torch.Tensor,
    formulas: FormulaTracking,
    formula_length: int,
    beam_size: int,
) -> tuple[Queue, FormulaTracking]:
    queue_scores, winner_indices, queue_valid = select_unique_topk(
        candidate_scores,
        queue.vectors,
        packed_features,
        retained_ids,
        beam_size,
    )

    retained_count = retained_ids.shape[1]
    operations = winner_indices.remainder(OPERATION_COUNT)
    parent_feature_indices = torch.div(
        winner_indices,
        OPERATION_COUNT,
        rounding_mode="floor",
    )
    feature_slots = parent_feature_indices.remainder(retained_count)
    parent_slots = torch.div(
        parent_feature_indices,
        retained_count,
        rounding_mode="floor",
    )

    neuron_count = candidate_scores.shape[0]
    neuron_rows = torch.arange(
        neuron_count,
        device=candidate_scores.device,
    )[:, None]
    parent_formula_ids = queue.formula_ids[
        neuron_rows,
        parent_slots,
    ]
    feature_ids = retained_ids[
        neuron_rows,
        feature_slots,
    ]

    node_start = (
        formula_length - 1
    ) * beam_size
    node_stop = node_start + beam_size
    operations_history = formulas.operations.clone()
    parent_history = formulas.parent_ids.clone()
    feature_history = formulas.feature_ids.clone()
    operations_history[:, node_start:node_stop] = operations
    parent_history[:, node_start:node_stop] = parent_formula_ids
    feature_history[:, node_start:node_stop] = feature_ids
    formulas = FormulaTracking(
        operations=operations_history,
        parent_ids=parent_history,
        feature_ids=feature_history,
    )

    parent_vectors = queue.vectors[
        neuron_rows,
        parent_slots,
    ]
    feature_vectors = packed_features[feature_ids]
    queue_vectors = torch.where(
        (operations == AND)[:, :, None],
        parent_vectors & feature_vectors,
        torch.where(
            (operations == OR)[:, :, None],
            parent_vectors | feature_vectors,
            parent_vectors & ~feature_vectors,
        ),
    )
    queue_vectors = torch.where(
        queue_valid[:, :, None],
        queue_vectors,
        0,
    )
    queue_formula_ids = (
        packed_features.shape[0]
        + node_start
        + torch.arange(
            beam_size,
            device=candidate_scores.device,
        )[None, :]
    ).expand(neuron_count, -1)

    return (
        Queue(
            vectors=queue_vectors,
            scores=queue_scores,
            formula_ids=queue_formula_ids,
            valid=queue_valid,
        ),
        formulas,
    )


def build_result_one(
    activation_index: int,
    best_score: torch.Tensor,
    best_formula_id: torch.Tensor,
    best_valid: torch.Tensor,
    feature_formulas: list[Symbol],
    formulas: FormulaTracking,
) -> SearchResult:
    if not bool(best_valid):
        return SearchResult(
            activation_index=activation_index,
            best_formula=NO_EXPLANATION_FORMULA,
            best_score=0.0,
        )

    formula = reconstruct_formula(
        formula_id=int(best_formula_id),
        feature_count=len(feature_formulas),
        feature_formulas=feature_formulas,
        operations=formulas.operations,
        parent_ids=formulas.parent_ids,
        feature_ids=formulas.feature_ids,
    )
    return SearchResult(
        activation_index=activation_index,
        best_formula=render_formula(formula),
        best_score=float(best_score),
    )


def build_results(
    neuron_indices: list[int],
    best: Best,
    feature_formulas: list[Symbol],
    formulas: FormulaTracking,
) -> list[SearchResult]:
    best = Best(
        scores=best.scores.cpu(),
        formula_ids=best.formula_ids.cpu(),
        valid=best.valid.cpu(),
    )
    formulas = FormulaTracking(
        operations=formulas.operations.cpu(),
        parent_ids=formulas.parent_ids.cpu(),
        feature_ids=formulas.feature_ids.cpu(),
    )
    return [
        build_result_one(
            activation_index,
            best.scores[batch_index],
            best.formula_ids[batch_index],
            best.valid[batch_index],
            feature_formulas,
            FormulaTracking(
                operations=formulas.operations[batch_index],
                parent_ids=formulas.parent_ids[batch_index],
                feature_ids=formulas.feature_ids[batch_index],
            ),
        )
        for batch_index, activation_index in enumerate(neuron_indices)
    ]


def search_worker(
    activation_vectors: np.ndarray,
    activation_indices: list[int],
    feature_vectors: list[tuple[Symbol, np.ndarray]],
    device,
    config: SearchConfig,
) -> list[SearchResult]:
    worker_device = torch.device(device or "cuda")
    feature_formulas = [
        formula
        for formula, _vector in feature_vectors
    ]
    packed_features = pack_vectors(
        np.stack(
            [
                vector
                for _formula, vector in feature_vectors
            ]
        ),
        worker_device,
    )
    packed_activations = pack_vectors(
        activation_vectors.T,
        worker_device,
    )

    results = []
    for start in range(
        0,
        len(activation_indices),
        config.neuron_batch_size,
    ):
        stop = min(
            start + config.neuron_batch_size,
            len(activation_indices),
        )
        neurons = packed_activations[start:stop]
        neuron_indices = activation_indices[start:stop]

        # 1. Score atomic features and remove zero-IoU features.
        retained_scores, retained_ids, retained_valid = filter_nonzero_features(
            neurons,
            packed_features,
            config.beam_size,
        )

        # 2. Create the initial queues and formula tracking.
        queue, best, formulas = setup_search(
            retained_scores,
            retained_ids,
            retained_valid,
            packed_features,
            config,
        )

        formula_length = 1
        while formula_length <= config.maximum_formula_length:
            # 3a. Record the best current formula.
            best, active = update_best(
                queue,
                best,
            )
            if formula_length == config.maximum_formula_length:
                break

            # 3b. Score every valid composition.
            candidate_scores = score_compositions(
                neurons,
                queue,
                packed_features,
                retained_ids,
                retained_valid,
                active,
            )

            # 3c. Select the next queue and record its ancestry.
            queue, formulas = prune_level(
                candidate_scores,
                queue,
                packed_features,
                retained_ids,
                formulas,
                formula_length,
                config.beam_size,
            )
            formula_length += 1

        # 4. Reconstruct only the winning formulas.
        results.extend(
            build_results(
                neuron_indices,
                best,
                feature_formulas,
                formulas,
            )
        )

    return results


def search_all(
    activation_vectors: torch.Tensor,
    feature_vectors: list[tuple[Symbol, np.ndarray]],
    num_workers: int = 1,
    device=None,
    config: SearchConfig | None = None,
) -> list[SearchResult]:
    activation_vectors = activation_vectors.numpy()
    if activation_vectors.ndim != 2:
        raise ValueError(
            "Activation vectors must have shape [examples, neurons]."
        )

    neuron_count = activation_vectors.shape[1]
    assert neuron_count >= 0

    config = config or SearchConfig()

    if num_workers <= 1:
        return search_worker(
            activation_vectors,
            list(range(neuron_count)),
            feature_vectors,
            device,
            config,
        )

    worker_count = min(num_workers, neuron_count)
    batch_size, remainder = divmod(neuron_count, worker_count)
    worker_arguments = []
    start = 0
    for worker_index in range(worker_count):
        stop = start + batch_size
        if worker_index < remainder:
            stop += 1
        worker_arguments.append(
            (
                activation_vectors[:, start:stop],
                list(range(start, stop)),
                feature_vectors,
                device,
                config,
            )
        )
        start = stop

    context = mp.get_context("spawn")
    with context.Pool(processes=len(worker_arguments)) as pool:
        result_chunks = pool.starmap(search_worker, worker_arguments)

    return [
        result
        for result_chunk in result_chunks
        for result in result_chunk
    ]
