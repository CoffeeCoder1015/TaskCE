"""Dense Boolean LevelSearch with mixed-neuron numerical batches.

CPU ranking retains stable best-K candidates in compact, device-owned storage.
Parent visits use descending raw IoU and encounter order, so tied explanations
can differ from legacy LevelSearch's incidental heap-array traversal.
"""

import heapq
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import batched, count, product, zip_longest

import torch
from sympy import Expr, Symbol
from sympy.logic import And, Not, Or

from .algorithm import SearchConfig, length_penalty_factor
from .formula import logic_str


@dataclass(frozen=True, eq=False)
class Candidate:
    """A CPU formula and raw IoU bound to its actual device mask.

    Freezing the record does not make the tensor contents immutable.
    """

    formula: Expr
    vector: torch.Tensor
    iou: float


class BoundedPriorityQueue:
    """Stable best-K since the previous drain, ranked on the CPU.

    Each bulk push copies survivors into one compact device allocation; offered
    temporary mask storage is not retained. Masks must have matching shapes and
    devices. Inputs use the ordinary current stream and must not be mutated
    concurrently. Higher raw IoU wins; ties favor earlier encounters.
    """

    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("queue capacity must be nonnegative")
        self.capacity = capacity
        self._encounter_ids = count()
        self._records: list[tuple[int, Candidate]] = []

    def __len__(self) -> int:
        return len(self._records)

    def push_many(self, records: Iterable[Candidate]) -> None:
        incoming = [(next(self._encounter_ids), record) for record in records]
        if not incoming or self.capacity == 0:
            return
        selected = heapq.nsmallest(
            self.capacity,
            self._records + incoming,
            key=lambda item: (-item[1].iou, item[0]),
        )
        masks = torch.stack([record.vector for _, record in selected])
        self._records = [
            (encounter_id, Candidate(record.formula, mask, record.iou))
            for (encounter_id, record), mask in zip(selected, masks, strict=True)
        ]

    def drain(self) -> list[Candidate]:
        """Return best-first records, transferring their storage to the caller."""
        records = [record for _, record in self._records]
        self._records = []
        return records


class SetPartition:
    """Three disjoint regions of matching Boolean batches, covering their union."""

    def __init__(self, left: torch.Tensor, right: torch.Tensor):
        self.shared = left & right
        self.left_only = left & ~right
        self.right_only = right & ~left

    def union(self) -> torch.Tensor:
        return self.shared | self.left_only | self.right_only


def intersection_over_union(
    masks: torch.Tensor, targets: torch.Tensor,
) -> torch.Tensor:
    """Raw float32 IoU over the last axis; leading axes broadcast.

    Empty unions have score zero. This operation performs no CPU transfer.
    """
    intersections = (masks & targets).sum(dim=-1)
    unions = (masks | targets).sum(dim=-1)
    return intersections.float() / unions.clamp_min(1).float()


@dataclass(eq=False)
class NeuronSearch:
    """One neuron's state, with identity ownership for grouped offers."""

    target: torch.Tensor
    retained_features: list[Candidate]
    beam: BoundedPriorityQueue
    score_track: dict[Expr, float]
    best_formula: str = ""
    best_score: float = 0.0


def advance_level(searches: list[NeuronSearch], config: SearchConfig) -> None:
    """Visit parents, then bulk-offer improving children to each bounded beam."""
    pair_streams = []
    for search in searches:
        expandable = []
        for parent in search.beam.drain():
            score = parent.iou * length_penalty_factor(
                parent.formula, config.length_penalty,
            )
            if score > search.best_score:
                search.best_formula = logic_str(parent.formula)
                search.best_score = score
            if parent.formula.count(Symbol) < config.maximum_formula_length:
                expandable.append(parent)
        # product binds this search immediately, without a late-bound closure.
        pair_streams.append(product((search,), expandable, search.retained_features))

    pairs = (
        pair
        for row in zip_longest(*pair_streams)
        for pair in row
        if pair is not None
    )
    for batch in batched(pairs, config.iou_calculation_batch_size):
        targets = torch.stack([search.target for search, parent, concept in batch])
        parents = torch.stack([parent.vector for search, parent, concept in batch])
        concepts = torch.stack([concept.vector for search, parent, concept in batch])
        partition = SetPartition(parents, concepts)
        shared = partition.shared.any(dim=-1)
        left_only = partition.left_only.any(dim=-1)
        right_only = partition.right_only.any(dim=-1)
        admitted = torch.stack((
            shared & left_only & right_only,
            left_only & right_only,
            shared & left_only,
        ), dim=1)
        # Materialize all three operations once, with adjacent operation rows.
        compositions = torch.stack((
            partition.shared, partition.union(), partition.left_only,
        ), dim=1)
        scores = intersection_over_union(compositions, targets[:, None, :])
        scores = scores.masked_fill(~admitted, -1.0).cpu().tolist()

        offers: dict[NeuronSearch, list[Candidate]] = {}
        for (search, parent, concept), pair_scores, pair_vectors in zip(
            batch, scores, compositions, strict=True,
        ):
            for operation, iou, vector in zip(
                ("AND", "OR", "AND NOT"), pair_scores, pair_vectors, strict=True,
            ):
                if iou < 0:
                    continue
                if operation == "AND":
                    formula = And(parent.formula, concept.formula)
                elif operation == "OR":
                    formula = Or(parent.formula, concept.formula)
                else:
                    if parent.formula == concept.formula:
                        continue
                    formula = And(parent.formula, Not(concept.formula))
                score = iou * length_penalty_factor(formula, config.length_penalty)
                if score > search.score_track.get(formula, 0):
                    # History includes every improvement, even a later top-K loss.
                    search.score_track[formula] = score
                    offers.setdefault(search, []).append(Candidate(formula, vector, iou))
        while offers:
            search, records = offers.popitem()
            search.beam.push_many(records)
            del records
        # Release offered row views before allocating the next numerical chunk.
        del vector, pair_vectors, compositions, partition
        del targets, parents, concepts, admitted, shared, left_only, right_only


def batched_level_search(
    neurons: torch.Tensor,
    feature_vectors: list[tuple[Expr, torch.Tensor]],
    config: SearchConfig | None = None,
) -> list[tuple[str, float]]:
    """Search Boolean neurons [N,E] against feature masks [E] on one device.

    Atomic scoring and composition chunks mix neurons. As in LevelSearch, final
    generated children are not visited when the iteration budget is exhausted.
    Empty feature sets yield empty explanations.
    """
    config = config if config is not None else SearchConfig()
    if config.iou_calculation_batch_size <= 0:
        raise ValueError("iou_calculation_batch_size must be positive")
    if config.beam_size < 0 or config.max_iterations < 0:
        raise ValueError("beam_size and max_iterations must be nonnegative")
    searches = [
        NeuronSearch(target, [], BoundedPriorityQueue(config.beam_size), {})
        for target in neurons
    ]
    # Feature-major traversal preserves feature encounter order for each neuron.
    for batch in batched(
        product(feature_vectors, searches), config.iou_calculation_batch_size,
    ):
        masks = torch.stack([feature[1] for feature, search in batch])
        targets = torch.stack([search.target for feature, search in batch])
        scores = intersection_over_union(masks, targets).cpu().tolist()
        for ((formula, vector), search), iou in zip(batch, scores, strict=True):
            if iou > 0:
                search.retained_features.append(Candidate(formula, vector, iou))
        del masks, targets

    for search in searches:
        search.retained_features.sort(key=lambda candidate: candidate.iou, reverse=True)
        for candidate in search.retained_features:
            prior = search.score_track.get(candidate.formula, 0)
            search.score_track[candidate.formula] = max(prior, candidate.iou)
        search.beam.push_many(search.retained_features)
    for _ in range(config.max_iterations):
        active = [search for search in searches if search.beam]
        if not active:
            break
        advance_level(active, config)
    return [(search.best_formula, search.best_score) for search in searches]
