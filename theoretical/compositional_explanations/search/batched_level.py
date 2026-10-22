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
