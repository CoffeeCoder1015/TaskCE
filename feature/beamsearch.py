from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import heapq
import multiprocessing as mp

import torch

from feature.formula import And, Not, Or


@dataclass(frozen=True)
class BeamCandidate:
    score: float
    formula: object
    mask: torch.Tensor


@dataclass(frozen=True)
class BeamSearchResult:
    activation_index: int
    best: tuple[str, float]
    best_noncomp: tuple[str, float]


def iou(v1, v2):
    intersection = (v1 & v2).sum()
    union = (v1 | v2).sum()
    if union.item() == 0:
        return 0.0
    return (intersection.float() / union.float()).item()


def batched_iou(masks, neuron):
    intersections = (masks & neuron).sum(dim=1)
    unions = (masks | neuron).sum(dim=1)
    raw_scores = intersections.float() / unions.clamp_min(1).float()
    return torch.where(unions > 0, raw_scores, torch.zeros_like(raw_scores))


def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_binary_tensor(value, device):
    return torch.as_tensor(value, dtype=torch.bool, device=device)


def prepare_feature_vectors(feature_vectors, device):
    return [
        (formula, to_binary_tensor(binary_vector, device))
        for formula, binary_vector in feature_vectors
    ]


def activation_index_chunks(total_activations, num_workers):
    if total_activations == 0:
        return []
    worker_count = min(max(1, num_workers), total_activations)
    chunk_size, remainder = divmod(total_activations, worker_count)
    chunks = []
    start = 0
    for worker_index in range(worker_count):
        stop = start + chunk_size + (1 if worker_index < remainder else 0)
        chunks.append((start, stop))
        start = stop
    return chunks


def score_formula_masks(formula_masks, neuron, complexity_penalty, score_batch_size):
    candidates = []
    for start in range(0, len(formula_masks), score_batch_size):
        batch = formula_masks[start:start + score_batch_size]
        formulas = [formula for formula, _ in batch]
        masks = torch.stack([mask for _, mask in batch])
        raw_scores = batched_iou(masks, neuron)
        penalties = torch.tensor(
            [
                complexity_penalty ** (len(formula) - 1)
                for formula in formulas
            ],
            dtype=raw_scores.dtype,
            device=raw_scores.device,
        )
        scores = (raw_scores * penalties).tolist()
        candidates.extend(
            BeamCandidate(score=score, formula=formula, mask=mask)
            for score, formula, (_, mask) in zip(scores, formulas, batch)
        )
    return candidates


def expand_candidate(candidate, feature_formula, feature_mask):
    formula = candidate.formula
    mask = candidate.mask
    return [
        (And(left=formula, right=feature_formula), mask & feature_mask),
        (Or(left=formula, right=feature_formula), mask | feature_mask),
        (And(left=formula, right=Not(feature_formula)), mask & ~feature_mask),
    ]


def beamsearch_chunk(
    feature_vectors,
    activation_vectors,
    activation_indices,
    beam_size=5,
    formula_length=5,
    complexity_penalty=1.0,
    score_batch_size=4096,
    max_expansions=None,
    device=None,
):
    device = resolve_device(device)
    feature_vectors = prepare_feature_vectors(feature_vectors, device)
    activation_vectors = to_binary_tensor(activation_vectors, device)
    if max_expansions is None:
        max_expansions = beam_size * max(0, formula_length - 1)
    results = []

    for local_activation_index, activation_index in enumerate(activation_indices):
        neuron = activation_vectors[:, local_activation_index]

        leaf_candidates = score_formula_masks(
            feature_vectors,
            neuron,
            complexity_penalty,
            score_batch_size,
        )
        nonzero_features = [
            (candidate.formula, candidate.mask)
            for candidate in leaf_candidates
            if candidate.score > 0
        ]

        best = max(leaf_candidates, key=lambda candidate: candidate.score, default=None)
        best_noncomp = best
        frontier = []
        queued_formulas = set()
        expanded_formulas = set()
        expansions = 0
        next_queue_id = 0

        for candidate in leaf_candidates:
            heapq.heappush(frontier, (-candidate.score, next_queue_id, candidate))
            queued_formulas.add(candidate.formula)
            next_queue_id += 1
        if len(frontier) > beam_size:
            frontier = heapq.nsmallest(beam_size, frontier)
            heapq.heapify(frontier)
            queued_formulas = {entry[2].formula for entry in frontier}

        while frontier and expansions < max_expansions:
            _, _, candidate = heapq.heappop(frontier)
            queued_formulas.discard(candidate.formula)
            if candidate.formula in expanded_formulas:
                continue
            expanded_formulas.add(candidate.formula)

            if len(candidate.formula) >= formula_length:
                continue

            pending_formula_masks = []
            for feature_formula, feature_mask in nonzero_features:
                for formula, mask in expand_candidate(
                    candidate,
                    feature_formula,
                    feature_mask,
                ):
                    pending_formula_masks.append((formula, mask))

            for new_candidate in score_formula_masks(
                pending_formula_masks,
                neuron,
                complexity_penalty,
                score_batch_size,
            ):
                if best is None or new_candidate.score > best.score:
                    best = new_candidate
                if (
                    new_candidate.formula not in expanded_formulas
                    and new_candidate.formula not in queued_formulas
                ):
                    heapq.heappush(
                        frontier,
                        (-new_candidate.score, next_queue_id, new_candidate),
                    )
                    queued_formulas.add(new_candidate.formula)
                    next_queue_id += 1

            if len(frontier) > beam_size:
                frontier = heapq.nsmallest(beam_size, frontier)
                heapq.heapify(frontier)
                queued_formulas = {entry[2].formula for entry in frontier}
            expansions += 1

        if best is not None:
            print(
                f"Neuron {activation_index}: "
                f"best_iou={best.score:.4f} "
                f"best={best.formula.flatten()}"
            )
        results.append(
            BeamSearchResult(
                activation_index=activation_index,
                best=(
                    best.formula.flatten(),
                    best.score,
                ) if best else ("", 0.0),
                best_noncomp=(
                    best_noncomp.formula.flatten(),
                    best_noncomp.score,
                ) if best_noncomp else ("", 0.0),
            )
        )

    return results


def beamsearch_worker(args):
    return beamsearch_chunk(*args)


def beamsearch_all(
    feature_vectors,
    activation_vectors,
    beam_size=5,
    formula_length=5,
    complexity_penalty=1.0,
    score_batch_size=4096,
    max_expansions=None,
    num_workers=1,
    device=None,
):
    if max_expansions is None:
        max_expansions = beam_size * max(0, formula_length - 1)

    if num_workers <= 1:
        activation_vectors = torch.as_tensor(activation_vectors)
        return beamsearch_chunk(
            feature_vectors,
            activation_vectors,
            range(activation_vectors.shape[1]),
            beam_size,
            formula_length,
            complexity_penalty,
            score_batch_size,
            max_expansions,
            device,
        )

    activation_vectors = to_binary_tensor(activation_vectors, torch.device("cpu"))
    feature_vectors = prepare_feature_vectors(feature_vectors, torch.device("cpu"))
    chunks = activation_index_chunks(activation_vectors.shape[1], num_workers)
    if not chunks:
        return []
    worker_args = [
        (
            feature_vectors,
            activation_vectors[:, start:stop],
            range(start, stop),
            beam_size,
            formula_length,
            complexity_penalty,
            score_batch_size,
            max_expansions,
            device,
        )
        for start, stop in chunks
    ]

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(worker_args),
        mp_context=context,
    ) as executor:
        result_chunks = list(executor.map(beamsearch_worker, worker_args))

    results = [
        result
        for chunk in result_chunks
        for result in chunk
    ]
    return sorted(results, key=lambda result: result.activation_index)


# Backward-compatible name used in older notes.
IoU = iou
