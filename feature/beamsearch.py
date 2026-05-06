from dataclasses import dataclass

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
    samples: list[tuple[float, str]]


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


def top_candidates(candidates, beam_size):
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:beam_size]


def pop_best_candidate(candidates_by_formula):
    candidate = top_candidates(candidates_by_formula.values(), 1)[0]
    del candidates_by_formula[candidate.formula]
    return candidate


def trim_candidate_queue(candidates_by_formula, beam_size):
    return {
        candidate.formula: candidate
        for candidate in top_candidates(candidates_by_formula.values(), beam_size)
    }


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


def add_best_candidate(candidates_by_formula, candidate):
    existing = candidates_by_formula.get(candidate.formula)
    if existing is None or candidate.score > existing.score:
        candidates_by_formula[candidate.formula] = candidate


def expand_candidate(candidate, feature_formula, feature_mask):
    formula = candidate.formula
    mask = candidate.mask
    return [
        (And(left=formula, right=feature_formula), mask & feature_mask),
        (Or(left=formula, right=feature_formula), mask | feature_mask),
        (And(left=formula, right=Not(feature_formula)), mask & ~feature_mask),
    ]


def beamsearch_all(
    feature_vectors,
    activation_vectors,
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

    for activation_index in range(activation_vectors.shape[1]):
        neuron = activation_vectors[:, activation_index]

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

        best_candidates_by_formula = {}
        for candidate in leaf_candidates:
            add_best_candidate(best_candidates_by_formula, candidate)

        queue_by_formula = trim_candidate_queue(
            best_candidates_by_formula,
            beam_size,
        )
        best_noncomp = top_candidates(leaf_candidates, 1)[0] if leaf_candidates else None
        expanded_formulas = set()
        expansions = 0

        while queue_by_formula and expansions < max_expansions:
            candidate = pop_best_candidate(queue_by_formula)
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
                add_best_candidate(best_candidates_by_formula, new_candidate)
                if new_candidate.formula not in expanded_formulas:
                    add_best_candidate(queue_by_formula, new_candidate)

            queue_by_formula = trim_candidate_queue(queue_by_formula, beam_size)
            expansions += 1

        best_candidates = top_candidates(best_candidates_by_formula.values(), beam_size)
        best = best_candidates[0] if best_candidates else None
        samples = [
            (candidate.score, candidate.formula.flatten())
            for candidate in best_candidates
        ]
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
                samples=samples,
            )
        )

    return results


# Backward-compatible name used in older notes.
IoU = iou
