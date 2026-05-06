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


def append_sample(samples, candidate, beam_size):
    samples.append((candidate.score, candidate.formula.flatten()))
    samples.sort(key=lambda sample: sample[0], reverse=True)
    del samples[beam_size:]


def candidate_for(formula, mask, neuron, complexity_penalty):
    raw_score = iou(mask, neuron)
    return BeamCandidate(
        score=(complexity_penalty ** (len(formula) - 1)) * raw_score,
        formula=formula,
        mask=mask,
    )


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
    device=None,
):
    device = resolve_device(device)
    feature_vectors = prepare_feature_vectors(feature_vectors, device)
    activation_vectors = to_binary_tensor(activation_vectors, device)
    results = []

    for activation_index in range(activation_vectors.shape[1]):
        neuron = activation_vectors[:, activation_index]
        samples = []

        leaf_candidates = [
            candidate_for(formula, mask, neuron, complexity_penalty)
            for formula, mask in feature_vectors
        ]
        nonzero_features = [
            (candidate.formula, candidate.mask)
            for candidate in leaf_candidates
            if candidate.score > 0
        ]

        beam = top_candidates(leaf_candidates, beam_size)
        best_noncomp = beam[0] if beam else None
        for candidate in beam:
            append_sample(samples, candidate, beam_size)

        for _ in range(formula_length - 1):
            candidates_by_formula = {
                candidate.formula: candidate
                for candidate in beam
            }

            for candidate in beam:
                for feature_formula, feature_mask in nonzero_features:
                    for formula, mask in expand_candidate(
                        candidate,
                        feature_formula,
                        feature_mask,
                    ):
                        new_candidate = candidate_for(
                            formula,
                            mask,
                            neuron,
                            complexity_penalty,
                        )
                        add_best_candidate(candidates_by_formula, new_candidate)

            beam = top_candidates(candidates_by_formula.values(), beam_size)
            for candidate in beam:
                append_sample(samples, candidate, beam_size)

        best = beam[0] if beam else None
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
