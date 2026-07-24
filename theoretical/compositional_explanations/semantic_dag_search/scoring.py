"""Pairwise count scoring for semantic-DAG search."""

import torch


AND_OPERATION = 0
OR_OPERATION = 1
LEFT_AND_NOT_RIGHT_OPERATION = 2
RIGHT_AND_NOT_LEFT_OPERATION = 3
OPERATION_COUNT = 4


def intersection_over_union_from_counts(
    intersections: torch.Tensor,
    candidate_sizes: torch.Tensor,
    target_sizes: torch.Tensor,
) -> torch.Tensor:
    """Calculate IoU from broadcast-compatible set cardinalities."""
    unions = candidate_sizes + target_sizes - intersections
    scores = intersections / unions.clamp_min(1)
    return torch.where(unions > 0, scores, torch.zeros_like(scores))


def pair_iou_scores(
    *,
    target_size: torch.Tensor,
    left_sizes: torch.Tensor,
    left_target_sizes: torch.Tensor,
    right_sizes: torch.Tensor,
    right_target_sizes: torch.Tensor,
    pair_sizes: torch.Tensor,
    pair_target_sizes: torch.Tensor,
) -> torch.Tensor:
    """Score every Boolean composition of two semantic-state collections."""
    left_sizes = left_sizes[:, None]
    left_target_sizes = left_target_sizes[:, None]
    right_sizes = right_sizes[None, :]
    right_target_sizes = right_target_sizes[None, :]

    and_sizes = pair_sizes
    and_targets = pair_target_sizes
    and_valid = (
        (and_sizes > 0)
        & (and_sizes < left_sizes)
        & (and_sizes < right_sizes)
    )

    or_sizes = left_sizes + right_sizes - pair_sizes
    or_targets = (
        left_target_sizes
        + right_target_sizes
        - pair_target_sizes
    )
    or_valid = (
        (pair_sizes < left_sizes)
        & (pair_sizes < right_sizes)
    )

    left_difference_sizes = left_sizes - pair_sizes
    left_difference_targets = left_target_sizes - pair_target_sizes
    left_difference_valid = (
        (left_difference_sizes > 0)
        & (pair_sizes > 0)
    )

    right_difference_sizes = right_sizes - pair_sizes
    right_difference_targets = right_target_sizes - pair_target_sizes
    right_difference_valid = (
        (right_difference_sizes > 0)
        & (pair_sizes > 0)
    )

    candidate_sizes = torch.stack(
        (
            and_sizes,
            or_sizes,
            left_difference_sizes,
            right_difference_sizes,
        ),
        dim=-1,
    )
    candidate_targets = torch.stack(
        (
            and_targets,
            or_targets,
            left_difference_targets,
            right_difference_targets,
        ),
        dim=-1,
    )
    valid = torch.stack(
        (
            and_valid,
            or_valid,
            left_difference_valid,
            right_difference_valid,
        ),
        dim=-1,
    )
    scores = intersection_over_union_from_counts(
        candidate_targets,
        candidate_sizes,
        target_size,
    )
    return scores.masked_fill(~valid, -torch.inf)
