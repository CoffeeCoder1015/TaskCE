"""Prepare captured activations for compositional search."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class SearchableActivations:
    """A retained activation matrix paired with its original neuron IDs."""

    matrix: torch.Tensor
    neuron_ids: tuple[int, ...]


def threshold(
    activation: torch.Tensor,
    alpha: float | None = None,
) -> torch.Tensor:
    """Convert continuous activations to binary example-by-neuron vectors."""
    if alpha is None:
        return activation > 0

    thresholds = torch.quantile(activation, 1 - alpha, dim=0)
    return activation > thresholds


def prune_min_acts(
    binary_activations: torch.Tensor,
    min_acts: int = 500,
) -> SearchableActivations:
    """Remove infrequently active neurons while retaining their identities."""
    keep_mask = binary_activations.sum(dim=0) >= min_acts
    neuron_ids = torch.nonzero(
        keep_mask,
        as_tuple=False,
    ).flatten()

    return SearchableActivations(
        matrix=binary_activations[:, keep_mask],
        neuron_ids=tuple(int(value) for value in neuron_ids.tolist()),
    )
