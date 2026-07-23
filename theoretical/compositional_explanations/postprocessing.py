"""Threshold and prune captured activations for compositional analysis."""
import torch


def threshold(activation, alpha=None):
    # shape = activation.shape  # [example count x activation size]
    if alpha is None:
        return (activation > 0).to(torch.int8)

    thresholds = torch.quantile(activation, 1 - alpha, dim=0)
    return (activation > thresholds).to(torch.int8)


def prune_min_acts(binary_acts, min_acts=500):
    frequency = binary_acts.sum(dim=0)
    keep_mask = frequency >= min_acts
    neuron_ids = torch.nonzero(keep_mask, as_tuple=False).flatten()
    return binary_acts[:, keep_mask], neuron_ids
