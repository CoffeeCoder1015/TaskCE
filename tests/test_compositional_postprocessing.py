import pytest

torch = pytest.importorskip("torch")

from theoretical.compositional_explanations.postprocessing import (
    prune_min_acts,
    threshold,
)


def test_threshold_uses_per_neuron_quantiles():
    activations = torch.tensor(
        [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
        ]
    )

    binary = threshold(activations, alpha=0.5)

    assert binary.dtype == torch.bool
    assert binary.tolist() == [
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
    ]


def test_pruning_retains_kept_and_removed_neuron_identities():
    binary_activations = torch.tensor(
        [
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=torch.int8,
    )

    result = prune_min_acts(binary_activations, min_acts=2)

    assert result.neuron_ids == (0, 2)
    assert torch.equal(result.matrix, binary_activations[:, [0, 2]])
