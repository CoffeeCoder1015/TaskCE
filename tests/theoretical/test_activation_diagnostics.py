import numpy as np
import pytest

pytest.importorskip("torch")

from theoretical.activation_diagnostics.analysis import (
    compare_raw_activations,
    raw_activation_correlation_matrix,
    raw_activation_cosine_similarity_matrix,
)


def test_raw_activation_metrics_preserve_constant_and_zero_column_behavior():
    values = np.array(
        [
            [1.0, 10.0, 5.0, 0.0],
            [2.0, 20.0, 5.0, 0.0],
            [3.0, 30.0, 5.0, 0.0],
        ]
    )

    correlation = raw_activation_correlation_matrix(values)
    cosine = raw_activation_cosine_similarity_matrix(values)

    np.testing.assert_allclose(correlation[:2, :2], np.ones((2, 2)))
    np.testing.assert_allclose(correlation[2, :2], np.zeros(2))
    np.testing.assert_allclose(np.diag(correlation), np.ones(4))
    np.testing.assert_allclose(cosine[:2, :2], np.ones((2, 2)))
    np.testing.assert_allclose(cosine[3, :3], np.zeros(3))
    np.testing.assert_allclose(np.diag(cosine), np.ones(4))


def test_compare_raw_activations_returns_existing_matrix_views():
    base = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    finetuned = np.array([[0.0, 4.0], [2.0, 2.0], [4.0, 0.0]])

    result = compare_raw_activations(base, finetuned)

    assert set(result) == {"pearson", "cosine"}
    for metric in result.values():
        assert metric["combined"].shape == (4, 4)
        assert metric["base_to_finetuned"].shape == (2, 2)
        assert metric["same_neuron_base_to_finetuned"].shape == (2,)


def test_compare_raw_activations_rejects_unaligned_data():
    with pytest.raises(ValueError, match="must have the same shape"):
        compare_raw_activations(np.zeros((3, 2)), np.zeros((4, 2)))
