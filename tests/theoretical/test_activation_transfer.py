import numpy as np
import pytest

from theoretical.activation_transfer.analysis import (
    calc_bidirectional_top_k_connections,
    calc_cross_similarity,
    calc_transport_latents,
)


def test_cross_similarity_uses_neurons_by_examples_orientation():
    base = np.array([[1.0, 0.0, -1.0], [1.0, 2.0, 3.0]])
    finetuned = np.array([[2.0, 0.0, -2.0], [-1.0, -2.0, -3.0]])

    pearson, cosine = calc_cross_similarity(base, finetuned)

    assert pearson.shape == (2, 2)
    assert cosine.shape == (2, 2)
    np.testing.assert_allclose(np.diag(pearson), [1.0, -1.0])
    np.testing.assert_allclose(np.diag(cosine), [1.0, -1.0])


def test_transport_latents_reconstruct_truncated_svd_product():
    matrix = np.array([[3.0, 0.0], [0.0, 2.0]])
    U, singular_values, Vt = np.linalg.svd(matrix)

    base, finetuned = calc_transport_latents(U, singular_values, Vt, rank=2)

    np.testing.assert_allclose(base @ finetuned.T, matrix)


def test_bidirectional_top_k_separates_correlation_and_anticorrelation():
    affinity = np.array([[0.8, -0.9], [0.3, -0.2]])

    forward_positive, _ = calc_bidirectional_top_k_connections(
        affinity, 1, "correlation"
    )
    forward_negative, _ = calc_bidirectional_top_k_connections(
        affinity, 1, "anticorrelation"
    )

    np.testing.assert_array_equal(forward_positive[1], [0, 0])
    np.testing.assert_array_equal(forward_negative[1], [1, 1])


def test_cross_similarity_requires_shared_example_axis():
    with pytest.raises(ValueError, match="share their example axis"):
        calc_cross_similarity(np.zeros((2, 3)), np.zeros((2, 4)))

