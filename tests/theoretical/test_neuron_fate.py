import numpy as np
import pytest

from theoretical.activations.neuron_fate.analysis import (
    analyze_neuron_fate,
    calc_bidirectional_top_k_connections,
    calc_transport_latents,
    identity_mse,
    matrix_ccc,
    matrix_regression,
)


def test_matrix_regression_recovers_exact_linear_relationships():
    base = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 2.0],
            [3.0, 1.0],
        ]
    )
    finetuned = base * np.array([2.0, -3.0]) + np.array([1.0, 4.0])

    result = matrix_regression(base, finetuned)

    np.testing.assert_allclose(np.diag(result["r_squared"]), np.ones(2))
    np.testing.assert_allclose(np.diag(result["alpha"]), [2.0, -3.0])
    np.testing.assert_allclose(np.diag(result["beta"]), [1.0, 4.0])


def test_identity_mse_and_ccc_measure_every_base_fine_pair():
    base = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    finetuned = base + np.array([1.0, -1.0])

    expected_mse = np.mean(
        (base[:, :, None] - finetuned[:, None, :]) ** 2,
        axis=0,
    )

    np.testing.assert_allclose(identity_mse(base, finetuned), expected_mse)
    assert matrix_ccc(base, finetuned).shape == (2, 2)


def test_neuron_fate_result_is_complete_before_persistence():
    base = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 2.0], [3.0, 5.0]])
    finetuned = base + np.array([0.5, -0.5])

    result = analyze_neuron_fate(base, finetuned)

    assert set(result) == {"affine", "literal", "relative_to_base"}
    assert set(result["affine"]) == {"r_squared", "alpha", "beta"}
    assert set(result["literal"]) == {"ccc", "identity_mse"}
    assert set(result["relative_to_base"]) == {
        "base_ccc",
        "cross_minus_base_ccc",
    }
    assert all(
        matrix.shape == (2, 2)
        for measurements in result.values()
        for matrix in measurements.values()
    )


def test_transport_latents_reconstruct_truncated_fate_matrix():
    matrix = np.array([[3.0, 0.0], [0.0, 2.0]])
    U, singular_values, Vt = np.linalg.svd(matrix)

    base, finetuned = calc_transport_latents(U, singular_values, Vt, rank=2)

    np.testing.assert_allclose(np.matmul(base, finetuned.T), matrix)


def test_bidirectional_top_k_separates_positive_and_negative_fate_edges():
    affinity = np.array([[0.8, -0.9], [0.3, -0.2]])

    forward_positive, _ = calc_bidirectional_top_k_connections(
        affinity,
        1,
        "positive",
    )
    forward_negative, _ = calc_bidirectional_top_k_connections(
        affinity,
        1,
        "negative",
    )

    np.testing.assert_array_equal(forward_positive[1], [0, 0])
    np.testing.assert_array_equal(forward_negative[1], [1, 1])


def test_neuron_fate_requires_corresponding_neuron_spaces():
    with pytest.raises(ValueError, match="same neuron count"):
        analyze_neuron_fate(np.zeros((3, 2)), np.zeros((3, 4)))
