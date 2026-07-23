import numpy as np

from theoretical.activation_regression.analysis import (
    identity_mse,
    matrix_ccc,
    matrix_regression,
    save_regression_artifacts,
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


def test_identity_mse_and_ccc_preserve_existing_shapes():
    base = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    finetuned = base + np.array([1.0, -1.0])

    np.testing.assert_allclose(identity_mse(base, finetuned), [1.0, 1.0])
    assert matrix_ccc(base, finetuned).shape == (2, 2)


def test_regression_artifacts_are_written_to_the_given_artifact_directory(tmp_path):
    base = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 2.0], [3.0, 5.0]])
    finetuned = base + np.array([0.5, -0.5])

    paths = save_regression_artifacts(
        base,
        finetuned,
        tmp_path,
        task_name="demo",
    )

    assert set(paths) == {
        "r_squared",
        "alpha",
        "beta",
        "top_r_squared",
        "ccc",
        "cross_minus_base_ccc",
        "identity_mse",
    }
    assert all(path.parent == tmp_path and path.stat().st_size > 0 for path in paths.values())

