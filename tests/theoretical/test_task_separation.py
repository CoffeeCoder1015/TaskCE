import numpy as np
import pytest

pytest.importorskip("sklearn", exc_type=ImportError)

from theoretical.activations.task_separation.analysis import (
    analyze_task_separation,
)


def test_task_separation_returns_projection_data_before_plotting():
    states = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0],
        ]
    )
    labels = ["left", "left", "right", "right"]

    result = analyze_task_separation(states, labels)

    assert set(result) == {
        "points",
        "labels",
        "label_counts",
        "explained_variance_ratio",
        "components",
    }
    assert result["points"].shape == (4, 2)
    assert result["components"].shape == (2, 3)
    assert result["label_counts"] == {"left": 2, "right": 2}


def test_task_separation_requires_aligned_labels():
    with pytest.raises(ValueError, match="dataset rows do not match"):
        analyze_task_separation(
            np.zeros((3, 2)),
            ["first", "second"],
        )
