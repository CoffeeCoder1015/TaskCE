import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from theoretical.activations.threshold_coverage.analysis import (
    analyze_threshold_coverage,
)


def test_threshold_coverage_constructs_selected_counts_and_sweep():
    values = np.array(
        [
            [0.0, 4.0],
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
        ]
    )

    result = analyze_threshold_coverage(
        values,
        task_name="demo",
        alpha=0.5,
        min_acts=2,
        alpha_candidates=[0.25, 0.5],
    )

    assert set(result) == {"selected", "sweep"}
    assert result["selected"]["counts"] == [2, 2]
    assert result["selected"]["summary"]["kept_count"] == 2
    assert [record["alpha"] for record in result["sweep"]["records"]] == [
        0.25,
        0.5,
    ]


def test_threshold_coverage_result_is_json_ready():
    result = analyze_threshold_coverage(
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        task_name="demo",
        alpha=0.5,
        min_acts=1,
    )

    assert json.loads(json.dumps(result)) == result


def test_threshold_coverage_rejects_invalid_selected_alpha():
    with pytest.raises(ValueError, match="between 0 and 1"):
        analyze_threshold_coverage(
            np.zeros((3, 2)),
            task_name="demo",
            alpha=1.0,
            min_acts=1,
        )
