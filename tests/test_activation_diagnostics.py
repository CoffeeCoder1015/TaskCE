import json
import os

import pytest
import torch

from analysis.activation_diagnostics import (
    local_alpha_candidates,
    save_binary_activation_count_diagnostics,
    save_raw_activation_alpha_diagnostics,
)


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def test_binary_activation_count_diagnostics_writes_metrics_and_plots(tmp_path):
    binary_acts = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=torch.int8,
    )

    paths = save_binary_activation_count_diagnostics(
        binary_acts,
        min_acts=3,
        output_dir=tmp_path,
    )

    summary = load_json(paths["summary"])
    assert summary["min_acts"] == 3
    assert summary["zero_activation_count"] == 2
    assert summary["zero_activation_percent"] == 50.0
    assert summary["below_min_acts_count"] == 2
    assert summary["below_min_acts_percent"] == 50.0
    assert summary["kept_count"] == 2
    assert summary["kept_percent"] == 50.0
    assert summary["activation_count_percentiles"]["max"] == 3.0

    assert os.path.getsize(paths["hist_full"]) > 0
    assert os.path.getsize(paths["hist_nonzero"]) > 0


def test_raw_activation_alpha_diagnostics_sweeps_candidates_and_plots(tmp_path):
    raw_acts = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [2.0, 0.0, 5.0],
            [3.0, 0.0, 5.0],
        ]
    )

    paths = save_raw_activation_alpha_diagnostics(
        raw_acts,
        alpha=0.5,
        min_acts=2,
        output_dir=tmp_path,
        alpha_candidates=[0.5],
    )

    summary = load_json(paths["alpha_sweep"])
    assert summary["min_acts"] == 2
    assert len(summary["alpha_sweep"]) == 1

    record = summary["alpha_sweep"][0]
    assert record["alpha"] == 0.5
    assert record["zero_activation_count"] == 2
    assert record["below_min_acts_count"] == 2
    assert record["kept_count"] == 1
    assert record["kept_percent"] == pytest.approx(100 / 3)
    assert record["activation_count_percentiles"]["max"] == 2.0

    assert os.path.getsize(paths["threshold_hist"]) > 0
    assert os.path.getsize(paths["correlation_heatmap"]) > 0


def test_local_alpha_candidates_are_clipped_and_deduplicated():
    assert local_alpha_candidates(0.05) == [0.05, 0.1, 0.15]
    assert local_alpha_candidates(0.95) == [0.85, 0.9, 0.95]
