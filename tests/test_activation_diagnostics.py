import os

import numpy as np
import pytest
import torch

from analysis.activation_diagnostics import (
    jaccard_similarity_matrix,
    local_alpha_candidates,
    raw_activation_correlation_matrix,
    raw_activation_cosine_similarity_matrix,
    save_binary_activation_count_diagnostics,
    save_raw_activation_alpha_diagnostics,
)


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

    report = paths["diagnostics_report"].read_text(encoding="utf-8")
    assert "POST-BINARIZATION ACTIVATION COUNT SUMMARY" in report
    assert "zero_activation_count: 2" in report
    assert "zero_activation_percent: 50.000000" in report
    assert "below_min_acts_count: 2" in report
    assert "below_min_acts_percent: 50.000000" in report
    assert "kept_count: 2" in report
    assert "kept_percent: 50.000000" in report
    assert "max: 3.000000" in report

    assert os.path.getsize(paths["hist_full"]) > 0
    assert os.path.getsize(paths["hist_nonzero"]) > 0
    assert os.path.getsize(paths["jaccard_heatmap"]) > 0


def test_raw_activation_alpha_diagnostics_sweeps_candidates_and_plots(tmp_path):
    raw_acts = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [2.0, 0.0, 5.0],
            [3.0, 0.0, 5.0],
        ]
    )
    raw_acts_base = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [1.0, 1.0, 5.0],
            [1.0, 0.0, 5.0],
            [2.0, 1.0, 5.0],
        ]
    )

    paths = save_raw_activation_alpha_diagnostics(
        raw_acts,
        alpha=0.5,
        min_acts=2,
        output_dir=tmp_path,
        raw_acts_base=raw_acts_base,
        alpha_candidates=[0.5],
        trace_neurons_per_plot=2,
        top_k=2,
    )

    report = paths["diagnostics_report"].read_text(encoding="utf-8")
    assert "RAW ACTIVATION ALPHA SWEEP" in report
    assert "min_acts: 2" in report
    assert "alpha=0.5" in report
    assert "zero_activation_count: 2" in report
    assert "below_min_acts_count: 2" in report
    assert "kept_count: 1" in report
    assert f"kept_percent: {100 / 3:.6f}" in report
    assert "max: 2.000000" in report
    assert "Pearson correlation base" in report
    assert "Pearson correlation finetuned" in report
    assert "Pearson correlation difference" in report
    assert "Cosine similarity base" in report
    assert "Cosine similarity finetuned" in report
    assert "Cosine similarity difference" in report
    assert "Top positive pairs" in report
    assert "Top increased pairs" in report

    assert os.path.getsize(paths["activation_traces"]) > 0
    assert os.path.getsize(paths["correlation_heatmap_base"]) > 0
    assert os.path.getsize(paths["correlation_heatmap_finetuned"]) > 0
    assert os.path.getsize(paths["correlation_heatmap_difference"]) > 0
    assert os.path.getsize(paths["cosine_similarity_heatmap_base"]) > 0
    assert os.path.getsize(paths["cosine_similarity_heatmap_finetuned"]) > 0
    assert os.path.getsize(paths["cosine_similarity_heatmap_difference"]) > 0


def test_local_alpha_candidates_are_clipped_and_deduplicated():
    assert local_alpha_candidates(0.05) == [0.05, 0.1, 0.15]
    assert local_alpha_candidates(0.95) == [0.85, 0.9, 0.95]


def test_raw_activation_cosine_similarity_matrix_normalizes_scale_and_handles_zero_columns():
    raw_acts = torch.tensor(
        [
            [1.0, 10.0, 0.0, 1.0],
            [2.0, 20.0, 0.0, 0.0],
            [3.0, 30.0, 0.0, -1.0],
        ]
    )

    cosine = raw_activation_cosine_similarity_matrix(raw_acts)

    np.testing.assert_allclose(
        cosine,
        np.array(
            [
                [1.0, 1.0, 0.0, -0.37796447],
                [1.0, 1.0, 0.0, -0.37796447],
                [0.0, 0.0, 1.0, 0.0],
                [-0.37796447, -0.37796447, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_raw_activation_correlation_matrix_normalizes_scale_and_handles_constants():
    raw_acts = torch.tensor(
        [
            [1.0, 10.0, 5.0],
            [2.0, 20.0, 5.0],
            [3.0, 30.0, 5.0],
        ]
    )

    correlation = raw_activation_correlation_matrix(raw_acts)

    np.testing.assert_allclose(
        correlation,
        np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def test_raw_activation_cosine_similarity_matrix_handles_empty_neuron_matrices():
    raw_acts = torch.empty((3, 0))

    cosine = raw_activation_cosine_similarity_matrix(raw_acts)

    assert cosine.shape == (0, 0)


def test_jaccard_similarity_matrix():
    binary_acts = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=torch.int8,
    )

    jaccard = jaccard_similarity_matrix(binary_acts)

    expected = np.array(
        [
            [1.0, 2/3, 0.0, 0.0],
            [2/3, 1.0, 0.0, 0.25],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.25, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(jaccard, expected)


def test_raw_activation_diagnostics_handles_zero_neuron_matrices(tmp_path):
    raw_acts = torch.empty((3, 0))

    paths = save_raw_activation_alpha_diagnostics(
        raw_acts,
        alpha=0.5,
        min_acts=2,
        output_dir=tmp_path,
        raw_acts_base=raw_acts,
        alpha_candidates=[0.5],
    )

    assert os.path.getsize(paths["activation_traces"]) > 0
    assert os.path.getsize(paths["correlation_heatmap_base"]) > 0
    assert os.path.getsize(paths["correlation_heatmap_finetuned"]) > 0
    assert os.path.getsize(paths["correlation_heatmap_difference"]) > 0
    assert os.path.getsize(paths["cosine_similarity_heatmap_base"]) > 0
    assert os.path.getsize(paths["cosine_similarity_heatmap_finetuned"]) > 0
    assert os.path.getsize(paths["cosine_similarity_heatmap_difference"]) > 0
    assert os.path.getsize(paths["diagnostics_report"]) > 0


def test_raw_activation_diagnostics_requires_base_activations(tmp_path):
    raw_acts = torch.empty((3, 0))

    with pytest.raises(ValueError, match="raw_acts_base is required"):
        save_raw_activation_alpha_diagnostics(
            raw_acts,
            alpha=0.5,
            min_acts=2,
            output_dir=tmp_path,
            raw_acts_base=None,
            alpha_candidates=[0.5],
        )
