import os

import numpy as np
import pytest
import torch

from analysis.activation_diagnostics import (
    jaccard_similarity_matrix,
    local_alpha_candidates,
    raw_activation_correlation_matrix,
    raw_activation_cosine_similarity_matrix,
    save_activation_diagnostics,
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
    assert paths["diagnostics_report"] == tmp_path / "activation_diagnostics_report.md"
    assert report.startswith("# Activation Diagnostics Report")
    assert "POST-BINARIZATION ACTIVATION COUNT SUMMARY" in report
    assert "| Zero Activation Count | 2 |" in report
    assert "| Zero Activation % | 50.000000 |" in report
    assert "| Below Min Acts Count | 2 |" in report
    assert "| Below Min Acts % | 50.000000 |" in report
    assert "| Kept Count | 2 |" in report
    assert "| Kept % | 50.000000 |" in report
    assert "| Max | 3.000000 |" in report

    assert os.path.getsize(paths["hist_full"]) > 0
    assert os.path.getsize(paths["hist_nonzero"]) > 0
    assert os.path.getsize(paths["jaccard_heatmap"]) > 0


def test_activation_diagnostics_writes_ordered_markdown_report_with_embedded_images(tmp_path):
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
    binary_acts = raw_acts > torch.quantile(raw_acts, 0.5, dim=0)

    paths = save_activation_diagnostics(
        raw_acts,
        raw_acts_base,
        binary_acts,
        min_acts=2,
        output_dir=tmp_path,
        alpha=0.5,
        alpha_candidates=[0.5],
        top_k=2,
    )

    report = paths["diagnostics_report"].read_text(encoding="utf-8")
    assert paths["diagnostics_report"] == tmp_path / "activation_diagnostics_report.md"
    assert report.index("## RAW ACTIVATION ALPHA SWEEP") < report.index(
        "## POST-BINARIZATION ACTIVATION COUNT SUMMARY"
    )
    assert report.index("## POST-BINARIZATION ACTIVATION COUNT SUMMARY") < report.index(
        "## SIMILARITY & CORRELATION ANALYSIS"
    )
    assert report.index("## SIMILARITY & CORRELATION ANALYSIS") < report.index(
        "## VISUALIZATIONS"
    )
    assert report.index("Top Pearson Correlation Neuron Pairs") < report.index(
        "## VISUALIZATIONS"
    )
    assert report.index("Top Cosine Similarity Neuron Pairs") < report.index(
        "## VISUALIZATIONS"
    )
    visualizations = report.split("## VISUALIZATIONS", 1)[1]
    for image_name in (
        "activation_counts_hist_full.png",
        "activation_counts_hist_nonzero.png",
        "binarized_activation_jaccard_heatmap.png",
        "raw_activation_correlation_heatmap_base.png",
        "raw_activation_correlation_heatmap_finetuned.png",
        "raw_activation_correlation_heatmap_difference.png",
        "raw_activation_cosine_similarity_heatmap_base.png",
        "raw_activation_cosine_similarity_heatmap_finetuned.png",
        "raw_activation_cosine_similarity_heatmap_difference.png",
    ):
        assert f"./{image_name}" in visualizations
        assert os.path.getsize(tmp_path / image_name) > 0


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
        top_k=2,
    )

    report = paths["diagnostics_report"].read_text(encoding="utf-8")
    assert paths["diagnostics_report"] == tmp_path / "activation_diagnostics_report.md"
    assert "RAW ACTIVATION ALPHA SWEEP" in report
    assert "min_acts: 2" in report
    assert f"| 0.5 | 2 | {100 * 2 / 3:.6f} | 2 | {100 * 2 / 3:.6f} | 1 | {100 / 3:.6f} |" in report
    assert "Pearson correlation base" in report
    assert "Pearson correlation finetuned" in report
    assert "Pearson correlation difference" in report
    assert "Cosine similarity base" in report
    assert "Cosine similarity finetuned" in report
    assert "Cosine similarity difference" in report
    assert "Top positive pairs" in report
    assert "Top increased pairs" in report

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
