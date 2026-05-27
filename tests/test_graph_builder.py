import warnings
from pathlib import Path

import pandas as pd
import pytest
import torch

from capture.capturer import CapturedResults
from capture.saving import save_captured_activations
from graph_builder import (
    build_project_graph_reports,
    report_graph,
    render_report_markdown,
    save_graph_report,
)


def formulas():
    return pd.DataFrame(
        [
            {
                "neuron": 0,
                "formula": "tok:A",
                "iou": 0.90,
                "weight_entailment": 1.0,
            },
            {
                "neuron": 1,
                "formula": "tok:B",
                "iou": 0.80,
                "weight_entailment": -2.0,
            },
            {
                "neuron": 2,
                "formula": "tok:C",
                "iou": 0.70,
                "weight_entailment": 3.0,
            },
            {
                "neuron": 3,
                "formula": "LOW_ACTS_PRUNED",
                "iou": 0.0,
                "weight_entailment": 0.0,
            },
        ]
    )


def test_report_graph_builds_one_signed_graph_from_one_adjacency_matrix():
    adjacency = [
        [1.0, 0.9, -0.8, 0.1],
        [0.9, 1.0, -0.7, 0.0],
        [-0.8, -0.7, 1.0, 0.2],
        [0.1, 0.0, 0.2, 1.0],
    ]

    report, graph = report_graph(
        adjacency,
        formulas(),
        metric_name="pearson",
        local_top_percentile=0.25,
        min_neighbors=2,
        k_core_start=2,
    )

    assert report["metric_name"] == "pearson"
    assert report["graph_summary"]["sparsified_nodes"] == 4
    assert report["k_core"]["selected_k"] == 2
    assert report["k_core"]["max_core_number"] == 2
    assert report["communities"]

    assert set(graph.nodes) == {0, 1, 2, 3}
    assert graph.nodes[0]["formula"] == "tok:A"
    assert graph.nodes[1]["iou"] == 0.80
    assert graph.nodes[2]["weight_entailment"] == 3.0
    assert graph.nodes[0]["core_number"] == 2

    assert graph.has_edge(0, 1)
    assert graph[0][1]["weight"] == 0.9
    assert graph[0][1]["strength"] == 0.9
    assert graph[0][1]["sign"] == "positive"

    assert graph.has_edge(0, 2)
    assert graph[0][2]["weight"] == -0.8
    assert graph[0][2]["strength"] == 0.8
    assert graph[0][2]["sign"] == "negative"
    assert not graph.has_edge(0, 0)


def test_report_graph_relu_sparsify_drops_non_positive_candidates():
    adjacency = [
        [1.0, -0.9, 0.3],
        [-0.9, 1.0, 0.2],
        [0.3, 0.2, 1.0],
    ]

    report, graph = report_graph(
        adjacency,
        formulas().iloc[:3],
        relu_sparsify=True,
        local_top_percentile=0.25,
        min_neighbors=1,
        k_core_start=1,
    )

    assert report["options"]["relu_sparsify"] is True
    assert not graph.has_edge(0, 1)
    assert graph.has_edge(0, 2)
    assert graph.has_edge(1, 2)
    assert all(data["sign"] == "positive" for _, _, data in graph.edges(data=True))


def test_report_graph_fails_fast_when_formula_metadata_does_not_cover_matrix():
    adjacency = [
        [1.0, 0.9, 0.8],
        [0.9, 1.0, 0.7],
        [0.8, 0.7, 1.0],
    ]
    incomplete_formulas = pd.DataFrame(
        [
            {"neuron": 0, "formula": "tok:A", "iou": 0.9},
            {"neuron": 1, "formula": "tok:B", "iou": 0.8},
        ]
    )

    with pytest.raises(AssertionError, match="formula neurons must match adjacency"):
        report_graph(adjacency, incomplete_formulas, min_neighbors=1)


def test_report_graph_allows_isolated_neurons_from_sparsification():
    adjacency = [
        [1.0, 0.9, -0.8, 0.0],
        [0.9, 1.0, -0.7, 0.0],
        [-0.8, -0.7, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    report, graph = report_graph(
        adjacency,
        formulas(),
        local_top_percentile=0.25,
        min_neighbors=2,
        k_core_start=2,
    )

    assert set(graph.nodes) == {0, 1, 2}
    assert report["k_core"]["core_numbers"][3] == 0


def test_report_graph_fails_fast_when_formula_columns_are_malformed():
    adjacency = [
        [1.0, 0.9],
        [0.9, 1.0],
    ]
    malformed_formulas = pd.DataFrame(
        [
            {"neuron": 0, "formula": "tok:A"},
            {"neuron": 1, "formula": "tok:B"},
        ]
    )

    with pytest.raises(AssertionError, match="formula rows missing required columns"):
        report_graph(adjacency, malformed_formulas, min_neighbors=1, k_core_start=1)


def test_report_graph_fails_fast_when_formula_rows_contain_nulls():
    adjacency = [
        [1.0, 0.9],
        [0.9, 1.0],
    ]
    null_formulas = pd.DataFrame(
        [
            {"neuron": 0, "formula": "tok:A", "iou": 0.9},
            {"neuron": 1, "formula": None, "iou": 0.8},
        ]
    )

    with pytest.raises(AssertionError, match="formula rows must not contain null values"):
        report_graph(adjacency, null_formulas, min_neighbors=1, k_core_start=1)


def test_report_graph_fails_fast_when_graph_settings_are_invalid():
    adjacency = [
        [1.0, 0.9],
        [0.9, 1.0],
    ]

    with pytest.raises(AssertionError, match="local_top_percentile"):
        report_graph(adjacency, formulas().iloc[:2], local_top_percentile=0)

    with pytest.raises(AssertionError, match="min_neighbors"):
        report_graph(adjacency, formulas().iloc[:2], min_neighbors=2)

    with pytest.raises(AssertionError, match="k_core_start"):
        report_graph(adjacency, formulas().iloc[:2], min_neighbors=1, k_core_start=0)


def test_report_graph_fails_fast_when_requested_k_core_is_empty():
    adjacency = [
        [1.0, 0.9, -0.8],
        [0.9, 1.0, -0.7],
        [-0.8, -0.7, 1.0],
    ]

    with pytest.raises(AssertionError, match="requested k-core is empty"):
        report_graph(
            adjacency,
            formulas().iloc[:3],
            local_top_percentile=0.25,
            min_neighbors=2,
            k_core_start=3,
        )


def test_save_graph_report_uses_name_and_graph_subdirectory(tmp_path):
    adjacency = [
        [1.0, 0.9, -0.8, 0.3],
        [0.9, 1.0, -0.7, 0.2],
        [-0.8, -0.7, 1.0, 0.4],
        [0.3, 0.2, 0.4, 1.0],
    ]
    report, graph = report_graph(
        adjacency,
        formulas(),
        metric_name="pearson",
        local_top_percentile=0.25,
        min_neighbors=2,
        k_core_start=2,
    )

    paths = save_graph_report(report, graph, tmp_path, name="pearson")

    assert paths["png"] == tmp_path / "graph" / "pearson_neuron_graph.png"
    assert paths["full_report"] == tmp_path / "graph" / "pearson_cluster_report_full.md"
    assert paths["summary_report"] == (
        tmp_path / "graph" / "pearson_cluster_report_summary.md"
    )
    assert set(paths) == {"png", "full_report", "summary_report"}
    assert paths["png"].stat().st_size > 0
    assert not (tmp_path / "graph" / "pearson_neuron_graph.graphml").exists()

    markdown = paths["full_report"].read_text(encoding="utf-8")
    assert "# Neuron Graph Cluster Report: pearson" in markdown
    assert "## Core Metadata" in markdown
    assert "### Shell Histogram" in markdown
    assert "## Degree Distribution" in markdown
    assert "## Weighted Degree Distribution" in markdown
    assert "| Selected K | 2 |" in markdown
    assert "| Max Core Number | 2 |" in markdown
    assert "core_number" in markdown
    assert "weighted_degree" in markdown
    assert "tok:A" in markdown
    assert "tok:B" in markdown
    assert "tok:C" in markdown


def test_summary_report_limits_members_to_top_ten_hubs_by_weighted_degree():
    adjacency = [
        [1.0 if row == column else 0.8 for column in range(12)]
        for row in range(12)
    ]
    formula_rows = pd.DataFrame(
        {"neuron": neuron, "formula": f"tok:{neuron}", "iou": 1.0}
        for neuron in range(12)
    )
    report, _ = report_graph(
        adjacency,
        formula_rows,
        local_top_percentile=1.0,
        min_neighbors=1,
        k_core_start=1,
    )

    assert len(report["communities"][0]["members"]) == 12
    assert len(report["communities"][0]["top_hubs"]) == 10
    assert report["k_core"]["shell_histogram"] == {11: 12}
    assert report["degree_distribution"]["max"] == 11.0
    assert report["weighted_degree_distribution"]["max"] == pytest.approx(8.8)

    full_markdown = render_report_markdown(report, member_key="members")
    summary_markdown = render_report_markdown(report, member_key="top_hubs")

    assert "#### Top Non-Pruned Hubs" in summary_markdown
    assert "tok:10" in full_markdown
    assert "tok:11" in full_markdown
    assert "tok:10" not in summary_markdown
    assert "tok:11" not in summary_markdown


def test_project_runner_calls_report_graph_separately_for_pearson_and_cosine(tmp_path):
    captured_results = {
        "snli": {
            "base": CapturedResults(
                states=torch.tensor(
                    [
                        [4.0, 1.0, -1.0],
                        [3.0, 2.0, -2.0],
                        [2.0, 3.0, -3.0],
                        [1.0, 4.0, -4.0],
                    ]
                ),
                labels=["entailment"] * 4,
                layer="model.layers.1",
            ),
            "finetuned": CapturedResults(
                states=torch.tensor(
                    [
                        [1.0, 1.0, -1.0],
                        [2.0, 2.0, -2.0],
                        [3.0, 3.0, -3.0],
                        [4.0, 4.0, -4.0],
                    ]
                ),
                labels=["entailment"] * 4,
                layer="model.layers.1",
            )
        },
    }
    save_captured_activations(captured_results, tmp_path)
    pd.DataFrame(
        [
            {"neuron": 0, "formula": "tok:A", "iou": 1.0, "weight_entailment": 0.1},
            {"neuron": 1, "formula": "tok:B", "iou": 0.9, "weight_entailment": 0.2},
            {"neuron": 2, "formula": "tok:C", "iou": 0.8, "weight_entailment": 0.3},
        ]
    ).to_csv(tmp_path / "snli_beam_results.csv", index=False)

    written = build_project_graph_reports(
        tmp_path,
        local_top_percentile=0.5,
        min_neighbors=2,
        k_core_start=2,
    )

    assert sorted(written) == [
        "snli/base_cosine",
        "snli/base_pearson",
        "snli/cosine",
        "snli/pearson",
    ]
    assert written["snli/base_pearson"]["summary_report"] == (
        Path(tmp_path) / "snli" / "graph" / "base_pearson_cluster_report_summary.md"
    )
    assert written["snli/base_cosine"]["summary_report"] == (
        Path(tmp_path) / "snli" / "graph" / "base_cosine_cluster_report_summary.md"
    )
    assert written["snli/pearson"]["summary_report"] == (
        Path(tmp_path) / "snli" / "graph" / "pearson_cluster_report_summary.md"
    )
    assert written["snli/cosine"]["summary_report"] == (
        Path(tmp_path) / "snli" / "graph" / "cosine_cluster_report_summary.md"
    )
    assert not (Path(tmp_path) / "snli" / "graph" / "combined_cluster_report.md").exists()
    assert not (Path(tmp_path) / "snli" / "pearson_cluster_report.md").exists()


def test_project_runner_warns_and_skips_tasks_without_formula_csv(tmp_path):
    captured_results = {
        "missing_csv": {
            "finetuned": CapturedResults(
                states=torch.ones((4, 3)),
                labels=["entailment"] * 4,
                layer="model.layers.1",
            )
        },
    }
    save_captured_activations(captured_results, tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        written = build_project_graph_reports(tmp_path)

    assert written == {}
    assert any("missing_csv" in str(w.message) for w in caught)
