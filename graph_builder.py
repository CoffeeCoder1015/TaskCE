import argparse
import math
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from analysis.activation_diagnostics import (
    raw_activation_correlation_matrix,
    raw_activation_cosine_similarity_matrix,
)
from capture import load_captured_activations


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_FORMULA_COLUMNS = {"neuron", "formula", "iou"}
DEFAULT_LOCAL_TOP_PERCENTILE = 0.01
DEFAULT_MIN_NEIGHBORS = 3
DEFAULT_K_CORE_START = 3
DEFAULT_COMMUNITY_SEED = 0


def report_graph(
    adj_matrix,
    formulas,
    *,
    metric_name="graph",
    relu_sparsify=False,
    local_top_percentile=DEFAULT_LOCAL_TOP_PERCENTILE,
    min_neighbors=DEFAULT_MIN_NEIGHBORS,
    k_core_start=DEFAULT_K_CORE_START,
    community_seed=DEFAULT_COMMUNITY_SEED,
):
    import networkx as nx

    # Stage 1: normalize one adjacency matrix and the formula metadata rows.
    matrix = np.asarray(adj_matrix, dtype=float)
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], (
        f"expected square adjacency matrix, got shape {matrix.shape}"
    )
    assert np.isfinite(matrix).all(), "adjacency matrix must not contain NaN or inf"
    relu_sparsify = bool(relu_sparsify)
    assert 0 < float(local_top_percentile) <= 1, "local_top_percentile must be in (0, 1]"
    assert 1 <= int(min_neighbors) < matrix.shape[0], (
        "min_neighbors must be at least 1 and smaller than neuron count"
    )
    assert int(k_core_start) >= 1, "k_core_start must be at least 1"

    formula_frame = pd.DataFrame(formulas)
    missing_formula_columns = REQUIRED_FORMULA_COLUMNS - set(formula_frame.columns)
    assert not missing_formula_columns, (
        f"formula rows missing required columns: {sorted(missing_formula_columns)}"
    )
    assert not formula_frame.isna().any().any(), "formula rows must not contain null values"

    metadata_by_neuron = {}
    for row in formula_frame.to_dict("records"):
        neuron = int(row.pop("neuron"))
        metadata_by_neuron[neuron] = {}
        for key, value in row.items():
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            metadata_by_neuron[neuron][key] = value
    expected_neurons = set(range(matrix.shape[0]))
    assert set(metadata_by_neuron) == expected_neurons, (
        "formula neurons must match adjacency indices: "
        f"formulas={sorted(metadata_by_neuron)}, adjacency={sorted(expected_neurons)}"
    )

    # Stage 2: construct a single NetworkX graph for this matrix and attach neuron metadata.
    graph = nx.Graph(metric_name=str(metric_name))
    for neuron in range(matrix.shape[0]):
        graph.add_node(
            int(neuron),
            neuron=int(neuron),
            **metadata_by_neuron[int(neuron)],
        )

    # Stage 3: sparsify by each neuron's strongest local neighbors, preserving signed weights.
    selected_edges = set()
    for row_index, row in enumerate(matrix):
        candidates = []
        for column_index, weight in enumerate(row):
            weight = float(weight)
            if column_index == row_index or not np.isfinite(weight):
                continue
            strength = max(weight, 0.0) if relu_sparsify else abs(weight)
            if strength == 0.0:
                continue
            candidates.append((int(column_index), strength))

        if not candidates:
            continue

        percentile_count = math.ceil((matrix.shape[0] - 1) * float(local_top_percentile))
        keep_count = max(1, int(min_neighbors), percentile_count)
        keep_count = min(keep_count, len(candidates))
        candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))

        for column_index, _ in candidates[:keep_count]:
            first, second = sorted((int(row_index), int(column_index)))
            selected_edges.add((first, second))

    for first, second in sorted(selected_edges):
        weight = float(matrix[first, second])
        if weight > 0:
            sign = "positive"
        elif weight < 0:
            sign = "negative"
        else:
            sign = "zero"
        graph.add_edge(
            first,
            second,
            weight=weight,
            strength=abs(weight),
            sign=sign,
        )


    # Stage 4: run k-core analysis on the sparsified graph.
    core_numbers = nx.core_number(graph)
    nx.set_node_attributes(graph, core_numbers, "core_number")

    max_core_number = max(core_numbers.values(), default=0)

    selected_k = int(k_core_start)
    selected_graph = nx.k_core(graph, k=selected_k).copy()
    assert selected_graph.number_of_nodes() > 0, (
        f"requested k-core is empty for k={selected_k}"
    )

    # Stage 5: run community detection on the selected k-core graph.
    communities = [
        {int(node) for node in community}
        for community in nx.community.louvain_communities(
            selected_graph,
            weight="strength",
            seed=int(community_seed),
        )
    ]


    community_by_node = {}
    for community_id, community_nodes in enumerate(communities):
        for node in community_nodes:
            community_by_node[node] = int(community_id)
    nx.set_node_attributes(selected_graph, community_by_node, "community")

    # Stage 6: shape the graph analysis into a report dictionary.
    community_reports = []
    for community_id, community_nodes in enumerate(communities):
        ordered_nodes = sorted(community_nodes)
        community_graph = selected_graph.subgraph(ordered_nodes)
        members = []
        for node in ordered_nodes:
            attrs = dict(selected_graph.nodes[node])
            attrs["neuron"] = int(node)
            members.append(attrs)
        summary_nodes = sorted(
            ordered_nodes,
            key=lambda node: (
                -community_graph.degree(node),
                -sum(
                    data.get("strength", 0.0)
                    for _, _, data in community_graph.edges(node, data=True)
                ),
                node,
            ),
        )[:10]
        summary_members = []
        for node in summary_nodes:
            attrs = dict(selected_graph.nodes[node])
            attrs["neuron"] = int(node)
            summary_members.append(attrs)

        strongest_edges = []
        for first, second, data in sorted(
            community_graph.edges(data=True),
            key=lambda edge: (-edge[2].get("strength", 0.0), edge[0], edge[1]),
        )[:10]:
            strongest_edges.append(
                {
                    "first": int(first),
                    "second": int(second),
                    "weight": float(data["weight"]),
                    "sign": data["sign"],
                }
            )

        community_reports.append(
            {
                "id": int(community_id),
                "nodes": community_graph.number_of_nodes(),
                "internal_edges": community_graph.number_of_edges(),
                "positive_internal_edges": sum(
                    1
                    for _, _, data in community_graph.edges(data=True)
                    if data.get("sign") == "positive"
                ),
                "negative_internal_edges": sum(
                    1
                    for _, _, data in community_graph.edges(data=True)
                    if data.get("sign") == "negative"
                ),
                "members": members,
                "summary_members": summary_members,
                "strongest_edges": strongest_edges,
            }
        )

    report = {
        "metric_name": str(metric_name),
        "options": {
            "relu_sparsify": bool(relu_sparsify),
            "local_top_percentile": float(local_top_percentile),
            "min_neighbors": int(min_neighbors),
            "k_core_start": int(k_core_start),
            "community_seed": int(community_seed),
        },
        "graph_summary": {
            "sparsified_nodes": graph.number_of_nodes(),
            "sparsified_edges": graph.number_of_edges(),
            "core_nodes": selected_graph.number_of_nodes(),
            "core_edges": selected_graph.number_of_edges(),
            "communities": len(communities),
            "positive_core_edges": sum(
                1
                for _, _, data in selected_graph.edges(data=True)
                if data.get("sign") == "positive"
            ),
            "negative_core_edges": sum(
                1
                for _, _, data in selected_graph.edges(data=True)
                if data.get("sign") == "negative"
            ),
        },
        "k_core": {
            "requested_k": int(k_core_start),
            "selected_k": int(selected_k),
            "max_core_number": int(max_core_number),
            "core_numbers": {int(node): int(value) for node, value in core_numbers.items()},
        },
        "communities": community_reports,
    }
    return report, selected_graph


def build_project_graph_reports(
    results_path,
    *,
    relu_sparsify=False,
    local_top_percentile=DEFAULT_LOCAL_TOP_PERCENTILE,
    min_neighbors=DEFAULT_MIN_NEIGHBORS,
    k_core_start=DEFAULT_K_CORE_START,
    community_seed=DEFAULT_COMMUNITY_SEED,
):
    results_path = Path(results_path)
    results_root = results_path if results_path.is_dir() else results_path.parent.parent
    captured_results = load_captured_activations(results_path)
    written_paths = {}

    for task_name in sorted(captured_results):
        result_csv_path = results_root / f"{task_name}_beam_results.csv"
        if not result_csv_path.exists():
            warnings.warn(f"skipping task {task_name}: no formula CSV at {result_csv_path}")
            continue

        task_results = captured_results[task_name]
        if "finetuned" not in task_results:
            warnings.warn(f"skipping task {task_name}: no finetuned capture found")
            continue

        formulas = pd.read_csv(result_csv_path)
        states = task_results["finetuned"].states
        output_dir = results_root / task_name

        pearson_matrix = raw_activation_correlation_matrix(states)
        pearson_report, pearson_graph = report_graph(
            pearson_matrix,
            formulas,
            metric_name="pearson",
            relu_sparsify=relu_sparsify,
            local_top_percentile=local_top_percentile,
            min_neighbors=min_neighbors,
            k_core_start=k_core_start,
            community_seed=community_seed,
        )
        written_paths[f"{task_name}/pearson"] = save_graph_report(
            pearson_report,
            pearson_graph,
            output_dir,
            name="pearson",
        )

        cosine_matrix = raw_activation_cosine_similarity_matrix(states)
        cosine_report, cosine_graph = report_graph(
            cosine_matrix,
            formulas,
            metric_name="cosine",
            relu_sparsify=relu_sparsify,
            local_top_percentile=local_top_percentile,
            min_neighbors=min_neighbors,
            k_core_start=k_core_start,
            community_seed=community_seed,
        )
        written_paths[f"{task_name}/cosine"] = save_graph_report(
            cosine_report,
            cosine_graph,
            output_dir,
            name="cosine",
        )

        if "base" not in task_results:
            warnings.warn(f"skipping task {task_name}: no base capture found")
            continue

        base_states = task_results["base"].states

        base_pearson_matrix = raw_activation_correlation_matrix(base_states)
        base_pearson_report, base_pearson_graph = report_graph(
            base_pearson_matrix,
            formulas,
            metric_name="base_pearson",
            relu_sparsify=relu_sparsify,
            local_top_percentile=local_top_percentile,
            min_neighbors=min_neighbors,
            k_core_start=k_core_start,
            community_seed=community_seed,
        )
        written_paths[f"{task_name}/base_pearson"] = save_graph_report(
            base_pearson_report,
            base_pearson_graph,
            output_dir,
            name="base_pearson",
        )

        base_cosine_matrix = raw_activation_cosine_similarity_matrix(base_states)
        base_cosine_report, base_cosine_graph = report_graph(
            base_cosine_matrix,
            formulas,
            metric_name="base_cosine",
            relu_sparsify=relu_sparsify,
            local_top_percentile=local_top_percentile,
            min_neighbors=min_neighbors,
            k_core_start=k_core_start,
            community_seed=community_seed,
        )
        written_paths[f"{task_name}/base_cosine"] = save_graph_report(
            base_cosine_report,
            base_cosine_graph,
            output_dir,
            name="base_cosine",
        )

    return written_paths


def render_report_markdown(report, *, member_key="members"):
    lines = [
        f"# Neuron Graph Cluster Report: {report['metric_name']}",
        "",
        "## Graph Summary",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
    ]
    for key, value in report["graph_summary"].items():
        lines.append(f"| {key.replace('_', ' ').title()} | {value} |")

    lines.extend(
        [
            "",
            "## K-Core Results",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| Requested K | {report['k_core']['requested_k']} |",
            f"| Selected K | {report['k_core']['selected_k']} |",
            f"| Max Core Number | {report['k_core']['max_core_number']} |",
        ]
    )

    lines.extend(["", "## Communities", ""])
    assert report["communities"], "report must contain communities"

    for community in report["communities"]:
        lines.extend(
            [
                f"### Community {community['id']}",
                "",
                "| Metric | Value |",
                "| :--- | :--- |",
                f"| Nodes | {community['nodes']} |",
                f"| Internal Edges | {community['internal_edges']} |",
                f"| Positive Internal Edges | {community['positive_internal_edges']} |",
                f"| Negative Internal Edges | {community['negative_internal_edges']} |",
                "",
            ]
        )

        members = community[member_key]
        member_columns = sorted({key for member in members for key in member})
        preferred = ["neuron", "formula", "iou", "core_number", "community"]
        weight_columns = [
            column for column in member_columns if column.startswith("weight_")
        ]
        columns = [
            *[column for column in preferred if column in member_columns],
            *weight_columns,
            *[
                column
                for column in member_columns
                if column not in {*preferred, *weight_columns}
            ],
        ]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(":---" for _ in columns) + " |")
        for member in members:
            lines.append(
                "| "
                + " | ".join(markdown_cell(member.get(column, "")) for column in columns)
                + " |"
            )

        lines.extend(["", "#### Strongest Internal Edges", ""])
        if not community["strongest_edges"]:
            lines.append("No internal edges.")
        for edge in community["strongest_edges"]:
            lines.append(
                f"- neuron {edge['first']} <-> neuron {edge['second']}: "
                f"{edge['weight']:.6f} ({edge['sign']})"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def save_graph_report(report, graph, output_dir, name):
    import networkx as nx

    graph_dir = Path(output_dir) / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": graph_dir / f"{name}_neuron_graph.png",
        "full_report": graph_dir / f"{name}_cluster_report_full.md",
        "summary_report": graph_dir / f"{name}_cluster_report_summary.md",
    }

    plt.figure(figsize=(24, 20))
    assert graph.number_of_nodes() > 0, "cannot save an empty graph"
    assert graph.number_of_edges() > 0, "cannot save a graph with no edges"

    positions = nx.spring_layout(
        graph,
        weight="strength",
        k=2.0 / math.sqrt(graph.number_of_nodes()),
        scale=3.0,
        seed=report["options"]["community_seed"],
    )
    positive_edges = [
        edge for edge in graph.edges if graph.edges[edge]["sign"] == "positive"
    ]
    negative_edges = [
        edge for edge in graph.edges if graph.edges[edge]["sign"] == "negative"
    ]
    node_colors = [graph.nodes[node]["community"] for node in graph.nodes]
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        node_size=450,
        edgecolors="black",
        linewidths=0.6,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=positive_edges,
        edge_color="#3366cc",
        alpha=0.6,
        width=1.3,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=negative_edges,
        edge_color="#cc3333",
        alpha=0.6,
        width=1.3,
        style="dashed",
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: str(node) for node in graph.nodes},
        font_size=8,
    )
    plt.title(f"{report['metric_name']} neuron graph (k-core {report['k_core']['selected_k']})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(paths["png"], dpi=250)
    plt.close()

    paths["full_report"].write_text(
        render_report_markdown(report, member_key="members"),
        encoding="utf-8",
    )
    paths["summary_report"].write_text(
        render_report_markdown(report, member_key="summary_members"),
        encoding="utf-8",
    )
    return paths


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build neuron graphs from captured activations and formula CSVs.",
    )
    parser.add_argument(
        "results_path",
        nargs="?",
        default="results",
        help="Results dir or captured_results.pt path. Defaults to results.",
    )
    parser.add_argument(
        "--relu",
        action="store_true",
        help="Drop non-positive weights before selecting strongest neighbors.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_project_graph_reports(args.results_path, relu_sparsify=args.relu_sparsify)
