import argparse
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from analysis.activation_diagnostics import (
    raw_activation_correlation_matrix,
    raw_activation_cosine_similarity_matrix,
)
from analysis.graph import (
    build_graph,
    zero_diag,
    relu,
    local_top_percent_union,
    analyze_k_core,
    analyze_degrees,
    analyze_communities,
    analyze_hubs,
    analyze_k_core_stats,
    render_full_report,
    save_graph_plot,
)
from capture import load_captured_activations


REQUIRED_FORMULA_COLUMNS = {"neuron", "formula", "iou"}
DEFAULT_LOCAL_TOP_PERCENTILE = 0.001
DEFAULT_K_CORE_START = 3
DEFAULT_COMMUNITY_SEED = 0
HUB_REPORT_LIMIT = 10


def _make_sparsify_fn(*, local_top_percentile, relu_sparsify):
    """Compose ``analysis.graph`` primitives into a sparsify function for
    ``build_graph``.

    Pipeline: ``zero_diag`` → (optional) ``relu`` → ``local_top_percent_union``.
    """

    def sparsify(adj_matrix):
        matrix = zero_diag(adj_matrix)
        if relu_sparsify:
            matrix = relu(matrix)
        return local_top_percent_union(matrix, local_top_percentile)

    return sparsify


def _add_extra_node_attributes(graph, formula_frame):
    """Attach formula columns beyond ``formula`` and ``iou`` to nodes.

    ``build_graph`` only copies ``formula`` and ``iou``; this fills in any
    remaining columns (e.g. ``weight_entailment``).
    """
    extra_columns = [
        col for col in formula_frame.columns
        if col not in {"neuron", "formula", "iou"}
    ]
    if not extra_columns:
        return
    extra_attrs = {}
    for row in formula_frame.to_dict("records"):
        neuron = int(row["neuron"])
        attrs = {}
        for col in extra_columns:
            value = row[col]
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            attrs[col] = value
        extra_attrs[neuron] = attrs
    nx.set_node_attributes(graph, extra_attrs)


def report_graph(
    adj_matrix,
    formulas,
    *,
    metric_name="graph",
    relu_sparsify=False,
    local_top_percentile=DEFAULT_LOCAL_TOP_PERCENTILE,
    k_core_start=DEFAULT_K_CORE_START,
    community_seed=DEFAULT_COMMUNITY_SEED,
    community_method="louvain",
):
    """Build a sparsified neuron graph and run structural analyses.

    Returns ``(report, k_core_graph)`` where *report* is a dict containing
    the full sparsified graph, the k-core subgraph, and analysis DataFrames
    (communities, degrees, k-core, hubs, k-core stats).

    Louvain uses relu preprocessing. Leiden uses the signed sparse graph.
    """

    # ── validate inputs ──────────────────────────────────────────────────
    matrix = np.asarray(adj_matrix, dtype=float)
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], (
        f"expected square adjacency matrix, got shape {matrix.shape}"
    )
    assert np.isfinite(matrix).all(), "adjacency matrix must not contain NaN or inf"
    relu_sparsify = bool(relu_sparsify)
    community_method = str(community_method)
    assert 0 < float(local_top_percentile) <= 1, "local_top_percentile must be in (0, 1]"
    assert int(k_core_start) >= 1, "k_core_start must be at least 1"

    formula_frame = pd.DataFrame(formulas)
    missing_formula_columns = REQUIRED_FORMULA_COLUMNS - set(formula_frame.columns)
    assert not missing_formula_columns, (
        f"formula rows missing required columns: {sorted(missing_formula_columns)}"
    )
    assert not formula_frame.isna().any().any(), "formula rows must not contain null values"

    expected_neurons = set(range(matrix.shape[0]))
    formula_neurons = {int(n) for n in formula_frame["neuron"]}
    assert formula_neurons == expected_neurons, (
        "formula neurons must match adjacency indices: "
        f"formulas={sorted(formula_neurons)}, adjacency={sorted(expected_neurons)}"
    )

    # ── build sparsified graph ───────────────────────────────────────────
    sparsify_fn = _make_sparsify_fn(
        local_top_percentile=local_top_percentile,
        relu_sparsify=relu_sparsify or community_method == "louvain",
    )
    graph = build_graph(matrix, sparsify_fn, formula_frame)
    _add_extra_node_attributes(graph, formula_frame)

    # ── k-core analysis on the full sparsified graph ─────────────────────
    k_core_df = analyze_k_core(graph)
    nx.set_node_attributes(
        graph,
        k_core_df.set_index("node")["core_number"].to_dict(),
        "core_number",
    )

    selected_k = int(k_core_start)
    k_core_graph = nx.k_core(graph, k=selected_k).copy()
    assert k_core_graph.number_of_nodes() > 0, (
        f"requested k-core is empty for k={selected_k}"
    )

    # ── analyses on the k-core subgraph ──────────────────────────────────
    communities_df = analyze_communities(
        k_core_graph, method=community_method, seed=int(community_seed)
    )
    degrees_df = analyze_degrees(k_core_graph)
    hubs_df = analyze_hubs(degrees_df, communities_df, limit=HUB_REPORT_LIMIT)

    # k-core stats use full-graph degrees so every shell is represented.
    full_degrees_df = analyze_degrees(graph)
    k_core_stats_df = analyze_k_core_stats(k_core_df, full_degrees_df)

    # Attach community labels to the k-core graph nodes.
    nx.set_node_attributes(
        k_core_graph,
        communities_df.set_index("node")["community"].to_dict(),
        "community",
    )

    report = {
        "metric_name": str(metric_name),
        "options": {
            "relu_sparsify": bool(relu_sparsify),
            "local_top_percentile": float(local_top_percentile),
            "k_core_start": int(k_core_start),
            "community_seed": int(community_seed),
            "community_method": community_method,
        },
        "graph": graph,
        "k_core_graph": k_core_graph,
        "formula_dataframe": formula_frame,
        "communities_df": communities_df,
        "degrees_df": degrees_df,
        "k_core_df": k_core_df,
        "hubs_df": hubs_df,
        "k_core_stats_df": k_core_stats_df,
    }
    return report, k_core_graph


def save_graph_report(report, graph, output_dir, name):
    """Save a community-coloured graph plot and markdown reports.

    Returns a dict mapping output keys to their paths.
    """
    graph_dir = Path(output_dir) / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    communities_df = report["communities_df"]
    formula_dataframe = report["formula_dataframe"]
    degrees_df = report["degrees_df"]
    k_core_df = report["k_core_df"]
    metric_name = report["metric_name"]
    selected_k = report["options"]["k_core_start"]
    seed = report["options"]["community_seed"]

    paths = {
        "png": graph_dir / f"{name}_neuron_graph.png",
        "full_report": graph_dir / f"{name}_cluster_report_full.md",
        "summary_report": graph_dir / f"{name}_cluster_report_summary.md",
    }

    # Community-coloured graph plot.
    save_graph_plot(
        graph,
        communities_df,
        paths["png"],
        title=f"{metric_name} neuron graph (k-core {selected_k})",
        seed=seed,
    )

    # Markdown reports.
    full_text = render_full_report(
        communities_df, formula_dataframe, degrees_df, k_core_df
    )
    paths["full_report"].write_text(full_text, encoding="utf-8")
    # Without a tokenizer we cannot produce atomic-frequency data for
    # the richer summary report, so the summary mirrors the full report.
    paths["summary_report"].write_text(full_text, encoding="utf-8")

    return paths


def build_project_graph_reports(
    results_path,
    *,
    relu_sparsify=False,
    local_top_percentile=DEFAULT_LOCAL_TOP_PERCENTILE,
    k_core_start=DEFAULT_K_CORE_START,
    community_seed=DEFAULT_COMMUNITY_SEED,
    community_method="louvain",
):
    """Run graph analysis for every task found under *results_path*."""
    results_path = Path(results_path)
    results_root = results_path if results_path.is_dir() else results_path.parent.parent
    captured_results = load_captured_activations(results_path)
    written_paths = {}

    graph_kwargs = dict(
        relu_sparsify=relu_sparsify,
        local_top_percentile=local_top_percentile,
        k_core_start=k_core_start,
        community_seed=community_seed,
        community_method=community_method,
    )

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
            pearson_matrix, formulas, metric_name="pearson", **graph_kwargs
        )
        written_paths[f"{task_name}/pearson"] = save_graph_report(
            pearson_report, pearson_graph, output_dir, name="pearson"
        )

        cosine_matrix = raw_activation_cosine_similarity_matrix(states)
        cosine_report, cosine_graph = report_graph(
            cosine_matrix, formulas, metric_name="cosine", **graph_kwargs
        )
        written_paths[f"{task_name}/cosine"] = save_graph_report(
            cosine_report, cosine_graph, output_dir, name="cosine"
        )

        if "base" not in task_results:
            warnings.warn(f"skipping task {task_name}: no base capture found")
            continue

        base_states = task_results["base"].states

        base_pearson_matrix = raw_activation_correlation_matrix(base_states)
        base_pearson_report, base_pearson_graph = report_graph(
            base_pearson_matrix, formulas, metric_name="base_pearson", **graph_kwargs
        )
        written_paths[f"{task_name}/base_pearson"] = save_graph_report(
            base_pearson_report, base_pearson_graph, output_dir, name="base_pearson"
        )

        base_cosine_matrix = raw_activation_cosine_similarity_matrix(base_states)
        base_cosine_report, base_cosine_graph = report_graph(
            base_cosine_matrix, formulas, metric_name="base_cosine", **graph_kwargs
        )
        written_paths[f"{task_name}/base_cosine"] = save_graph_report(
            base_cosine_report, base_cosine_graph, output_dir, name="base_cosine"
        )

    return written_paths


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
    parser.add_argument(
        "--community-method",
        choices=["louvain", "leiden"],
        default="louvain",
        help="Community detection method. Louvain applies relu preprocessing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_project_graph_reports(
        args.results_path,
        relu_sparsify=args.relu,
        community_method=args.community_method,
    )
