import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

from analysis.activation_diagnostics import (
    raw_activation_correlation_matrix,
    raw_activation_cosine_similarity_matrix,
)
from analysis.graph import (
    analyze_communities,
    analyze_degrees,
    analyze_hubs,
    analyze_k_core,
    analyze_k_core_stats,
    build_graph,
    local_top_percent_union,
    relu,
    save_graph_outputs,
    zero_diag,
)
from capture import load_captured_activations


DEFAULT_LOCAL_TOP_PERCENTILE = 0.001
DEFAULT_K_CORE_START = 3
DEFAULT_COMMUNITY_SEED = 0
HUB_REPORT_LIMIT = 10

SIGNED_LOCAL_TOP_PERCENT = lambda percent: (
    lambda adj_matrix: local_top_percent_union(zero_diag(adj_matrix), percent)
)
RELU_LOCAL_TOP_PERCENT = lambda percent: (
    lambda adj_matrix: local_top_percent_union(relu(zero_diag(adj_matrix)), percent)
)

COMMUNITY_PIPELINES = {
    "louvain": {
        "community_algorithm": "louvain",
        "sparsifier": RELU_LOCAL_TOP_PERCENT,
        "sparsification": "zero_diag -> relu -> local_top_percent_union",
    },
    "leiden": {
        "community_algorithm": "leiden",
        "sparsifier": SIGNED_LOCAL_TOP_PERCENT,
        "sparsification": "zero_diag -> local_top_percent_union",
    },
}


def add_formula_node_attributes(graph, formula_frame):
    node_attributes = {
        row["neuron"]: {
            column: row[column]
            for column in formula_frame.columns
            if column != "neuron"
        }
        for row in formula_frame.to_dict("records")
    }
    nx.set_node_attributes(graph, node_attributes)


def analyze_graph(
    adj_matrix,
    formulas,
    *,
    metric_name="graph",
    local_top_percentile=DEFAULT_LOCAL_TOP_PERCENTILE,
    k_core_start=DEFAULT_K_CORE_START,
    community_seed=DEFAULT_COMMUNITY_SEED,
    community_method="louvain",
):
    formula_frame = pd.DataFrame(formulas)
    pipeline = COMMUNITY_PIPELINES[community_method]
    sparsify_fn = pipeline["sparsifier"](local_top_percentile)

    graph = build_graph(adj_matrix, sparsify_fn, formula_frame)
    add_formula_node_attributes(graph, formula_frame)

    k_core_df = analyze_k_core(graph)
    nx.set_node_attributes(
        graph,
        k_core_df.set_index("node")["core_number"].to_dict(),
        "core_number",
    )

    k_core_graph = nx.k_core(graph, k=k_core_start).copy()
    communities_df = analyze_communities(
        k_core_graph,
        method=pipeline["community_algorithm"],
        seed=community_seed,
    )
    degrees_df = analyze_degrees(k_core_graph)
    hubs_df = analyze_hubs(degrees_df, communities_df, limit=HUB_REPORT_LIMIT)
    full_degrees_df = analyze_degrees(graph)
    k_core_stats_df = analyze_k_core_stats(k_core_df, full_degrees_df)

    nx.set_node_attributes(
        k_core_graph,
        communities_df.set_index("node")["community"].to_dict(),
        "community",
    )

    report = {
        "metric_name": metric_name,
        "options": {
            "local_top_percentile": local_top_percentile,
            "k_core_start": k_core_start,
            "community_seed": community_seed,
            "community_algorithm": pipeline["community_algorithm"],
            "sparsification": pipeline["sparsification"],
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


def report_graph(
    adj_matrix,
    formulas,
    output_dir,
    name,
    *,
    metric_name="graph",
    local_top_percentile=DEFAULT_LOCAL_TOP_PERCENTILE,
    k_core_start=DEFAULT_K_CORE_START,
    community_seed=DEFAULT_COMMUNITY_SEED,
    community_method="louvain",
):
    report, k_core_graph = analyze_graph(
        adj_matrix,
        formulas,
        metric_name=metric_name,
        local_top_percentile=local_top_percentile,
        k_core_start=k_core_start,
        community_seed=community_seed,
        community_method=community_method,
    )
    report["paths"] = save_graph_outputs(
        report["graph"],
        k_core_graph,
        output_dir,
        name,
        title=f"{metric_name} neuron graph (k-core {k_core_start})",
        seed=community_seed,
        communities_df=report["communities_df"],
        formula_dataframe=report["formula_dataframe"],
        degrees_df=report["degrees_df"],
        k_core_df=report["k_core_df"],
        hubs_df=report["hubs_df"],
        k_core_stats_df=report["k_core_stats_df"],
    )
    return report, k_core_graph


def build_project_graph_reports(
    results_path,
    *,
    local_top_percentile=DEFAULT_LOCAL_TOP_PERCENTILE,
    k_core_start=DEFAULT_K_CORE_START,
    community_seed=DEFAULT_COMMUNITY_SEED,
    community_method="louvain",
):
    results_path = Path(results_path)
    results_root = results_path if results_path.is_dir() else results_path.parent.parent
    captured_results = load_captured_activations(results_path)
    written_paths = {}

    graph_kwargs = {
        "local_top_percentile": local_top_percentile,
        "k_core_start": k_core_start,
        "community_seed": community_seed,
        "community_method": community_method,
    }

    for task_name in sorted(captured_results):
        formulas = pd.read_csv(results_root / f"{task_name}_beam_results.csv")
        output_dir = results_root / task_name

        states = captured_results[task_name]["finetuned"].states
        pearson_report, _ = report_graph(
            raw_activation_correlation_matrix(states),
            formulas,
            output_dir,
            "pearson",
            metric_name="pearson",
            **graph_kwargs,
        )
        cosine_report, _ = report_graph(
            raw_activation_cosine_similarity_matrix(states),
            formulas,
            output_dir,
            "cosine",
            metric_name="cosine",
            **graph_kwargs,
        )
        written_paths[f"{task_name}/pearson"] = pearson_report["paths"]
        written_paths[f"{task_name}/cosine"] = cosine_report["paths"]

        base_states = captured_results[task_name]["base"].states
        base_pearson_report, _ = report_graph(
            raw_activation_correlation_matrix(base_states),
            formulas,
            output_dir,
            "base_pearson",
            metric_name="base_pearson",
            **graph_kwargs,
        )
        base_cosine_report, _ = report_graph(
            raw_activation_cosine_similarity_matrix(base_states),
            formulas,
            output_dir,
            "base_cosine",
            metric_name="base_cosine",
            **graph_kwargs,
        )
        written_paths[f"{task_name}/base_pearson"] = base_pearson_report["paths"]
        written_paths[f"{task_name}/base_cosine"] = base_cosine_report["paths"]

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
        "--community-algorithm",
        dest="community_method",
        default="louvain",
        help=(
            "Community pipeline. louvain uses zero_diag -> relu -> "
            "local_top_percent_union; leiden uses zero_diag -> "
            "local_top_percent_union."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_project_graph_reports(
        args.results_path,
        community_method=args.community_method,
    )
