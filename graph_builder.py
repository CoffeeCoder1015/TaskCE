from pathlib import Path

from analysis.activation_diagnostics import raw_activation_correlation_matrix, raw_activation_cosine_similarity_matrix
from analysis.graph import (
    analyze_communities,
    analyze_community_atomic_frequencies,
    analyze_degrees,
    analyze_hubs,
    analyze_k_core,
    analyze_k_core_atomic_frequencies,
    build_graph,
    local_top_neighbors_union,
    local_top_percent_union,
    relu,
    render_full_report,
    render_k_core_full_report,
    render_summary_report,
    save_graph_plot,
    save_k_core_plot,
    zero_diag,
)
from capture.saving import load_captured_activations
import pandas as pd

results_path = Path("results")

captured = load_captured_activations(results_path)

top_percent = 0.001
min_neighbors = 3
k_core_range = range(1, 8)
pipeline = "leiden"
layout_backend = "spring"
negative_mode = "render_only"
sparsifying_functions = {
    "louvain": lambda adj_matrix: local_top_neighbors_union(
        relu(zero_diag(adj_matrix)),
        top_percent,
        min_neighbors=min_neighbors,
    ),
    "leiden": lambda adj_matrix: local_top_neighbors_union(
        zero_diag(adj_matrix),
        top_percent,
        min_neighbors=min_neighbors,
    ),
    "louvain_threshold": lambda adj_matrix: local_top_percent_union(
        relu(zero_diag(adj_matrix)),
        top_percent,
    ),
    "leiden_threshold": lambda adj_matrix: local_top_percent_union(
        zero_diag(adj_matrix),
        top_percent,
    ),
}


def graph_pipeline(
    adj_matrix,
    searched_formulas,
    output_dir,
    name,
    *,
    layout_backend=layout_backend,
    negative_mode=negative_mode,
):
    G = build_graph(adj_matrix,sparsifying_functions[pipeline],searched_formulas)
    communities = analyze_communities(G,pipeline)
    degrees = analyze_degrees(G)
    hubs = analyze_hubs(degrees,communities)
    community_atomics = analyze_community_atomic_frequencies(G,communities) 
    k_core_nodes_df, k_core_stats_df = analyze_k_core(G, k_range=k_core_range)
    k_core_atomics = analyze_k_core_atomic_frequencies(G,k_core_nodes_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_graph_plot(
        G,
        communities,
        output_dir / f"{name}_graph.png",
        title=f"{name} graph",
        layout_backend=layout_backend,
        negative_mode=negative_mode,
    )
    save_k_core_plot(
        G,
        k_core_nodes_df,
        output_dir / f"{name}_k_core_graph.png",
        title=f"{name} k-core layers",
        layout_backend=layout_backend,
        negative_mode=negative_mode,
    )
    (output_dir / f"{name}_cluster_report_full.md").write_text(
        render_full_report(communities, searched_formulas, degrees),
        encoding="utf-8",
    )
    (output_dir / f"{name}_cluster_report_summary.md").write_text(
        render_summary_report(
            communities,
            searched_formulas,
            degrees,
            community_atomics,
            hubs,
            k_core_stats_df,
        ),
        encoding="utf-8",
    )
    (output_dir / f"{name}_k_core_report_full.md").write_text(
        render_k_core_full_report(
            k_core_nodes_df,
            k_core_stats_df,
            k_core_atomics,
        ),
        encoding="utf-8",
    )



for task in captured:
    task_data = captured[task]
    for checkpoint in ["finetuned","base"]:
        activations = task_data[checkpoint].states
        formulas = pd.read_csv(results_path / f"{task}_beam_results.csv")
        
        pearson = raw_activation_correlation_matrix(activations)
        cosine = raw_activation_cosine_similarity_matrix(activations)

        # pearson
        graph_pipeline(
            pearson,
            formulas,
            results_path / "graph",
            f"{task}_{checkpoint}_pearson_{pipeline}",
        )
        
        #cosine
        graph_pipeline(
            cosine,
            formulas,
            results_path / "graph",
            f"{task}_{checkpoint}_cosine_{pipeline}",
        )
