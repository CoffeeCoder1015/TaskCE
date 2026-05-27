from collections import Counter
from pathlib import Path
import re

import networkx as nx
import numpy as np
import pandas as pd
from sympy import Symbol
from sympy.logic.boolalg import Not as SymNot
from sympy.parsing.sympy_parser import parse_expr


ATOMIC_PATTERN_TEXT = r"[A-Za-z_][\w-]*:(?:<[^>]+>|[^\s()]+)"
ATOMIC_PATTERN = re.compile(ATOMIC_PATTERN_TEXT)
NEGATED_ATOMIC_PATTERN = re.compile(rf"\(\s*NOT\s+({ATOMIC_PATTERN_TEXT})\s*\)")


def zero_diag(adj_matrix):
    matrix = np.asarray(adj_matrix, dtype=float).copy()
    np.fill_diagonal(matrix, 0.0)
    return matrix


def relu(adj_matrix):
    matrix = np.asarray(adj_matrix, dtype=float).copy()
    matrix[matrix < 0.0] = 0.0
    return matrix


def top_k(adj_matrix, k):
    matrix = np.asarray(adj_matrix, dtype=float)
    sparse = np.zeros_like(matrix, dtype=float)
    rows, columns = np.triu_indices(matrix.shape[0], k=1)
    strengths = np.abs(matrix[rows, columns])
    selected = np.argsort(strengths)[-int(k):]
    selected = selected[strengths[selected] > 0.0]
    selected_rows = rows[selected]
    selected_columns = columns[selected]
    sparse[selected_rows, selected_columns] = matrix[selected_rows, selected_columns]
    sparse[selected_columns, selected_rows] = matrix[selected_columns, selected_rows]
    return sparse


def top_percent(adj_matrix, percent):
    matrix = np.asarray(adj_matrix, dtype=float)
    strengths = np.abs(matrix).flatten()
    threshold = np.quantile(strengths[strengths > 0.0], 1.0 - float(percent))
    return np.where(np.abs(matrix) >= threshold, matrix, 0.0)


def local_top_percent_union(adj_matrix, percent):
    matrix = np.asarray(adj_matrix, dtype=float)
    strengths = np.abs(matrix)
    row_sparse = np.zeros_like(matrix, dtype=float)

    for row_index, row in enumerate(strengths):
        nonzero_weights = row[row > 0.0]
        if nonzero_weights.size:
            threshold = np.quantile(nonzero_weights, 1.0 - float(percent))
            keep_columns = row >= threshold
            row_sparse[row_index, keep_columns] = matrix[row_index, keep_columns]

    union_mask = (row_sparse != 0.0) | (row_sparse.T != 0.0)
    return np.where(union_mask, matrix, 0.0)


def signed_atomic_counts(formula):
    counts = Counter()
    formula_text = str(formula)

    for atomic in NEGATED_ATOMIC_PATTERN.findall(formula_text):
        counts[f"NOT {atomic}"] += 1

    formula_text = NEGATED_ATOMIC_PATTERN.sub(" ", formula_text)
    counts.update(ATOMIC_PATTERN.findall(formula_text))
    return dict(counts)


def build_graph(adj_matrix, sparsify_fn, formula_dataframe):
    matrix = np.asarray(adj_matrix, dtype=float)
    
    sparse_matrix = sparsify_fn(matrix)
    # Sparsify_fn should set all unimporant connections (whatever that criteria is) to 0

    graph = nx.from_numpy_array(sparse_matrix)
    node_metadata = {}
    
    for row in formula_dataframe.iloc:
        node_metadata[row.neuron] = {
            "formula": row.formula,
            "iou": row.iou,
            "signed_atomic_counts": signed_atomic_counts(row.formula),
        }

    nx.set_node_attributes(graph,node_metadata)
    
    return graph


def analyze_k_core(graph, k_range=range(2, 8)):
    node_rows = []
    stat_rows = []
    for k in k_range:
        k_core_graph = nx.k_core(graph, k=k).copy()
        node_rows.extend(
            {"k": k, "node": node}
            for node in sorted(k_core_graph.nodes)
        )
        degrees_df = analyze_degrees(k_core_graph)
        stat_rows.append(
            {
                "k": k,
                "node_count": k_core_graph.number_of_nodes(),
                "edge_count": k_core_graph.number_of_edges(),
                "avg_degree": degrees_df["degree"].mean(),
                "max_degree": degrees_df["degree"].max(),
                "avg_positive_edge_count": degrees_df["positive_edge_count"].mean(),
                "max_positive_edge_count": degrees_df["positive_edge_count"].max(),
                "avg_negative_edge_count": degrees_df["negative_edge_count"].mean(),
                "max_negative_edge_count": degrees_df["negative_edge_count"].max(),
                "avg_positive_weighted_degree": degrees_df[
                    "positive_weighted_degree"
                ].mean(),
                "max_positive_weighted_degree": degrees_df[
                    "positive_weighted_degree"
                ].max(),
                "avg_negative_weighted_degree": degrees_df[
                    "negative_weighted_degree"
                ].mean(),
                "min_negative_weighted_degree": degrees_df[
                    "negative_weighted_degree"
                ].min(),
            }
        )
    return (
        pd.DataFrame(node_rows, columns=["k", "node"]),
        pd.DataFrame(
            stat_rows,
            columns=[
                "k",
                "node_count",
                "edge_count",
                "avg_degree",
                "max_degree",
                "avg_positive_edge_count",
                "max_positive_edge_count",
                "avg_negative_edge_count",
                "max_negative_edge_count",
                "avg_positive_weighted_degree",
                "max_positive_weighted_degree",
                "avg_negative_weighted_degree",
                "min_negative_weighted_degree",
            ],
        ),
    )


def analyze_degrees(graph):
    rows = []
    for node, degree in graph.degree():
        edge_weights = [
            edge_data["weight"]
            for _, _, edge_data in graph.edges(node, data=True)
        ]
        positive_weights = [weight for weight in edge_weights if weight > 0.0]
        negative_weights = [weight for weight in edge_weights if weight < 0.0]
        rows.append(
            {
                "node": node,
                "degree": degree,
                "positive_edge_count": len(positive_weights),
                "negative_edge_count": len(negative_weights),
                "positive_weighted_degree": sum(positive_weights),
                "negative_weighted_degree": sum(negative_weights),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "node",
            "degree",
            "positive_edge_count",
            "negative_edge_count",
            "positive_weighted_degree",
            "negative_weighted_degree",
        ],
    )


def analyze_communities(graph, method="louvain", seed=0, backend="cugraph"):
    if method == "louvain":
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            seed=seed,
        )
    elif method == "leiden":
        communities = nx.community.leiden_communities(
            graph,
            weight="weight",
            seed=seed,
            backend=backend,
        )
    else:
        raise NotImplementedError(f"Unsupported community method: {method}")

    return pd.DataFrame(
        {"node": node, "community": community_id}
        for community_id, community in enumerate(communities)
        for node in community
    )


def sympy_formula_text(formula, placeholders):
    rewritten = str(formula)
    for atomic, placeholder in placeholders.items():
        rewritten = rewritten.replace(atomic, placeholder)
    rewritten = re.sub(r"\bAND\b", "&", rewritten)
    rewritten = re.sub(r"\bOR\b", "|", rewritten)
    rewritten = re.sub(r"\bNOT\b", "~", rewritten)
    return rewritten


def signed_atomics(expr, negated=False):
    if expr.is_Symbol:
        atomic = str(expr)
        signed_atomic = f"NOT {atomic}" if negated else atomic
        return [(atomic, signed_atomic)]
    if expr.func is SymNot:
        return signed_atomics(expr.args[0], not negated)
    return [
        signed_atomic
        for arg in expr.args
        for signed_atomic in signed_atomics(arg, negated)
    ]


def analyze_community_atomic_frequencies(graph, communities_df):
    rows = []
    for row in communities_df.itertuples(index=False):
        for signed_atomic, count in graph.nodes[row.node][
            "signed_atomic_counts"
        ].items():
            rows.append(
                {
                    "community": row.community,
                    "node": row.node,
                    "signed_atomic": signed_atomic,
                    "count": count,
                }
            )

    atomic_counts = pd.DataFrame(
        rows,
        columns=["community", "node", "signed_atomic", "count"],
    )
    return (
        atomic_counts.groupby(["community", "signed_atomic"], as_index=False)
        .agg(
            frequency=("count", "sum"),
            neuron_presence=("node", "nunique"),
        )
        .sort_values(
            ["community", "frequency", "neuron_presence", "signed_atomic"],
            ascending=[True, False, False, True],
        )
        .reset_index(drop=True)
    )


def analyze_k_core_atomic_frequencies(graph, k_core_nodes_df):
    rows = []
    for row in k_core_nodes_df.itertuples(index=False):
        for signed_atomic, count in graph.nodes[row.node][
            "signed_atomic_counts"
        ].items():
            rows.append(
                {
                    "k": row.k,
                    "node": row.node,
                    "signed_atomic": signed_atomic,
                    "count": count,
                }
            )

    atomic_counts = pd.DataFrame(
        rows,
        columns=["k", "node", "signed_atomic", "count"],
    )
    k_core_atomics = (
        atomic_counts.groupby(["k", "signed_atomic"], as_index=False)
        .agg(
            frequency=("count", "sum"),
            neuron_presence=("node", "nunique"),
        )
        .sort_values(
            ["k", "neuron_presence", "frequency", "signed_atomic"],
            ascending=[True, False, False, True],
        )
        .reset_index(drop=True)
    )
    k_core_atomics["rank"] = k_core_atomics.groupby("k").cumcount() + 1
    return k_core_atomics[
        ["k", "rank", "signed_atomic", "frequency", "neuron_presence"]
    ]


def analyze_hubs(degrees_df, communities_df, limit=5):
    community_degrees = communities_df.merge(degrees_df, on="node")
    positive_hubs = (
        community_degrees.sort_values(
            [
                "community",
                "positive_weighted_degree",
                "positive_edge_count",
                "node",
            ],
            ascending=[True, False, False, True],
        )
        .groupby("community", as_index=False)
        .head(limit)
        .copy()
    )
    positive_hubs["hub_type"] = "positive"
    positive_hubs["rank"] = positive_hubs.groupby("community").cumcount() + 1

    negative_hubs = (
        community_degrees.sort_values(
            [
                "community",
                "negative_weighted_degree",
                "negative_edge_count",
                "node",
            ],
            ascending=[True, True, False, True],
        )
        .groupby("community", as_index=False)
        .head(limit)
        .copy()
    )
    negative_hubs["hub_type"] = "negative"
    negative_hubs["rank"] = negative_hubs.groupby("community").cumcount() + 1

    hubs = pd.concat([positive_hubs, negative_hubs], ignore_index=True)
    return hubs[
        [
            "community",
            "hub_type",
            "rank",
            "node",
            "degree",
            "positive_edge_count",
            "negative_edge_count",
            "positive_weighted_degree",
            "negative_weighted_degree",
        ]
    ].reset_index(drop=True)

def save_graph_plot(graph, communities_df, output_path, *, title, seed=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    community_by_node = communities_df.set_index("node")["community"].to_dict()
    community_ids = sorted(communities_df["community"].unique())
    color_map = plt.colormaps["tab20"]
    colors = {
        community_id: color_map(index % color_map.N)
        for index, community_id in enumerate(community_ids)
    }
    positions = nx.spring_layout(graph, weight="weight", seed=seed)
    node_colors = [colors[community_by_node[node]] for node in graph.nodes]

    plt.figure(figsize=(24, 20))
    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color="#8c8c8c",
        alpha=0.45,
        width=1.1,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=420,
        edgecolors="black",
        linewidths=0.6,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: str(node) for node in graph.nodes},
        font_size=8,
    )
    plt.legend(
        handles=[
            Patch(
                facecolor=colors[community_id],
                edgecolor="black",
                label=f"Community {community_id}",
            )
            for community_id in community_ids
        ],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=11,
    )
    plt.title(title, fontsize=22, pad=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def save_k_core_plot(graph, k_core_nodes_df, output_path, *, title, seed=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    k_values = sorted(k_core_nodes_df["k"].unique())
    color_map = plt.colormaps["viridis"]
    colors = {
        k: color_map(index / max(len(k_values) - 1, 1))
        for index, k in enumerate(k_values)
    }
    positions = nx.spring_layout(graph, weight="weight", seed=seed)

    plt.figure(figsize=(24, 20))
    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color="#8c8c8c",
        alpha=0.45,
        width=1.1,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color="#f2f2f2",
        node_size=300,
        edgecolors="#555555",
        linewidths=0.6,
    )
    for index, k in enumerate(k_values):
        k_nodes = k_core_nodes_df[k_core_nodes_df["k"] == k]["node"].tolist()
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=k_nodes,
            node_color="none",
            node_size=460 + index * 150,
            edgecolors=colors[k],
            linewidths=2.0,
        )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: str(node) for node in graph.nodes},
        font_size=8,
    )
    plt.legend(
        handles=[
            Patch(
                facecolor="none",
                edgecolor=colors[k],
                label=f"k={k}",
            )
            for k in k_values
        ],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=11,
    )
    plt.title(title, fontsize=22, pad=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def neuron_report_frame(communities_df, formula_dataframe, degrees_df):
    formula_df = formula_dataframe.rename(columns={"neuron": "node"})
    return (
        communities_df.merge(formula_df, on="node")
        .merge(degrees_df, on="node")
        .sort_values(["community", "node"])
        .reset_index(drop=True)
    )


def markdown_cell(value):
    if pd.isna(value):
        return ""
    if type(value) in (float, np.float16, np.float32, np.float64):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(dataframe, columns):
    if not columns or dataframe.empty:
        return ["No data."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(":---" for _ in columns) + " |",
    ]
    for row in dataframe[columns].to_dict("records"):
        lines.append(
            "| "
            + " | ".join(markdown_cell(row[column]) for column in columns)
            + " |"
        )
    return lines


def render_full_report(communities_df, formula_dataframe, degrees_df):
    report_df = neuron_report_frame(
        communities_df,
        formula_dataframe,
        degrees_df,
    )
    base_columns = [
        "community",
        "family",
        "node",
        "formula",
        "iou",
        "degree",
        "positive_edge_count",
        "negative_edge_count",
        "positive_weighted_degree",
        "negative_weighted_degree",
    ]
    columns = [column for column in base_columns if column in report_df.columns]
    lines = ["# Cluster Report Full", ""]

    group_columns = ["community"]
    if "family" in report_df.columns:
        group_columns.append("family")

    for group_key, group_df in report_df.groupby(group_columns, sort=True):
        title = f"Community {group_key}"
        if len(group_columns) > 1:
            title = f"Community {group_key[0]} / Family {group_key[1]}"
        lines.extend([f"## {title}", ""])
        lines.extend(markdown_table(group_df, columns))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_summary_report(
    communities_df,
    formula_dataframe,
    degrees_df,
    community_atomic_df,
    hubs_df,
    k_core_stats_df,
):
    report_df = neuron_report_frame(
        communities_df,
        formula_dataframe,
        degrees_df,
    )
    formula_columns = [
        column
        for column in [
            "node",
            "formula",
            "iou",
            "degree",
            "positive_edge_count",
            "negative_edge_count",
            "positive_weighted_degree",
            "negative_weighted_degree",
        ]
        if column in report_df.columns
    ]
    atomic_columns = [
        column
        for column in ["signed_atomic", "frequency", "neuron_presence"]
        if column in community_atomic_df.columns
    ]
    hub_columns = [
        "hub_type",
        "rank",
        "node",
        "degree",
        "positive_edge_count",
        "negative_edge_count",
        "positive_weighted_degree",
        "negative_weighted_degree",
    ]
    stat_columns = [
        "k",
        "node_count",
        "edge_count",
        "avg_degree",
        "max_degree",
        "avg_positive_edge_count",
        "max_positive_edge_count",
        "avg_negative_edge_count",
        "max_negative_edge_count",
        "avg_positive_weighted_degree",
        "max_positive_weighted_degree",
        "avg_negative_weighted_degree",
        "min_negative_weighted_degree",
    ]
    lines = ["# Cluster Report Summary", ""]

    for community_id, community_df in report_df.groupby("community", sort=True):
        top_formulas = community_df.sort_values(
            [
                "positive_weighted_degree",
                "negative_weighted_degree",
                "iou",
                "node",
            ],
            ascending=[False, False, False, True],
        ).head(5)
        top_atomics = community_atomic_df
        if "community" in community_atomic_df.columns:
            top_atomics = community_atomic_df[
                community_atomic_df["community"] == community_id
            ].head(10)
        community_hubs = hubs_df[hubs_df["community"] == community_id]

        lines.extend([f"## Community {community_id}", ""])
        lines.extend(["### Top Formulas", ""])
        lines.extend(markdown_table(top_formulas, formula_columns))
        lines.extend(["", "### Top Atomics", ""])
        lines.extend(markdown_table(top_atomics, atomic_columns))
        lines.extend(["", "### Hubs", ""])
        lines.extend(markdown_table(community_hubs, hub_columns))
        lines.append("")

    lines.extend(["## K-Core Sweep", ""])
    lines.extend(markdown_table(k_core_stats_df, stat_columns))
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_k_core_full_report(k_core_nodes_df, k_core_stats_df, k_core_atomic_df):
    node_columns = ["node"]
    stat_columns = [
        "k",
        "node_count",
        "edge_count",
        "avg_degree",
        "max_degree",
        "avg_positive_edge_count",
        "max_positive_edge_count",
        "avg_negative_edge_count",
        "max_negative_edge_count",
        "avg_positive_weighted_degree",
        "max_positive_weighted_degree",
        "avg_negative_weighted_degree",
        "min_negative_weighted_degree",
    ]
    union_columns = ["rank", "signed_atomic", "neuron_presence", "frequency"]
    lines = ["# K-Core Report Full", ""]

    for k in k_core_stats_df.sort_values("k")["k"]:
        core_df = k_core_nodes_df[k_core_nodes_df["k"] == k].sort_values("node")
        core_stats = k_core_stats_df[k_core_stats_df["k"] == k]
        core_atomics = k_core_atomic_df[k_core_atomic_df["k"] == k]

        lines.extend([f"## k={k}", ""])
        lines.extend(["### Nodes", ""])
        lines.extend(markdown_table(core_df, node_columns))
        lines.extend(["", "### Degree Stats", ""])
        lines.extend(markdown_table(core_stats, stat_columns))
        lines.extend(["", "### Shared Signed Atomics", ""])
        lines.extend(markdown_table(core_atomics, union_columns))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
