from collections import Counter
from pathlib import Path
import re

import networkx as nx
import numpy as np
import pandas as pd
from sympy import Symbol
from sympy.logic.boolalg import Not as SymNot
from sympy.parsing.sympy_parser import parse_expr


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


def build_graph(adj_matrix, sparsify_fn, formula_dataframe):
    matrix = np.asarray(adj_matrix, dtype=float)
    
    sparse_matrix = sparsify_fn(matrix)
    # Sparsify_fn should set all unimporant connections (whatever that criteria is) to 0

    graph = nx.from_numpy_array(sparse_matrix)
    node_metadata = {}
    
    for row in formula_dataframe.iloc:
        node_metadata[row.neuron] = {"formula":row.formula,"iou":row.iou}

    nx.set_node_attributes(graph,node_metadata)
    
    return graph


def analyze_k_core(graph):
    core_numbers = nx.core_number(graph)
    return pd.DataFrame(
        {"node": node, "core_number": core_number}
        for node, core_number in core_numbers.items()
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


def tokenizer_atomic_symbols(tokenizer, labels):
    vocab_tokens = sorted(tokenizer.get_vocab())
    atomics = [f"{label}:{token}" for label in labels for token in vocab_tokens]
    placeholders = {
        atomic: f"atomic_{index}"
        for index, atomic in enumerate(sorted(atomics, key=len, reverse=True))
    }
    symbols = {
        placeholder: Symbol(atomic)
        for atomic, placeholder in placeholders.items()
    }
    return placeholders, symbols


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


def analyze_atomic_frequencies(formula_dataframe, tokenizer, labels):
    placeholders, symbols = tokenizer_atomic_symbols(tokenizer, labels)
    rows = []

    for formula_row in formula_dataframe.to_dict("records"):
        expression = parse_expr(
            sympy_formula_text(formula_row["formula"], placeholders),
            local_dict=symbols,
            evaluate=False,
        )
        counts = Counter(signed_atomics(expression))
        for (atomic, signed_atomic), count in counts.items():
            rows.append(
                {
                    "node": formula_row["neuron"],
                    "atomic": atomic,
                    "signed_atomic": signed_atomic,
                    "count": int(count),
                }
            )

    return pd.DataFrame(rows, columns=["node", "atomic", "signed_atomic", "count"])


def analyze_community_atomic_frequencies(communities_df, atomic_df):
    joined = communities_df.merge(atomic_df, on="node")
    grouped = (
        joined.groupby(["community", "signed_atomic"], as_index=False)
        .agg(
            atomic=("atomic", "first"),
            frequency=("count", "sum"),
            neuron_presence=("node", "nunique"),
        )
        .sort_values(
            ["community", "frequency", "neuron_presence", "signed_atomic"],
            ascending=[True, False, False, True],
        )
    )
    return grouped[
        ["community", "atomic", "signed_atomic", "frequency", "neuron_presence"]
    ].reset_index(drop=True)


def analyze_hubs(degrees_df, communities_df, limit=5):
    hubs = (
        communities_df.merge(degrees_df, on="node")
        .sort_values(
            [
                "community",
                "positive_weighted_degree",
                "negative_weighted_degree",
                "positive_edge_count",
                "negative_edge_count",
                "node",
            ],
            ascending=[True, False, False, False, True, True],
        )
        .groupby("community", as_index=False)
        .head(int(limit))
        .copy()
    )
    hubs["rank"] = hubs.groupby("community").cumcount() + 1
    return hubs[
        [
            "community",
            "rank",
            "node",
            "degree",
            "positive_edge_count",
            "negative_edge_count",
            "positive_weighted_degree",
            "negative_weighted_degree",
        ]
    ].reset_index(drop=True)


def analyze_k_core_union(k_core_df, atomic_df):
    joined = k_core_df.merge(atomic_df, on="node")
    union = (
        joined.groupby(["core_number", "signed_atomic"], as_index=False)
        .agg(
            atomic=("atomic", "first"),
            frequency=("count", "sum"),
            neuron_presence=("node", "nunique"),
        )
        .sort_values(
            ["core_number", "neuron_presence", "frequency", "signed_atomic"],
            ascending=[True, False, False, True],
        )
        .copy()
    )
    union["rank"] = union.groupby("core_number").cumcount() + 1
    return union[
        [
            "core_number",
            "rank",
            "atomic",
            "signed_atomic",
            "frequency",
            "neuron_presence",
        ]
    ].reset_index(drop=True)


def analyze_k_core_stats(k_core_df, degrees_df):
    joined = k_core_df.merge(degrees_df, on="node")
    return (
        joined.groupby("core_number", as_index=False)
        .agg(
            node_count=("node", "nunique"),
            avg_degree=("degree", "mean"),
            max_degree=("degree", "max"),
            avg_positive_edge_count=("positive_edge_count", "mean"),
            max_positive_edge_count=("positive_edge_count", "max"),
            avg_negative_edge_count=("negative_edge_count", "mean"),
            max_negative_edge_count=("negative_edge_count", "max"),
            avg_positive_weighted_degree=("positive_weighted_degree", "mean"),
            max_positive_weighted_degree=("positive_weighted_degree", "max"),
            avg_negative_weighted_degree=("negative_weighted_degree", "mean"),
            min_negative_weighted_degree=("negative_weighted_degree", "min"),
        )
        .sort_values("core_number")
        .reset_index(drop=True)
    )


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


def save_k_core_plot(graph, k_core_df, output_path, *, title, seed=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    core_by_node = k_core_df.set_index("node")["core_number"].to_dict()
    core_numbers = sorted(k_core_df["core_number"].unique())
    color_map = plt.colormaps["viridis"]
    colors = {
        core_number: color_map(index / max(len(core_numbers) - 1, 1))
        for index, core_number in enumerate(core_numbers)
    }
    positions = nx.spring_layout(graph, weight="weight", seed=seed)
    node_colors = [colors[core_by_node[node]] for node in graph.nodes]

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
                facecolor=colors[core_number],
                edgecolor="black",
                label=f"Core {core_number}",
            )
            for core_number in core_numbers
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


def neuron_report_frame(communities_df, formula_dataframe, degrees_df, k_core_df):
    formula_df = formula_dataframe.rename(columns={"neuron": "node"})
    return (
        communities_df.merge(formula_df, on="node")
        .merge(degrees_df, on="node")
        .merge(k_core_df, on="node")
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
    if not columns:
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


def render_full_report(communities_df, formula_dataframe, degrees_df, k_core_df):
    report_df = neuron_report_frame(
        communities_df,
        formula_dataframe,
        degrees_df,
        k_core_df,
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
        "core_number",
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
    k_core_df,
    community_atomic_df,
    hubs_df,
    k_core_stats_df,
    k_core_union_df,
):
    report_df = neuron_report_frame(
        communities_df,
        formula_dataframe,
        degrees_df,
        k_core_df,
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
            "core_number",
        ]
        if column in report_df.columns
    ]
    atomic_columns = [
        column
        for column in ["signed_atomic", "frequency", "neuron_presence"]
        if column in community_atomic_df.columns
    ]
    hub_columns = [
        "rank",
        "node",
        "degree",
        "positive_edge_count",
        "negative_edge_count",
        "positive_weighted_degree",
        "negative_weighted_degree",
    ]
    stat_columns = [
        "core_number",
        "node_count",
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
    union_columns = [
        column
        for column in [
            "core_number",
            "rank",
            "signed_atomic",
            "neuron_presence",
            "frequency",
        ]
        if column in k_core_union_df.columns
    ]
    lines = ["# Cluster Report Summary", ""]

    for community_id, community_df in report_df.groupby("community", sort=True):
        core_numbers = sorted(community_df["core_number"].unique())
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
        related_stats = k_core_stats_df[
            k_core_stats_df["core_number"].isin(core_numbers)
        ]
        related_union = k_core_union_df
        if "core_number" in k_core_union_df.columns:
            related_union = k_core_union_df[
                k_core_union_df["core_number"].isin(core_numbers)
            ].groupby("core_number", as_index=False).head(5)

        lines.extend([f"## Community {community_id}", ""])
        lines.extend(["### Top Formulas", ""])
        lines.extend(markdown_table(top_formulas, formula_columns))
        lines.extend(["", "### Top Atomics", ""])
        lines.extend(markdown_table(top_atomics, atomic_columns))
        lines.extend(["", "### Hubs", ""])
        lines.extend(markdown_table(community_hubs, hub_columns))
        lines.extend(["", "### Related K-Core Stats", ""])
        lines.extend(markdown_table(related_stats, stat_columns))
        lines.extend(["", "### Related K-Core Shared Atomics", ""])
        lines.extend(markdown_table(related_union, union_columns))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_k_core_full_report(k_core_df, k_core_stats_df, k_core_union_df):
    node_columns = ["node", "core_number"]
    stat_columns = [
        "node_count",
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

    for core_number, core_df in k_core_df.sort_values(["core_number", "node"]).groupby(
        "core_number",
        sort=True,
    ):
        core_stats = k_core_stats_df[k_core_stats_df["core_number"] == core_number]
        core_union = k_core_union_df[k_core_union_df["core_number"] == core_number]

        lines.extend([f"## Core {core_number}", ""])
        lines.extend(["### Nodes", ""])
        lines.extend(markdown_table(core_df, node_columns))
        lines.extend(["", "### Degree Stats", ""])
        lines.extend(markdown_table(core_stats, stat_columns))
        lines.extend(["", "### Shared Signed Atomics", ""])
        lines.extend(markdown_table(core_union, union_columns))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_reports(
    output_dir,
    name,
    communities_df,
    formula_dataframe,
    degrees_df,
    k_core_df,
    community_atomic_df,
    hubs_df,
    k_core_stats_df,
    k_core_union_df,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "full_report": output_dir / f"{name}_cluster_report_full.md",
        "summary_report": output_dir / f"{name}_cluster_report_summary.md",
        "k_core_full_report": output_dir / f"{name}_k_core_report_full.md",
    }
    paths["full_report"].write_text(
        render_full_report(communities_df, formula_dataframe, degrees_df, k_core_df),
        encoding="utf-8",
    )
    paths["summary_report"].write_text(
        render_summary_report(
            communities_df,
            formula_dataframe,
            degrees_df,
            k_core_df,
            community_atomic_df,
            hubs_df,
            k_core_stats_df,
            k_core_union_df,
        ),
        encoding="utf-8",
    )
    paths["k_core_full_report"].write_text(
        render_k_core_full_report(k_core_df, k_core_stats_df, k_core_union_df),
        encoding="utf-8",
    )
    return paths


def save_graph_outputs(
    graph,
    k_core_graph,
    output_dir,
    name,
    *,
    title,
    seed,
    communities_df,
    formula_dataframe,
    degrees_df,
    k_core_df,
    hubs_df,
    k_core_stats_df,
    community_atomic_df=None,
    k_core_union_df=None,
):
    graph_dir = Path(output_dir) / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    community_atomic_df = (
        pd.DataFrame() if community_atomic_df is None else community_atomic_df
    )
    k_core_union_df = pd.DataFrame() if k_core_union_df is None else k_core_union_df
    paths = {
        "png": graph_dir / f"{name}_neuron_graph.png",
        "k_core_png": graph_dir / f"{name}_k_core_graph.png",
        **save_reports(
            graph_dir,
            name,
            communities_df,
            formula_dataframe,
            degrees_df,
            k_core_df,
            community_atomic_df,
            hubs_df,
            k_core_stats_df,
            k_core_union_df,
        ),
    }
    save_graph_plot(k_core_graph, communities_df, paths["png"], title=title, seed=seed)
    save_k_core_plot(
        graph,
        k_core_df,
        paths["k_core_png"],
        title=f"{title} k-core",
        seed=seed,
    )
    return paths
