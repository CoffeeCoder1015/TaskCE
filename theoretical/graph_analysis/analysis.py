from collections import Counter
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd
from sympy.logic.boolalg import Not as SymNot


ATOMIC_PATTERN_TEXT = r"[A-Za-z_][\w-]*:(?:<[^>]+>|[^\s()]+)"
ATOMIC_PATTERN = re.compile(ATOMIC_PATTERN_TEXT)
NEGATED_ATOMIC_PATTERN = re.compile(rf"\(\s*NOT\s+({ATOMIC_PATTERN_TEXT})\s*\)")
SMALL_COMMUNITY_ID = -1

DEFAULT_LOCAL_TOP_PERCENT = 0.001
DEFAULT_MIN_NEIGHBORS = 3
DEFAULT_MIN_COMMUNITY_SIZE = 2
DEFAULT_PIPELINE = "louvain"
DEFAULT_PLOT_CONFIG = {
    "layout_backend": "spring",
    "negative_mode": "render_only",
}
ACTIVATION_DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data"
DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
COMPOSITIONAL_DATA_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "compositional_explanations"
    / "data"
)


def activation_path(
    model_id: str,
    dataset_name: str,
    layer: str,
    *,
    task_name: str | None = None,
    checkpoint_name: str | None = None,
) -> Path:
    """Return one generated activation file."""
    model_directory = ACTIVATION_DATA_DIRECTORY.joinpath(*model_id.split("/"))

    if task_name is None:
        # Select the base capture directory.
        capture_directory = model_directory / "base"
    elif checkpoint_name is not None:
        # Select the requested task checkpoint directory.
        capture_directory = model_directory / task_name / checkpoint_name
    else:
        # Select the numerically latest task checkpoint directory.
        task_directory = model_directory / task_name
        checkpoint_directories = []

        for checkpoint_directory in task_directory.iterdir():
            if checkpoint_directory.is_dir():
                checkpoint_directories.append(checkpoint_directory)

        capture_directory = max(
            checkpoint_directories,
            key=lambda checkpoint: int(
                checkpoint.name.removeprefix("checkpoint-")
            ),
        )

    activation_name = f"{dataset_name}_activation.pt"

    exact_layer_directory = capture_directory / layer
    if exact_layer_directory.is_dir():
        return exact_layer_directory / activation_name

    for saved_layer_directory in capture_directory.iterdir():
        if saved_layer_directory.name.endswith(f".{layer}"):
            return saved_layer_directory / activation_name

    raise KeyError(layer)


def compositional_result_path(task_name):
    """Return the upstream compositional data file consumed by this analysis."""
    return COMPOSITIONAL_DATA_DIRECTORY / f"{task_name}_beam_results.csv"

# Layout
GOLDEN_ANGLE_RADIANS = math.pi * (3.0 - math.sqrt(5.0))
SPRING_LAYOUT_K_SCALE = 2.4
NEGATIVE_EDGE_DISTANCE_SCALE = 0.9

# Node overlap
NODE_OVERLAP_DISTANCE_SCALE = 0.55
NODE_OVERLAP_MIN_DISTANCE = 0.015

# Density-scaled plot style
MAX_NODE_SIZE = 420.0
MIN_NODE_SIZE = 28.0
NODE_SIZE_DENSITY_SCALE = 3800.0
MAX_LABEL_FONT_SIZE = 8.5
MIN_LABEL_FONT_SIZE = 2.8
LABEL_FONT_DENSITY_SCALE = 42.0
MAX_NODE_LINEWIDTH = 0.8
MIN_NODE_LINEWIDTH = 0.2
NODE_LINEWIDTH_DENSITY_SCALE = 4.5
MAX_LABEL_ALPHA = 0.95
MIN_LABEL_ALPHA = 0.28
LABEL_ALPHA_DENSITY_SCALE = 16.0
MAX_EDGE_WIDTH = 1.15
MIN_EDGE_WIDTH = 0.25
EDGE_WIDTH_DENSITY_SCALE = 7.5


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


def local_top_neighbors_union(adj_matrix, percent, min_neighbors=1):
    matrix = np.asarray(adj_matrix, dtype=float)
    sparse = np.zeros_like(matrix, dtype=float)
    selected_edges = set()
    node_count = matrix.shape[0]

    for row_index, row in enumerate(matrix):
        candidates = []
        for column_index, weight in enumerate(row):
            weight = float(weight)
            if column_index == row_index or weight == 0.0:
                continue
            candidates.append((int(column_index), abs(weight)))

        if not candidates:
            continue

        percent_count = math.ceil((node_count - 1) * float(percent))
        keep_count = max(1, int(min_neighbors), percent_count)
        keep_count = min(keep_count, len(candidates))
        candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))

        for column_index, _ in candidates[:keep_count]:
            first, second = sorted((int(row_index), int(column_index)))
            selected_edges.add((first, second))

    for first, second in selected_edges:
        weight = matrix[first, second]
        sparse[first, second] = weight
        sparse[second, first] = weight

    return sparse


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

    nx.set_node_attributes(graph, node_metadata)
    
    return graph


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


def analyze_communities(
    graph,
    method="louvain",
    seed=0,
    backend="cugraph",
    min_community_size=2,
):
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

    communities_df = pd.DataFrame(
        {"node": node, "community": community_id}
        for community_id, community in enumerate(communities)
        for node in community
    )
    community_sizes = communities_df["community"].value_counts()
    small_communities = community_sizes[community_sizes < min_community_size].index
    communities_df.loc[
        communities_df["community"].isin(small_communities),
        "community",
    ] = SMALL_COMMUNITY_ID
    return communities_df


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


# Layout uses rank bands so edge magnitudes shape structure without overfitting tiny weight differences.
def rank_banded_layout_weights(graph, *, bands=4):
    strengths_by_edge = {
        (u, v): abs(float(edge_data["weight"]))
        for u, v, edge_data in graph.edges(data=True)
    }
    positive_strengths = sorted(
        {
            strength
            for strength in strengths_by_edge.values()
            if strength > 0.0
        }
    )
    if not positive_strengths:
        return {edge: 1.0 for edge in strengths_by_edge}

    band_count = max(1, int(bands))
    band_by_strength = {}
    for index, strength in enumerate(positive_strengths):
        band = 1 + math.floor(index * band_count / len(positive_strengths))
        band_by_strength[strength] = float(min(band, band_count))

    return {
        edge: band_by_strength[strength]
        for edge, strength in strengths_by_edge.items()
    }


# Alpha encodes edge strength so dense weak connections remain visible without drowning stronger structure.
def edge_alpha_values(graph, *, min_alpha=0.05, max_alpha=0.75):
    strengths_by_edge = {
        (u, v): abs(float(edge_data["weight"]))
        for u, v, edge_data in graph.edges(data=True)
    }
    positive_strengths = [
        strength
        for strength in strengths_by_edge.values()
        if strength > 0.0
    ]
    if not positive_strengths:
        return {edge: float(min_alpha) for edge in strengths_by_edge}

    low = min(positive_strengths)
    high = max(positive_strengths)
    if high == low:
        alpha = (float(min_alpha) + float(max_alpha)) / 2.0
        return {edge: alpha for edge in strengths_by_edge}

    return {
        edge: float(min_alpha)
        + (float(max_alpha) - float(min_alpha)) * ((strength - low) / (high - low))
        for edge, strength in strengths_by_edge.items()
    }


def banded_layout_graph(graph, *, bands=4, include_negative=False):
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(graph.nodes(data=True))
    band_by_edge = rank_banded_layout_weights(graph, bands=bands)

    for u, v, edge_data in graph.edges(data=True):
        weight = float(edge_data["weight"])
        sign = 1 if weight > 0.0 else -1 if weight < 0.0 else 0
        if sign < 0 and not include_negative:
            continue
        strength = abs(weight)
        if strength <= 0.0:
            continue
        layout_graph.add_edge(u, v, layout_weight=band_by_edge[(u, v)])

    return layout_graph


def normalize_positions(positions):
    if not positions:
        return positions

    nodes = list(positions)
    points = np.asarray([positions[node] for node in nodes], dtype=float)
    points -= points.mean(axis=0)
    span = np.ptp(points, axis=0).max()
    if span > 0.0:
        points /= span
    return {
        node: points[index]
        for index, node in enumerate(nodes)
    }


def relax_node_overlap(positions, *, node_count, max_passes=20):
    if len(positions) < 2:
        return positions

    nodes = list(positions)
    points = np.asarray([positions[node] for node in nodes], dtype=float)
    target_distance = max(
        NODE_OVERLAP_MIN_DISTANCE,
        NODE_OVERLAP_DISTANCE_SCALE / math.sqrt(max(int(node_count), 1)),
    )
    max_step = target_distance * 0.25

    for _ in range(int(max_passes)):
        movement = np.zeros_like(points)
        moved = False
        for first_index in range(len(points)):
            deltas = points[first_index] - points[first_index + 1 :]
            distances = np.linalg.norm(deltas, axis=1)
            too_close = distances < target_distance
            if not np.any(too_close):
                continue

            close_offsets = np.where(too_close)[0]
            for offset in close_offsets:
                second_index = first_index + 1 + int(offset)
                distance = distances[offset]
                if distance == 0.0:
                    angle = (first_index + second_index + 1) * GOLDEN_ANGLE_RADIANS
                    direction = np.array([math.cos(angle), math.sin(angle)])
                else:
                    direction = deltas[offset] / distance

                push = min((target_distance - distance) * 0.35, max_step)
                movement[first_index] += direction * push
                movement[second_index] -= direction * push
                moved = True

        if not moved:
            break
        step_lengths = np.linalg.norm(movement, axis=1)
        oversized = step_lengths > max_step
        if np.any(oversized):
            movement[oversized] *= (max_step / step_lengths[oversized])[:, None]
        points += movement

    return normalize_positions(
        {
            node: points[index]
            for index, node in enumerate(nodes)
        }
    )


def repel_negative_edges(positions, graph, *, max_passes=40):
    negative_edges = [
        (u, v, abs(float(edge_data["weight"])))
        for u, v, edge_data in graph.edges(data=True)
        if float(edge_data["weight"]) < 0.0
    ]
    if not negative_edges:
        return positions

    nodes = list(positions)
    node_index = {
        node: index
        for index, node in enumerate(nodes)
    }
    points = np.asarray([positions[node] for node in nodes], dtype=float)
    band_by_edge = rank_banded_layout_weights(graph)
    target_distance = NEGATIVE_EDGE_DISTANCE_SCALE / math.sqrt(max(len(nodes), 1))
    max_step = target_distance * 0.18

    for _ in range(int(max_passes)):
        movement = np.zeros_like(points)
        for u, v, _ in negative_edges:
            first_index = node_index[u]
            second_index = node_index[v]
            delta = points[first_index] - points[second_index]
            distance = np.linalg.norm(delta)
            if distance == 0.0:
                angle = (first_index + second_index + 1) * GOLDEN_ANGLE_RADIANS
                direction = np.array([math.cos(angle), math.sin(angle)])
            else:
                direction = delta / distance

            desired_distance = target_distance * band_by_edge[(u, v)]
            if distance >= desired_distance:
                continue

            push = min((desired_distance - distance) * 0.16, max_step)
            movement[first_index] += direction * push
            movement[second_index] -= direction * push

        step_lengths = np.linalg.norm(movement, axis=1)
        if not np.any(step_lengths > 0.0):
            break
        oversized = step_lengths > max_step
        if np.any(oversized):
            movement[oversized] *= (max_step / step_lengths[oversized])[:, None]
        points += movement

    return normalize_positions(
        {
            node: points[index]
            for index, node in enumerate(nodes)
        }
    )


def compute_graph_layout(
    graph,
    *,
    seed=0,
    backend="spring",
    negative_mode="render_only",
):
    if negative_mode not in {"render_only", "repel"}:
        raise NotImplementedError(f"Unsupported negative mode: {negative_mode}")
    if graph.number_of_nodes() == 0:
        return {}

    node_count = graph.number_of_nodes()
    layout_k = SPRING_LAYOUT_K_SCALE / math.sqrt(max(node_count, 1))

    if backend == "spring":
        layout_graph = banded_layout_graph(graph, include_negative=False)
        positions = nx.spring_layout(
            layout_graph,
            weight="layout_weight",
            seed=seed,
            k=layout_k,
            iterations=300,
            scale=2.0,
        )
    elif backend == "spectral":
        layout_graph = banded_layout_graph(graph, include_negative=False)
        if layout_graph.number_of_edges() == 0:
            positions = nx.spring_layout(
                layout_graph,
                weight=None,
                seed=seed,
                k=layout_k,
                iterations=300,
                scale=2.0,
            )
        else:
            positions = nx.spectral_layout(
                layout_graph,
                weight="layout_weight",
                scale=2.0,
            )
    elif backend == "shell":
        layout_graph = banded_layout_graph(graph, include_negative=False)
        positions = nx.shell_layout(layout_graph, scale=2.0)
    else:
        raise NotImplementedError(f"Unsupported layout backend: {backend}")

    positions = normalize_positions(positions)
    if negative_mode == "repel":
        positions = repel_negative_edges(positions, graph)
    return relax_node_overlap(positions, node_count=node_count)


def graph_plot_style(node_count):
    count = max(int(node_count), 1)
    density_scale = math.sqrt(count)
    return {
        "node_size": max(
            MIN_NODE_SIZE,
            min(MAX_NODE_SIZE, NODE_SIZE_DENSITY_SCALE / density_scale),
        ),
        "font_size": max(
            MIN_LABEL_FONT_SIZE,
            min(MAX_LABEL_FONT_SIZE, LABEL_FONT_DENSITY_SCALE / density_scale),
        ),
        "node_linewidth": max(
            MIN_NODE_LINEWIDTH,
            min(MAX_NODE_LINEWIDTH, NODE_LINEWIDTH_DENSITY_SCALE / density_scale),
        ),
        "label_alpha": max(
            MIN_LABEL_ALPHA,
            min(MAX_LABEL_ALPHA, LABEL_ALPHA_DENSITY_SCALE / density_scale),
        ),
        "edge_width": max(
            MIN_EDGE_WIDTH,
            min(MAX_EDGE_WIDTH, EDGE_WIDTH_DENSITY_SCALE / density_scale),
        ),
    }


def draw_signed_strength_edges(axis, graph, positions, *, base_width=0.8):
    alpha_by_edge = edge_alpha_values(graph)
    edge_groups = {
        1: {"edges": [], "colors": [], "style": "solid"},
        -1: {"edges": [], "colors": [], "style": "dashed"},
    }
    color_by_sign = {
        1: "#4f7cac",
        -1: "#c85a54",
    }

    for u, v, edge_data in graph.edges(data=True):
        weight = float(edge_data["weight"])
        sign = 1 if weight > 0.0 else -1 if weight < 0.0 else 0
        if sign == 0:
            continue
        edge = (u, v)
        edge_groups[sign]["edges"].append(edge)
        edge_groups[sign]["colors"].append(
            to_rgba(color_by_sign[sign], alpha_by_edge[edge])
        )

    for edge_group in edge_groups.values():
        if not edge_group["edges"]:
            continue
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edge_group["edges"],
            edge_color=edge_group["colors"],
            width=base_width,
            style=edge_group["style"],
            ax=axis,
        )


def save_graph_plot(
    graph,
    communities_df,
    output_path,
    *,
    title,
    seed=0,
    layout_backend="spring",
    negative_mode="render_only",
):
    # Plot setup
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    community_by_node = communities_df.set_index("node")["community"].to_dict()
    community_ids = sorted(communities_df["community"].unique())
    color_map = plt.colormaps["tab20"]
    colors = {
        community_id: color_map(index % color_map.N)
        for index, community_id in enumerate(community_ids)
    }

    # Layout
    positions = compute_graph_layout(
        graph,
        seed=seed,
        backend=layout_backend,
        negative_mode=negative_mode,
    )
    style = graph_plot_style(graph.number_of_nodes())
    node_colors = [colors[community_by_node[node]] for node in graph.nodes]

    # Drawing
    figure, axis = plt.subplots(figsize=(24, 20))
    draw_signed_strength_edges(axis, graph, positions, base_width=style["edge_width"])
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=style["node_size"],
        edgecolors="black",
        linewidths=style["node_linewidth"],
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: str(node) for node in graph.nodes},
        font_size=style["font_size"],
        alpha=style["label_alpha"],
        ax=axis,
    )
    axis.legend(
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
    axis.set_title(title, fontsize=22, pad=18)
    axis.axis("off")

    # Save
    figure.tight_layout()
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)


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


def load_activation(path):
    import torch

    activation = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(activation, torch.Tensor):
        raise TypeError("Experimental activation data must be a tensor.")
    if activation.ndim != 2:
        raise ValueError(
            "Experimental activation data must have shape [examples, neurons], "
            f"got {tuple(activation.shape)}."
        )
    return activation.detach().cpu().to(torch.float32).numpy()


def activation_correlation_matrix(activations):
    values = np.asarray(activations, dtype=float)
    if values.shape[1] == 0:
        return np.empty((0, 0))
    if values.shape[0] <= 1:
        return np.eye(values.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.atleast_2d(np.corrcoef(values, rowvar=False))
    constant_columns = values.std(axis=0) == 0
    correlation[constant_columns, :] = 0.0
    correlation[:, constant_columns] = 0.0
    np.fill_diagonal(correlation, 1.0)
    return correlation


def activation_cosine_similarity_matrix(activations):
    values = np.asarray(activations, dtype=float)
    dot_products = values.T @ values
    norms = np.linalg.norm(values, axis=0)
    norm_products = norms[:, None] * norms[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.divide(
            dot_products,
            norm_products,
            out=np.zeros_like(dot_products, dtype=float),
            where=norm_products != 0,
        )
    if cosine.size:
        np.fill_diagonal(cosine, 1.0)
    return cosine


def _sparsify(adjacency, pipeline, top_fraction, min_neighbors):
    if pipeline == "louvain":
        return local_top_neighbors_union(
            relu(zero_diag(adjacency)),
            top_fraction,
            min_neighbors=min_neighbors,
        )
    if pipeline == "leiden":
        return local_top_neighbors_union(
            zero_diag(adjacency),
            top_fraction,
            min_neighbors=min_neighbors,
        )
    if pipeline == "louvain_threshold":
        return local_top_percent_union(
            relu(zero_diag(adjacency)),
            top_fraction,
        )
    if pipeline == "leiden_threshold":
        return local_top_percent_union(
            zero_diag(adjacency),
            top_fraction,
        )
    raise ValueError(f"Unknown graph pipeline: {pipeline}.")


def graph_pipeline(
    adjacency,
    searched_formulas,
    output_directory,
    name,
    *,
    pipeline=DEFAULT_PIPELINE,
    top_fraction=DEFAULT_LOCAL_TOP_PERCENT,
    min_neighbors=DEFAULT_MIN_NEIGHBORS,
    min_community_size=DEFAULT_MIN_COMMUNITY_SIZE,
    plot_config=None,
):
    graph = build_graph(
        adjacency,
        lambda matrix: _sparsify(
            matrix,
            pipeline,
            top_fraction,
            min_neighbors,
        ),
        searched_formulas,
    )
    communities = analyze_communities(
        graph,
        pipeline,
        min_community_size=min_community_size,
    )
    degrees = analyze_degrees(graph)
    hubs = analyze_hubs(degrees, communities)
    community_atomics = analyze_community_atomic_frequencies(graph, communities)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    full_report_path = output_directory / f"{name}_cluster_report_full.md"
    summary_report_path = output_directory / f"{name}_cluster_report_summary.md"
    plot_path = output_directory / f"{name}_graph.png"
    save_graph_plot(
        graph,
        communities,
        plot_path,
        title=f"{name} graph",
        **(DEFAULT_PLOT_CONFIG if plot_config is None else plot_config),
    )
    full_report_path.write_text(
        render_full_report(communities, searched_formulas, degrees),
        encoding="utf-8",
    )
    summary_report_path.write_text(
        render_summary_report(
            communities,
            searched_formulas,
            degrees,
            community_atomics,
            hubs,
        ),
        encoding="utf-8",
    )
    return {
        "plot": plot_path,
        "full_report": full_report_path,
        "summary_report": summary_report_path,
    }


def run(
    task_name,
    base_path,
    finetuned_path,
    *,
    formula_path=None,
    output_directory=None,
    pipeline=DEFAULT_PIPELINE,
    top_fraction=DEFAULT_LOCAL_TOP_PERCENT,
    min_neighbors=DEFAULT_MIN_NEIGHBORS,
    min_community_size=DEFAULT_MIN_COMMUNITY_SIZE,
    plot_config=None,
):
    formula_path = (
        Path(formula_path)
        if formula_path is not None
        else compositional_result_path(task_name)
    )
    output_directory = (
        Path(output_directory)
        if output_directory is not None
        else DATA_DIRECTORY
    )
    formulas = pd.read_csv(formula_path)
    outputs = {}
    for variant, activation_path in (
        ("base", base_path),
        ("finetuned", finetuned_path),
    ):
        activations = load_activation(activation_path)
        metrics = {
            "pearson": activation_correlation_matrix(activations),
            "cosine": activation_cosine_similarity_matrix(activations),
        }
        for metric_name, adjacency in metrics.items():
            key = f"{variant}_{metric_name}"
            outputs[key] = graph_pipeline(
                adjacency,
                formulas,
                output_directory,
                f"{task_name}_{variant}_{metric_name}_{pipeline}",
                pipeline=pipeline,
                top_fraction=top_fraction,
                min_neighbors=min_neighbors,
                min_community_size=min_community_size,
                plot_config=plot_config,
            )
    return outputs


def render_summary_report(
    communities_df,
    formula_dataframe,
    degrees_df,
    community_atomic_df,
    hubs_df,
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

    return "\n".join(lines).rstrip() + "\n"
