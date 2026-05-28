from collections import Counter
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Polygon
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sympy import Symbol
from sympy.logic.boolalg import Not as SymNot
from sympy.parsing.sympy_parser import parse_expr


ATOMIC_PATTERN_TEXT = r"[A-Za-z_][\w-]*:(?:<[^>]+>|[^\s()]+)"
ATOMIC_PATTERN = re.compile(ATOMIC_PATTERN_TEXT)
NEGATED_ATOMIC_PATTERN = re.compile(rf"\(\s*NOT\s+({ATOMIC_PATTERN_TEXT})\s*\)")

# Layout
GOLDEN_ANGLE_RADIANS = math.pi * (3.0 - math.sqrt(5.0))
SPRING_LAYOUT_K_SCALE = 2.4
NEGATIVE_EDGE_DISTANCE_SCALE = 0.9

# Node overlap
NODE_OVERLAP_DISTANCE_SCALE = 0.55
NODE_OVERLAP_MIN_DISTANCE = 0.015

# K-core hulls
K_CORE_HULL_PADDING_SCALE = 0.045
K_CORE_HULL_MIN_PADDING = 0.04
K_CORE_HULL_ALPHA = 0.2
K_CORE_HULL_LINEWIDTH = 1.2
K_CORE_NODE_SIZE_SCALE = 0.85

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


def distinct_k_core_layers(graph, k_core_nodes_df):
    graph_nodes = set(graph.nodes)
    layers = []
    seen_node_sets = set()

    for k in sorted(k_core_nodes_df["k"].unique()):
        k_nodes = frozenset(
            node
            for node in k_core_nodes_df[k_core_nodes_df["k"] == k]["node"]
            if node in graph_nodes
        )
        if not k_nodes:
            continue

        if layers and k_nodes == layers[-1]["nodes"]:
            layers[-1]["k_values"].append(k)
            continue

        if k_nodes in seen_node_sets:
            continue

        layers.append({"k_values": [k], "nodes": k_nodes})
        seen_node_sets.add(k_nodes)

    return layers


def k_core_layer_label(k_values):
    values = [int(k) for k in k_values]
    if len(values) == 1:
        return f"k={values[0]}"
    return f"k={values[0]}-{values[-1]}"


def padded_hull_polygon(points, padding):
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 0:
        return None
    if points.shape[0] == 1:
        return circle_polygon(points[0], padding)
    if points.shape[0] == 2:
        return capsule_polygon(points[0], points[1], padding)

    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except QhullError:
        return capsule_for_collinear_points(points, padding)

    centroid = hull_points.mean(axis=0)
    vectors = hull_points - centroid
    distances = np.linalg.norm(vectors, axis=1)
    padded = hull_points.copy()
    nonzero = distances > 0.0
    padded[nonzero] += vectors[nonzero] / distances[nonzero, None] * padding
    return padded


def circle_polygon(center, radius, segments=24):
    angles = np.linspace(0.0, 2.0 * math.pi, int(segments), endpoint=False)
    return np.column_stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        ]
    )


def capsule_polygon(first, second, radius, segments=12):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    axis = second - first
    length = np.linalg.norm(axis)
    if length == 0.0:
        return circle_polygon(first, radius)

    unit = axis / length
    angle = math.atan2(unit[1], unit[0])
    first_angles = np.linspace(
        angle + math.pi / 2.0,
        angle + 3.0 * math.pi / 2.0,
        int(segments),
    )
    second_angles = np.linspace(
        angle - math.pi / 2.0,
        angle + math.pi / 2.0,
        int(segments),
    )
    first_arc = np.column_stack(
        [
            first[0] + radius * np.cos(first_angles),
            first[1] + radius * np.sin(first_angles),
        ]
    )
    second_arc = np.column_stack(
        [
            second[0] + radius * np.cos(second_angles),
            second[1] + radius * np.sin(second_angles),
        ]
    )
    return np.vstack([first_arc, second_arc])


def capsule_for_collinear_points(points, padding):
    points = np.asarray(points, dtype=float)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    first_index, second_index = np.unravel_index(
        np.argmax(distances),
        distances.shape,
    )
    return capsule_polygon(points[first_index], points[second_index], padding)


def k_core_component_hulls(graph, positions, k_core_nodes_df, padding):
    hulls = []
    for layer in distinct_k_core_layers(graph, k_core_nodes_df):
        subgraph = graph.subgraph(layer["nodes"])
        for component in nx.connected_components(subgraph):
            component_points = [
                positions[node]
                for node in component
            ]
            polygon = padded_hull_polygon(component_points, padding)
            if polygon is not None:
                hulls.append(
                    {
                        "k_values": layer["k_values"],
                        "nodes": frozenset(component),
                        "polygon": polygon,
                    }
                )
    return hulls


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


def save_k_core_plot(
    graph,
    k_core_nodes_df,
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
    layers = distinct_k_core_layers(graph, k_core_nodes_df)
    k_values = [layer["k_values"][0] for layer in layers]
    color_map = plt.colormaps["viridis"]
    colors = {
        k: color_map(index / max(len(k_values) - 1, 1))
        for index, k in enumerate(k_values)
    }

    # Layout and hulls
    positions = compute_graph_layout(
        graph,
        seed=seed,
        backend=layout_backend,
        negative_mode=negative_mode,
    )
    style = graph_plot_style(graph.number_of_nodes())
    position_array = np.asarray(list(positions.values()), dtype=float)
    layout_span = np.ptp(position_array, axis=0).max() if len(position_array) else 1.0
    hull_padding = max(
        layout_span * K_CORE_HULL_PADDING_SCALE,
        K_CORE_HULL_MIN_PADDING,
    )

    # Drawing
    figure, axis = plt.subplots(figsize=(24, 20))
    for hull in k_core_component_hulls(
        graph,
        positions,
        k_core_nodes_df,
        hull_padding,
    ):
        k = hull["k_values"][0]
        axis.add_patch(
            Polygon(
                hull["polygon"],
                closed=True,
                facecolor=colors[k],
                edgecolor=colors[k],
                alpha=K_CORE_HULL_ALPHA,
                linewidth=K_CORE_HULL_LINEWIDTH,
                zorder=0,
            )
        )

    draw_signed_strength_edges(axis, graph, positions, base_width=style["edge_width"])
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color="#f2f2f2",
        node_size=style["node_size"] * K_CORE_NODE_SIZE_SCALE,
        edgecolors="#555555",
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
                facecolor=colors[layer["k_values"][0]],
                edgecolor=colors[layer["k_values"][0]],
                alpha=K_CORE_HULL_ALPHA,
                label=k_core_layer_label(layer["k_values"]),
            )
            for layer in layers
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
