import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def load_graph_module():
    module_path = (
        Path(__file__).parents[1] / "theoretical" / "graph_analysis" / "analysis.py"
    )
    spec = importlib.util.spec_from_file_location("analysis_graph", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distinct_k_core_layers_collapses_identical_cumulative_node_sets():
    graph_mod = load_graph_module()
    graph = nx.path_graph(5)
    k_core_nodes = pd.DataFrame(
        [
            {"k": 1, "node": 0},
            {"k": 1, "node": 1},
            {"k": 1, "node": 2},
            {"k": 1, "node": 3},
            {"k": 2, "node": 0},
            {"k": 2, "node": 1},
            {"k": 2, "node": 2},
            {"k": 2, "node": 3},
            {"k": 3, "node": 1},
            {"k": 3, "node": 2},
        ]
    )

    layers = graph_mod.distinct_k_core_layers(graph, k_core_nodes)

    assert [layer["k_values"] for layer in layers] == [[1, 2], [3]]
    assert [layer["nodes"] for layer in layers] == [
        frozenset({0, 1, 2, 3}),
        frozenset({1, 2}),
    ]
    assert graph_mod.k_core_layer_label(layers[0]["k_values"]) == "k=1-2"


def test_k_core_component_hulls_draws_one_hull_per_component():
    graph_mod = load_graph_module()
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (3, 4), (4, 5)])
    positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 0.0]),
        2: np.array([0.5, 1.0]),
        3: np.array([5.0, 0.0]),
        4: np.array([6.0, 0.0]),
        5: np.array([5.5, 1.0]),
    }
    k_core_nodes = pd.DataFrame(
        {"k": 1, "node": node}
        for node in range(6)
    )

    hulls = graph_mod.k_core_component_hulls(
        graph,
        positions,
        k_core_nodes,
        padding=0.1,
    )

    assert len(hulls) == 2
    assert {hull["nodes"] for hull in hulls} == {
        frozenset({0, 1, 2}),
        frozenset({3, 4, 5}),
    }
    for hull in hulls:
        assert hull["polygon"].shape[1] == 2


def test_k_core_component_hulls_handles_singleton_and_two_node_layers():
    graph_mod = load_graph_module()
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(1, 2)
    positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([2.0, 0.0]),
        2: np.array([3.0, 0.0]),
    }
    k_core_nodes = pd.DataFrame(
        [
            {"k": 1, "node": 0},
            {"k": 2, "node": 1},
            {"k": 2, "node": 2},
        ]
    )

    hulls = graph_mod.k_core_component_hulls(
        graph,
        positions,
        k_core_nodes,
        padding=0.1,
    )

    assert {hull["nodes"] for hull in hulls} == {
        frozenset({0}),
        frozenset({1, 2}),
    }
    for hull in hulls:
        assert len(hull["polygon"]) >= 12


def test_distinct_k_core_layers_skips_repeated_node_sets_even_when_non_adjacent():
    graph_mod = load_graph_module()
    graph = nx.path_graph(4)
    k_core_nodes = pd.DataFrame(
        [
            {"k": 1, "node": 0},
            {"k": 1, "node": 1},
            {"k": 2, "node": 1},
            {"k": 2, "node": 2},
            {"k": 3, "node": 0},
            {"k": 3, "node": 1},
        ]
    )

    layers = graph_mod.distinct_k_core_layers(graph, k_core_nodes)

    assert [layer["k_values"] for layer in layers] == [[1], [2]]
