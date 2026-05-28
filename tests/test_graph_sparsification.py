import importlib.util
from pathlib import Path

import numpy as np


def load_graph_module():
    module_path = Path(__file__).parents[1] / "analysis" / "graph.py"
    spec = importlib.util.spec_from_file_location("analysis_graph", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_top_neighbors_union_matches_old_rank_based_selection():
    graph = load_graph_module()
    adjacency = np.array(
        [
            [1.0, 0.9, -0.8, 0.1],
            [0.9, 1.0, -0.7, 0.0],
            [-0.8, -0.7, 1.0, 0.2],
            [0.1, 0.0, 0.2, 1.0],
        ]
    )

    sparse = graph.local_top_neighbors_union(
        graph.zero_diag(adjacency),
        percent=0.25,
        min_neighbors=2,
    )

    expected = np.array(
        [
            [0.0, 0.9, -0.8, 0.1],
            [0.9, 0.0, -0.7, 0.0],
            [-0.8, -0.7, 0.0, 0.2],
            [0.1, 0.0, 0.2, 0.0],
        ]
    )
    np.testing.assert_array_equal(sparse, expected)


def test_local_top_neighbors_union_uses_rank_not_quantile_threshold():
    graph = load_graph_module()
    adjacency = np.array(
        [
            [0.0, 100.0, 99.0, 1.0],
            [100.0, 0.0, 0.0, 50.0],
            [99.0, 0.0, 0.0, 60.0],
            [1.0, 50.0, 60.0, 0.0],
        ]
    )

    sparse = graph.local_top_neighbors_union(
        adjacency,
        percent=0.34,
        min_neighbors=1,
    )

    assert sparse[0, 1] == 100.0
    assert sparse[0, 2] == 99.0
    assert sparse[0, 3] == 0.0


def test_local_top_neighbors_union_tie_breaks_by_neighbor_id():
    graph = load_graph_module()
    adjacency = np.array(
        [
            [0.0, 0.5, -0.5, 0.4],
            [0.5, 0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0, 0.6],
            [0.4, 0.0, 0.6, 0.0],
        ]
    )

    sparse = graph.local_top_neighbors_union(
        adjacency,
        percent=0.01,
        min_neighbors=1,
    )

    assert sparse[0, 1] == 0.5
    assert sparse[0, 2] == 0.0


def test_local_top_neighbors_union_respects_relu_preprocessing():
    graph = load_graph_module()
    adjacency = np.array(
        [
            [1.0, -0.9, 0.3],
            [-0.9, 1.0, 0.2],
            [0.3, 0.2, 1.0],
        ]
    )

    sparse = graph.local_top_neighbors_union(
        graph.relu(graph.zero_diag(adjacency)),
        percent=0.25,
        min_neighbors=1,
    )

    assert sparse[0, 1] == 0.0
    assert sparse[0, 2] == 0.3
    assert sparse[1, 2] == 0.2
