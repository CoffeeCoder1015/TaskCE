import importlib.util
from pathlib import Path

import networkx as nx


def load_graph_module():
    module_path = Path(__file__).parents[1] / "analysis" / "graph.py"
    spec = importlib.util.spec_from_file_location("analysis_graph", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weighted_graph():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=0.9)
    graph.add_edge("b", "c", weight=0.4)
    graph.add_edge("c", "d", weight=-0.7)
    graph.add_edge("a", "d", weight=0.1)
    return graph


def test_compute_graph_layout_supports_requested_public_backends():
    graph_mod = load_graph_module()
    graph = weighted_graph()

    for backend in ["spring", "spectral", "shell"]:
        positions = graph_mod.compute_graph_layout(graph, seed=0, backend=backend)
        assert set(positions) == set(graph.nodes)
        for point in positions.values():
            assert len(point) == 2


def test_compute_graph_layout_rejects_old_spring_banded_public_name():
    graph_mod = load_graph_module()

    try:
        graph_mod.compute_graph_layout(weighted_graph(), backend="spring_banded")
    except NotImplementedError as error:
        assert "Unsupported layout backend" in str(error)
    else:
        raise AssertionError("spring_banded should not be a public layout backend")


def test_rank_banded_layout_weights_preserve_rank_structure_not_raw_precision():
    graph_mod = load_graph_module()
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=0.10)
    graph.add_edge("b", "c", weight=0.11)
    graph.add_edge("c", "d", weight=0.90)
    graph.add_edge("d", "e", weight=0.91)

    bands = graph_mod.rank_banded_layout_weights(graph, bands=2)

    assert bands[("a", "b")] == 1.0
    assert bands[("b", "c")] == 1.0
    assert bands[("c", "d")] == 2.0
    assert bands[("d", "e")] == 2.0


def test_banded_layout_graph_uses_absolute_magnitude_and_skips_negative_by_default():
    graph_mod = load_graph_module()
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=0.5)
    graph.add_edge("b", "c", weight=-0.8)

    layout_graph = graph_mod.banded_layout_graph(graph, bands=4, include_negative=False)

    assert layout_graph.has_edge("a", "b")
    assert not layout_graph.has_edge("b", "c")
    assert "layout_weight" in layout_graph["a"]["b"]


def test_edge_alpha_values_lerp_strength_into_alpha_range():
    graph_mod = load_graph_module()
    graph = nx.Graph()
    graph.add_edge("weak", "mid", weight=0.25)
    graph.add_edge("mid", "strong", weight=1.0)

    alpha = graph_mod.edge_alpha_values(graph, min_alpha=0.2, max_alpha=0.8)

    assert alpha[("weak", "mid")] == 0.2
    assert alpha[("mid", "strong")] == 0.8
