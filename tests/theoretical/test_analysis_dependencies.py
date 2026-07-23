from types import SimpleNamespace

import numpy as np
import pandas as pd

import theoretical.ablation.analysis as ablation_analysis
import theoretical.graph_analysis.analysis as graph_analysis


def test_graph_uses_compositional_data_and_local_output_by_default(
    tmp_path,
    monkeypatch,
):
    compositional_data = tmp_path / "compositional_explanations" / "data"
    graph_data = tmp_path / "graph_analysis" / "data"
    observed = {}

    def fake_read_csv(path):
        observed["formula_path"] = path
        return pd.DataFrame(
            [{"neuron": 0, "formula": "premise:cat", "iou": 0.5}]
        )

    def fake_pipeline(_adjacency, _formulas, output_directory, name, **_kwargs):
        observed.setdefault("outputs", []).append((output_directory, name))
        return {"name": name}

    monkeypatch.setattr(
        graph_analysis,
        "COMPOSITIONAL_DATA_DIRECTORY",
        compositional_data,
    )
    monkeypatch.setattr(graph_analysis, "DATA_DIRECTORY", graph_data)
    monkeypatch.setattr(graph_analysis.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        graph_analysis,
        "load_activation",
        lambda _path: np.ones((2, 1)),
    )
    monkeypatch.setattr(
        graph_analysis,
        "activation_correlation_matrix",
        lambda _activation: np.eye(1),
    )
    monkeypatch.setattr(
        graph_analysis,
        "activation_cosine_similarity_matrix",
        lambda _activation: np.eye(1),
    )
    monkeypatch.setattr(graph_analysis, "graph_pipeline", fake_pipeline)

    graph_analysis.run("snli", "base.pt", "finetuned.pt")

    assert observed["formula_path"] == (
        compositional_data / "snli_beam_results.csv"
    )
    assert all(
        output_directory == graph_data
        for output_directory, _name in observed["outputs"]
    )


def test_ablation_uses_compositional_data_and_local_output_by_default(
    tmp_path,
    monkeypatch,
):
    compositional_data = tmp_path / "compositional_explanations" / "data"
    ablation_data = tmp_path / "ablation" / "data"
    observed = {}
    analysis_result = SimpleNamespace(output_dir=ablation_data)

    def fake_select(path, **kwargs):
        observed["formula_path"] = path
        observed["output_directory"] = kwargs["output_directory"]
        return analysis_result

    class FakeEngine:
        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(
        ablation_analysis,
        "COMPOSITIONAL_DATA_DIRECTORY",
        compositional_data,
    )
    monkeypatch.setattr(ablation_analysis, "DATA_DIRECTORY", ablation_data)
    monkeypatch.setattr(ablation_analysis, "_select_neurons", fake_select)
    monkeypatch.setattr(
        ablation_analysis,
        "_load_dataset",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        ablation_analysis,
        "latest_task_checkpoint",
        lambda *_args, **_kwargs: tmp_path / "checkpoint-1",
    )
    monkeypatch.setattr(
        ablation_analysis,
        "_build_inference_engine",
        lambda **_kwargs: FakeEngine(),
    )
    monkeypatch.setattr(
        ablation_analysis,
        "run_ablation",
        lambda result, _engine, output_dir: (result, output_dir),
    )

    result, output_directory = ablation_analysis.run("snli")

    assert result is analysis_result
    assert observed["formula_path"] == (
        compositional_data / "snli_beam_results.csv"
    )
    assert observed["output_directory"] == ablation_data
    assert observed["closed"] is True
    assert output_directory == ablation_data
