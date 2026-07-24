from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import theoretical.ablation.analysis as ablation_analysis
import theoretical.activations.neuron_fate.analysis as neuron_fate_analysis
import theoretical.activations.threshold_coverage.analysis as threshold_coverage_analysis
import theoretical.graph_analysis.analysis as graph_analysis


def save_pair(directory):
    base_path = directory / "base.pt"
    finetuned_path = directory / "finetuned.pt"
    torch.save(
        torch.tensor(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
        ),
        base_path,
    )
    torch.save(
        torch.tensor(
            [[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]]
        ),
        finetuned_path,
    )
    return base_path, finetuned_path


def test_neuron_fate_and_threshold_coverage_load_tensor_files(tmp_path):
    base_path, finetuned_path = save_pair(tmp_path)

    fate = neuron_fate_analysis.run(
        base_path,
        finetuned_path,
    )
    coverage = threshold_coverage_analysis.run(
        finetuned_path,
        task_name="demo",
        alpha=0.5,
        min_acts=1,
    )

    assert set(fate) == {"affine", "literal", "relative_to_base"}
    assert set(coverage) == {"selected", "sweep"}


def test_graph_analysis_reads_raw_tensors_and_compositional_csv(tmp_path, monkeypatch):
    base_path, finetuned_path = save_pair(tmp_path)
    compositional_data = tmp_path / "compositional_explanations" / "data"
    compositional_data.mkdir(parents=True)
    formula_path = compositional_data / "snli_beam_results.csv"
    pd.DataFrame(
        [
            {"neuron": 0, "formula": "premise:cat", "iou": 0.5},
            {"neuron": 1, "formula": "hypothesis:animal", "iou": 0.4},
        ]
    ).to_csv(formula_path, index=False)
    calls = []

    def fake_graph_pipeline(adjacency, formulas, output_directory, name, **_kwargs):
        calls.append((adjacency.shape, tuple(formulas["neuron"]), output_directory, name))
        return {"name": name}

    monkeypatch.setattr(graph_analysis, "graph_pipeline", fake_graph_pipeline)
    monkeypatch.setattr(
        graph_analysis,
        "COMPOSITIONAL_DATA_DIRECTORY",
        compositional_data,
    )
    graph_output = tmp_path / "graph_analysis" / "data"
    monkeypatch.setattr(graph_analysis, "DATA_DIRECTORY", graph_output)

    outputs = graph_analysis.run(
        "snli",
        base_path,
        finetuned_path,
    )

    assert set(outputs) == {
        "base_pearson",
        "base_cosine",
        "finetuned_pearson",
        "finetuned_cosine",
    }
    assert all(shape == (2, 2) for shape, *_rest in calls)
    assert all(neurons == (0, 1) for _shape, neurons, *_rest in calls)
    assert all(directory == graph_output for *_prefix, directory, _name in calls)


def test_ablation_analysis_reads_default_compositional_file_path(tmp_path, monkeypatch):
    compositional_data = tmp_path / "compositional_explanations" / "data"
    compositional_data.mkdir(parents=True)
    formula_path = compositional_data / "snli_beam_results.csv"
    formula_path.write_text("neuron,formula,iou,weight_entailment,weight_neutral,weight_contradiction\n")
    ablation_output = tmp_path / "ablation" / "data"
    analysis_result = SimpleNamespace(output_dir=ablation_output)
    observed = {}

    def fake_analysis(path, **kwargs):
        observed["formula_path"] = path
        observed["analysis_output"] = kwargs["output_directory"]
        return analysis_result

    class FakeEngine:
        def __init__(self, **kwargs):
            observed["engine"] = kwargs

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(ablation_analysis, "_select_neurons", fake_analysis)
    monkeypatch.setattr(
        ablation_analysis,
        "COMPOSITIONAL_DATA_DIRECTORY",
        compositional_data,
    )
    monkeypatch.setattr(ablation_analysis, "DATA_DIRECTORY", ablation_output)
    monkeypatch.setattr(ablation_analysis, "_load_dataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        ablation_analysis,
        "latest_task_checkpoint",
        lambda *_args, **_kwargs: tmp_path / "checkpoint-1",
    )
    monkeypatch.setattr(
        ablation_analysis,
        "_build_inference_engine",
        lambda **kwargs: FakeEngine(**kwargs),
    )
    monkeypatch.setattr(
        ablation_analysis,
        "run_ablation",
        lambda result, engine, output_dir: (result, engine, output_dir),
    )

    result, _engine, output_directory = ablation_analysis.run(
        "snli",
    )

    assert result is analysis_result
    assert observed["formula_path"] == formula_path
    assert observed["analysis_output"] == ablation_output
    assert observed["closed"] is True
    assert output_directory == ablation_output
