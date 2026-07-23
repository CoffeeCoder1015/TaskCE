from dataclasses import replace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import theoretical.compositional_explanations.analysis as analysis_module


class FakeDataset:
    column_names = ["premise", "hypothesis", "label"]

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def map(self, *_args, **_kwargs):
        return self


def test_load_activation_accepts_only_raw_two_dimensional_tensors(tmp_path):
    activation_path = tmp_path / "activation.pt"
    torch.save(torch.ones((3, 4), dtype=torch.float16), activation_path)

    loaded = analysis_module.load_activation(activation_path)

    assert loaded.shape == (3, 4)
    assert loaded.dtype == torch.float32

    torch.save({"states": torch.ones((3, 4))}, activation_path)
    with pytest.raises(TypeError, match="must be a tensor"):
        analysis_module.load_activation(activation_path)


def test_analysis_rejects_dataset_and_activation_row_mismatch(tmp_path, monkeypatch):
    activation_path = tmp_path / "activation.pt"
    torch.save(torch.ones((3, 4)), activation_path)
    monkeypatch.setattr(analysis_module, "load_dataset", lambda *_args, **_kwargs: FakeDataset(2))

    with pytest.raises(ValueError, match="dataset rows do not match activation rows"):
        analysis_module.run(
            "snli",
            activation_path,
            output_path=tmp_path / "snli_beam_results.csv",
        )


def test_analysis_writes_to_its_owned_data_path_by_default(tmp_path, monkeypatch):
    activation_path = tmp_path / "activation.pt"
    torch.save(torch.ones((2, 3)), activation_path)
    monkeypatch.setattr(analysis_module, "load_dataset", lambda *_args, **_kwargs: FakeDataset(2))
    monkeypatch.setattr(analysis_module, "get_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(analysis_module, "ConstructFeatures", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(analysis_module, "search_all", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        analysis_module,
        "load_classification_weights",
        lambda *_args, **_kwargs: torch.arange(9, dtype=torch.float32).reshape(3, 3),
    )
    monkeypatch.setitem(
        analysis_module.TASKS,
        "snli",
        replace(analysis_module.TASKS["snli"], min_acts=3),
    )
    monkeypatch.setattr(analysis_module, "DATA_DIRECTORY", tmp_path)

    output_path = analysis_module.run(
        "snli",
        activation_path,
        num_workers=1,
    )

    assert output_path == tmp_path / "snli_beam_results.csv"
    dataframe = pd.read_csv(output_path)
    assert dataframe["neuron"].tolist() == [0, 1, 2]
    assert set(dataframe["formula"]) == {"LOW_ACTS_PRUNED"}
    assert list(dataframe.columns) == [
        "neuron",
        "formula",
        "iou",
        "weight_entailment",
        "weight_neutral",
        "weight_contradiction",
    ]
