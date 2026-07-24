import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("transformers")

import theoretical.activations.task_separation.analysis as task_separation
import theoretical.evaluation.analysis as evaluation_analysis


class FakeDataset:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        return self.values


def test_task_separation_reloads_labels_for_raw_activation_rows(
    tmp_path,
    monkeypatch,
):
    activation_path = tmp_path / "snli.pt"
    torch.save(torch.arange(12, dtype=torch.float32).reshape(3, 4), activation_path)
    monkeypatch.setattr(
        task_separation,
        "_load_dataset",
        lambda *_args, **_kwargs: FakeDataset([0, 1]),
    )

    with pytest.raises(ValueError, match="dataset rows do not match activation rows"):
        task_separation.run("snli", activation_path)


def test_evaluation_analysis_persists_measurements_at_selected_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        evaluation_analysis,
        "_load_dataset",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        evaluation_analysis,
        "Evaluate",
        lambda **_kwargs: [
            {
                "task": "snli",
                "variant": "base",
                "success": 8,
                "fail": 2,
                "reject": 0,
            }
        ],
    )

    output_path = evaluation_analysis.run(
        output_path=tmp_path / "evaluation_results.csv"
    )

    assert output_path == tmp_path / "evaluation_results.csv"
    dataframe = pd.read_csv(output_path)
    assert dataframe.to_dict("records") == [
        {
            "task": "snli",
            "variant": "base",
            "success": 8,
            "fail": 2,
            "reject": 0,
        }
    ]
