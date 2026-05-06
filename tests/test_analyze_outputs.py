import pandas as pd
import torch

from analyze import (
    CLASS_TOKEN_IDS,
    build_beam_results_dataframe,
    get_classification_weights,
    log_class_token_decodes,
)


class FakeTokenizer:
    def decode(self, token_ids):
        return f"tok-{token_ids[0]}"


class FakeModel:
    def __init__(self):
        max_token_id = max(CLASS_TOKEN_IDS.values())
        self.lm_head = type(
            "LMHead",
            (),
            {"weight": torch.arange((max_token_id + 1) * 4).reshape(max_token_id + 1, 4)},
        )()


class FakeBeamResult:
    def __init__(self, activation_index, formula, iou):
        self.activation_index = activation_index
        self.best = (formula, iou)


def test_lm_head_weights_are_detached_and_indexed_by_class_token_ids():
    weights = get_classification_weights(FakeModel())

    assert weights.shape == (4, 3)
    assert weights[:, 0].tolist() == FakeModel().lm_head.weight[CLASS_TOKEN_IDS["entailment"]].tolist()
    assert weights[:, 1].tolist() == FakeModel().lm_head.weight[CLASS_TOKEN_IDS["neutral"]].tolist()
    assert weights[:, 2].tolist() == FakeModel().lm_head.weight[CLASS_TOKEN_IDS["contradiction"]].tolist()


def test_log_class_token_decodes_returns_label_token_mapping(capsys):
    decoded = log_class_token_decodes(FakeTokenizer())

    assert decoded == {
        "entailment": "tok-806",
        "neutral": "tok-25919",
        "contradiction": "tok-10913",
    }
    out = capsys.readouterr().out
    assert "entailment token_id=806 decoded='tok-806'" in out


def test_beam_results_dataframe_uses_original_neuron_ids_for_weights():
    weights = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [1.1, 1.2, 1.3],
            [2.1, 2.2, 2.3],
        ]
    )
    results = [
        FakeBeamResult(activation_index=1, formula="tok:A", iou=0.75),
        FakeBeamResult(activation_index=0, formula="tok:B", iou=0.25),
    ]

    df = build_beam_results_dataframe(results, torch.tensor([2, 1]), weights)

    assert list(df.columns) == [
        "neuron",
        "iou",
        "formula",
        "weight_ent",
        "weight_neut",
        "weight_contr",
    ]
    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame(
            [
                {
                    "neuron": 1,
                    "iou": 0.75,
                    "formula": "tok:A",
                    "weight_ent": 1.1,
                    "weight_neut": 1.2,
                    "weight_contr": 1.3,
                },
                {
                    "neuron": 2,
                    "iou": 0.25,
                    "formula": "tok:B",
                    "weight_ent": 2.1,
                    "weight_neut": 2.2,
                    "weight_contr": 2.3,
                },
            ]
        ),
        check_exact=False,
    )
