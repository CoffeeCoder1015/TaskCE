import torch

from analyze import (
    CLASS_TOKEN_IDS,
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
