import torch

from analyze import (
    CLASS_TOKEN_IDS,
    get_classification_weights,
)


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
