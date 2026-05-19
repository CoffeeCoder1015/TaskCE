import torch

from analyze import (
    ANALYSIS_TARGETS,
    get_classification_weights,
)


class FakeModel:
    def __init__(self):
        max_token_id = max(ANALYSIS_TARGETS["snli"].class_token_ids.values())
        self.lm_head = type(
            "LMHead",
            (),
            {"weight": torch.arange((max_token_id + 1) * 4).reshape(max_token_id + 1, 4)},
        )()


def test_lm_head_weights_are_detached_and_indexed_by_class_token_ids():
    token_ids = ANALYSIS_TARGETS["snli"].class_token_ids
    weights = get_classification_weights(FakeModel(), token_ids)

    assert weights.shape == (4, 3)
    assert weights[:, 0].tolist() == FakeModel().lm_head.weight[token_ids["entailment"]].tolist()
    assert weights[:, 1].tolist() == FakeModel().lm_head.weight[token_ids["neutral"]].tolist()
    assert weights[:, 2].tolist() == FakeModel().lm_head.weight[token_ids["contradiction"]].tolist()


def test_vitaminc_analysis_target_uses_expected_labels_and_token_ids():
    target = ANALYSIS_TARGETS["vitaminc"]

    assert target.dataset_name == "tals/vitaminc"
    assert target.split == "validation[:10_000]"
    assert target.capture_name == "claim"
    assert target.labels == ("supports", "refutes", "not enough info")
    assert target.class_token_ids == {
        "supports": 56744,
        "refutes": 1891,
        "not enough info": 2897,
    }
    assert target.weight_column_names == (
        "weight_supports",
        "weight_refutes",
        "weight_not_enough_info",
    )


def test_vitaminc_formatter_builds_prompt_and_answer_from_label_text():
    example = {
        "evidence": "The evidence text.",
        "claim": "The claim text.",
        "label": "SUPPORTS",
    }

    formatted = ANALYSIS_TARGETS["vitaminc"].data_formatter(example)

    assert formatted["prompt"][1]["content"] == "Evidence: The evidence text.\nClaim: The claim text."
    assert formatted["answer"] == "supports"
