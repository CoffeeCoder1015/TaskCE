from datasets import Dataset
import numpy as np

from feature.construct import ConstructFeatures
from feature.find import select_feature_token_ids


class TinyTokenizer:
    token_to_id = {
        "red": 1,
        "blue": 2,
        "circle": 3,
        "square": 4,
        "ignored": 5,
        "<NN>": 6,
        "<VB>": 7,
        "[EXTRA]": 8,
    }
    id_to_token = {token_id: token for token, token_id in token_to_id.items()}
    vocab_size = 10
    additional_special_tokens_ids = [6, 7, 8]
    all_special_ids = [6, 7, 8]

    def __len__(self):
        return self.vocab_size

    def __call__(self, batch, add_special_tokens=False, return_attention_mask=False):
        del add_special_tokens, return_attention_mask
        return {
            "input_ids": [
                [self.token_to_id[token] for token in text.split()]
                for text in batch
            ]
        }

    def decode(self, token_ids):
        return self.id_to_token[token_ids[0]]

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id[token] for token in tokens]


def select_feature_text(example):
    return {"color": example["color"], "shape": example["shape"]}


def test_construct_features_selects_data_for_feature_construction():
    dataset = Dataset.from_dict(
        {
            "color": ["red", "blue"],
            "shape": ["circle", "square"],
            "unused": ["ignored", "ignored"],
        }
    )

    feature_vectors = ConstructFeatures(
        dataset,
        TinyTokenizer(),
        feature_text_selector=select_feature_text,
        top_k=5,
    )

    formulas = {str(formula) for formula, _ in feature_vectors}

    assert {
        "color:red",
        "color:blue",
        "color:circle",
        "color:square",
        "color:<NN>",
        "color:<VB>",
        "shape:red",
        "shape:blue",
        "shape:circle",
        "shape:square",
        "shape:<NN>",
        "shape:<VB>",
    } <= formulas
    assert all("unused:" not in formula for formula in formulas)
    assert all("ignored" not in formula for formula in formulas)


def test_feature_token_selection_keeps_pos_tags_out_of_top_k_quota():
    tokenizer = TinyTokenizer()
    counts = np.zeros(len(tokenizer), dtype=np.int64)
    counts[1] = 10
    counts[2] = 9
    counts[tokenizer.token_to_id["<NN>"]] = 100
    counts[tokenizer.token_to_id["<VB>"]] = 1

    token_ids = select_feature_token_ids(counts, tokenizer, top_k=2)

    assert token_ids == [1, 2, 6, 7]


def test_construct_features_includes_pos_tags_after_top_k_tokens():
    dataset = Dataset.from_dict(
        {
            "color": ["red <NN>", "blue <VB>"],
            "shape": ["circle <NN>", "square <VB>"],
        }
    )

    feature_vectors = ConstructFeatures(
        dataset,
        TinyTokenizer(),
        feature_text_selector=select_feature_text,
        top_k=1,
    )

    formulas = {str(formula) for formula, _ in feature_vectors}

    assert {"color:red", "shape:red"} <= formulas
    assert {"color:<NN>", "shape:<NN>", "color:<VB>", "shape:<VB>"} <= formulas
    assert "color:blue" not in formulas
