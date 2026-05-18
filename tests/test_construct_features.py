from datasets import Dataset

from feature.construct import ConstructFeatures


class TinyTokenizer:
    vocab_size = 8
    all_special_ids = []

    def __len__(self):
        return self.vocab_size

    def __call__(self, batch, add_special_tokens=False, return_attention_mask=False):
        del add_special_tokens, return_attention_mask
        vocab = {
            "red": 1,
            "blue": 2,
            "circle": 3,
            "square": 4,
            "ignored": 5,
        }
        return {
            "input_ids": [
                [vocab[token] for token in text.split()]
                for text in batch
            ]
        }

    def decode(self, token_ids):
        tokens = {
            1: "red",
            2: "blue",
            3: "circle",
            4: "square",
            5: "ignored",
        }
        return tokens[token_ids[0]]


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

    assert formulas == {
        "color:red",
        "color:blue",
        "color:circle",
        "color:square",
        "shape:red",
        "shape:blue",
        "shape:circle",
        "shape:square",
    }
    assert all("unused:" not in formula for formula in formulas)
    assert all("ignored" not in formula for formula in formulas)
