import importlib
from pathlib import Path
import sys

import pytest
from datasets import Dataset

PreTrainedTokenizerFast = pytest.importorskip(
    "transformers"
).PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit


class FakeSpacyToken:
    text = "hello"
    tag_ = "NN"
    idx = 0

    def __len__(self):
        return len(self.text)


class FakeSpacyModel:
    def __call__(self, _text):
        return [FakeSpacyToken()]


def load_orchestration_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))
    package = (
        "theoretical.compositional_explanations"
        ".construction.tokenizer"
    )
    sys.modules.pop(package, None)
    sys.modules.pop(f"{package}.orchestration", None)
    return importlib.import_module(f"{package}.orchestration")


def test_get_tokenizer_loads_existing_tokenizer(tmp_path, monkeypatch):
    module = load_orchestration_module(monkeypatch)
    attached = []
    monkeypatch.setattr(
        module,
        "attach_spacy_pretokenizer",
        lambda tokenizer, **_kwargs: attached.append(tokenizer) or tokenizer,
    )

    tokenizer_dir = tmp_path / "demo"
    tokenizer_dir.mkdir()
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(tokenizer_dir)

    monkeypatch.setattr(module, "TOKENIZER_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        module,
        "build_tokenizer",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("should load existing tokenizer")
        ),
    )

    loaded = module.get_tokenizer(
        "demo",
        Dataset.from_dict({"text": ["ignored"]}),
        ("text",),
    )

    assert loaded("hello", add_special_tokens=False)["input_ids"] == [1]
    assert len(attached) == 1


def test_get_tokenizer_trains_and_saves_when_missing(tmp_path, monkeypatch):
    module = load_orchestration_module(monkeypatch)
    attached = []
    detached = []
    monkeypatch.setattr(
        module,
        "attach_spacy_pretokenizer",
        lambda tokenizer, **_kwargs: attached.append(tokenizer) or tokenizer,
    )
    monkeypatch.setattr(
        module,
        "detach_spacy_pretokenizer",
        lambda tokenizer: detached.append(tokenizer) or tokenizer,
    )
    monkeypatch.setattr(module, "TOKENIZER_DIRECTORY", tmp_path)

    def build_tokenizer(**_kwargs):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        return tokenizer

    monkeypatch.setattr(module, "build_tokenizer", build_tokenizer)
    load_calls = []
    original_loader = module.AutoTokenizer.from_pretrained

    def load_from_pretrained(path):
        load_calls.append(Path(path))
        return original_loader(path)

    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        load_from_pretrained,
    )

    tokenizer = module.get_tokenizer(
        "demo",
        Dataset.from_dict({"text": ["hello world", "hello again"]}),
        ("text",),
    )

    assert (tmp_path / "demo" / "tokenizer.json").exists()
    assert load_calls == [tmp_path / "demo"]
    assert tokenizer("hello", add_special_tokens=False)["input_ids"]
    assert tokenizer.additional_special_tokens == []
    assert len(attached) == 1
    assert len(detached) == 1


def test_get_tokenizer_serializes_pos_tokenizer_without_custom_pretokenizer(
    tmp_path,
    monkeypatch,
):
    module = load_orchestration_module(monkeypatch)
    monkeypatch.setattr(module, "TOKENIZER_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        module.build_tokenizer.__globals__["spacy"],
        "load",
        lambda _name, disable: FakeSpacyModel(),
    )

    tokenizer = module.get_tokenizer(
        "demo",
        Dataset.from_dict({"text": ["hello"]}),
        ("text",),
        enable_pos=True,
    )

    assert (tmp_path / "demo_pos" / "tokenizer.json").exists()
    assert tokenizer("hello", add_special_tokens=False)["input_ids"]
    assert set(module.SPACY_POS_TAG_TOKENS) <= set(
        tokenizer.additional_special_tokens
    )
    assert max(tokenizer.additional_special_tokens_ids) < len(tokenizer)
