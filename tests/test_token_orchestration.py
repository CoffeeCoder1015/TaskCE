from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedTokenizerFast
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
    def __call__(self, text):
        return [FakeSpacyToken()]


def load_orchestration_module(monkeypatch):
    import importlib
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))
    sys.modules.pop("WordTokenizer", None)
    sys.modules.pop("WordTokenizer.orchestration", None)
    return importlib.import_module("WordTokenizer.orchestration")


def test_get_tokenizer_loads_existing_tokenizer(tmp_path, monkeypatch):
    module = load_orchestration_module(monkeypatch)
    attached = []
    monkeypatch.setattr(
        module,
        "attach_spacy_pretokenizer",
        lambda tokenizer: attached.append(tokenizer) or tokenizer,
    )

    tokenizer_dir = tmp_path / "demo"
    tokenizer_dir.mkdir()
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=module.SPACY_POS_TAG_TOKENS,
    ).save_pretrained(tokenizer_dir)

    monkeypatch.setattr(module, "TOKENIZER_DIR", tmp_path)
    monkeypatch.setattr(
        module,
        "build_tokenizer",
        lambda: (_ for _ in ()).throw(AssertionError("should load existing tokenizer")),
    )

    loaded = module.get_tokenizer("demo", Dataset.from_dict({"text": ["ignored"]}))

    assert loaded("hello", add_special_tokens=False)["input_ids"] == [1]
    assert set(module.SPACY_POS_TAG_TOKENS) <= set(loaded.additional_special_tokens)
    assert max(loaded.additional_special_tokens_ids) < len(loaded)
    assert len(attached) == 1


def test_get_tokenizer_trains_and_saves_when_missing(tmp_path, monkeypatch):
    module = load_orchestration_module(monkeypatch)
    attached = []
    detached = []
    monkeypatch.setattr(
        module,
        "attach_spacy_pretokenizer",
        lambda tokenizer: attached.append(tokenizer) or tokenizer,
    )
    monkeypatch.setattr(
        module,
        "detach_spacy_pretokenizer",
        lambda tokenizer: detached.append(tokenizer) or tokenizer,
    )

    monkeypatch.setattr(module, "TOKENIZER_DIR", tmp_path)
    def build_tokenizer():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        return tokenizer

    monkeypatch.setattr(module, "build_tokenizer", build_tokenizer)
    load_calls = []
    original_loader = module.AutoTokenizer.from_pretrained

    def load_from_pretrained(path):
        load_calls.append(Path(path))
        return original_loader(path)

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", load_from_pretrained)

    tokenizer = module.get_tokenizer(
        "demo",
        Dataset.from_dict({"text": ["hello world", "hello again"]}),
    )

    assert (tmp_path / "demo" / "tokenizer.json").exists()
    assert load_calls == [tmp_path / "demo"]
    assert tokenizer("hello", add_special_tokens=False)["input_ids"]
    assert set(module.SPACY_POS_TAG_TOKENS) <= set(tokenizer.additional_special_tokens)
    assert max(tokenizer.additional_special_tokens_ids) < len(tokenizer)
    assert len(attached) == 1
    assert len(detached) == 1


def test_get_tokenizer_detaches_spacy_pretokenizer_before_fast_wrapper(
    tmp_path,
    monkeypatch,
):
    module = load_orchestration_module(monkeypatch)
    monkeypatch.setattr(module, "TOKENIZER_DIR", tmp_path)
    monkeypatch.setattr(module.build_tokenizer.__globals__["spacy"], "load", lambda _: FakeSpacyModel())

    tokenizer = module.get_tokenizer(
        "demo",
        Dataset.from_dict({"text": ["hello"]}),
    )

    assert tokenizer("hello", add_special_tokens=False)["input_ids"]
    assert set(module.SPACY_POS_TAG_TOKENS) <= set(tokenizer.additional_special_tokens)
    assert max(tokenizer.additional_special_tokens_ids) < len(tokenizer)
