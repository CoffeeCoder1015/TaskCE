from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit


def load_orchestration_module(monkeypatch):
    import importlib
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))
    sys.modules.pop("token", None)
    sys.modules.pop("token.orchestration", None)
    return importlib.import_module("token.orchestration")


def test_get_tokenizer_loads_existing_tokenizer(tmp_path, monkeypatch):
    module = load_orchestration_module(monkeypatch)

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


def test_get_tokenizer_trains_and_saves_when_missing(tmp_path, monkeypatch):
    module = load_orchestration_module(monkeypatch)

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
