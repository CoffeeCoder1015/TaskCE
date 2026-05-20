from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit


def test_get_tokenizer_loads_existing_tokenizer(tmp_path, monkeypatch):
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "token" / "orchestration.py"
    monkeypatch.syspath_prepend(str(module_path.parent))
    spec = importlib.util.spec_from_file_location("taskce_token_orchestration", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tokenizer_dir = tmp_path / "demo"
    tokenizer_dir.mkdir()
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(tokenizer_dir)

    monkeypatch.setattr(module, "TOKENIZER_DIR", tmp_path)

    loaded = module.get_tokenizer("demo", Dataset.from_dict({"text": ["ignored"]}))

    assert loaded("hello", add_special_tokens=False)["input_ids"] == [1]


def test_get_tokenizer_trains_and_saves_when_missing(tmp_path, monkeypatch):
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "token" / "orchestration.py"
    monkeypatch.syspath_prepend(str(module_path.parent))
    spec = importlib.util.spec_from_file_location("taskce_token_orchestration", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
