import importlib
import sys
import types
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


fake_torch = types.SimpleNamespace(
    bfloat16=object(),
    cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    inference_mode=lambda: FakeInferenceMode(),
)


class FakeTokenized(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, model_id):
        return cls()

    def apply_chat_template(self, prompts, **kwargs):
        self.prompts = prompts
        self.chat_template_kwargs = kwargs
        return FakeTokenized({"input_ids": [[1, 1], [1, 1]]})


class FakeLayer:
    def register_forward_hook(self, hook):
        self.hook = hook
        return types.SimpleNamespace(remove=lambda: setattr(self, "hook", None))


class FakePredictions:
    def tolist(self):
        return [0, 2]


class FakeClassLogits:
    def argmax(self, dim):
        assert dim == 1
        return FakePredictions()


class FakeLogits:
    def __getitem__(self, key):
        batch_slice, token_position, class_ids = key
        assert batch_slice == slice(None)
        assert token_position == -1
        assert class_ids == [11, 22, 33]
        return FakeClassLogits()


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.layer = FakeLayer()
        self.forward_kwargs = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls()

    def eval(self):
        return None

    def named_modules(self):
        yield "", self
        yield "layer", self.layer

    def __call__(self, **kwargs):
        self.forward_kwargs = kwargs
        return types.SimpleNamespace(logits=FakeLogits())


class FakeDataset:
    def map(self, formatter):
        rows = [
            {"prompt": [{"role": "user", "content": "p1"}], "answer": "supports"},
            {"prompt": [{"role": "user", "content": "p2"}], "answer": "refutes"},
        ]
        return {key: [formatter(row)[key] for row in rows] for key in ("prompt", "answer")}


def test_ablation_inference_uses_final_token_class_logits(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=object))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=FakeModel, AutoTokenizer=FakeTokenizer),
    )

    ablation_inference = importlib.import_module("analysis.ablation_inference")
    ablation_inference = importlib.reload(ablation_inference)

    task = types.SimpleNamespace(
        name="fever",
        dataset=FakeDataset(),
        data_formatter=lambda row: row,
    )

    engine = ablation_inference.AblationInferenceEngine(
        model_id="fake",
        task=task,
        layer="layer",
        class_token_ids={
            "supports": 11,
            "refutes": 22,
            "not enough info": 33,
        },
        batch_size=2,
    )

    assert engine([7]) == Counter({"success": 1, "fail": 1})
    assert engine.tokenizer.chat_template_kwargs == {
        "add_generation_prompt": True,
        "padding": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert engine.model.forward_kwargs["device"] == "cpu"
