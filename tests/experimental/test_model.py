import pytest

torch = pytest.importorskip("torch")

import experimental.model as model_module
from experimental.model import CaptureIdentity, WrappedModel


class ProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = torch.nn.Identity()

    def forward(self, values):
        return self.probe(values)


class TwoProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Identity()
        self.second = torch.nn.Identity()

    def forward(self, values):
        return self.second(self.first(values))


class PeftShapedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = torch.nn.Module()
        self.base_model.model.model = torch.nn.Module()
        self.base_model.model.model.layers = torch.nn.ModuleList(
            [torch.nn.Module()]
        )
        self.base_model.model.model.layers[0].feed_forward = torch.nn.Identity()

    def forward(self, values):
        return self.base_model.model.model.layers[0].feed_forward(values)


class ArchitectureBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.Identity()
        self.ffn = torch.nn.Identity()

    def forward(self, values):
        return self.ffn(self.attention(values))


class ArchitectureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embedding = torch.nn.Identity()
        self.model.layers = torch.nn.ModuleList(
            [ArchitectureBlock(), ArchitectureBlock()]
        )

    def forward(self, values):
        values = self.model.embedding(values)
        for layer in self.model.layers:
            values = layer(values)
        return values


def append_output(current, _module, _inputs, output):
    if current is None:
        current = []
    current.append(output.detach().cpu().clone())
    return current


def test_capture_identity_rejects_unsafe_path_components():
    with pytest.raises(ValueError, match="cannot be empty"):
        CaptureIdentity(prefix=(), dataset="snli")
    with pytest.raises(ValueError, match="one relative path component"):
        CaptureIdentity(prefix=("creator/model",), dataset="snli")
    with pytest.raises(ValueError, match="one relative path component"):
        CaptureIdentity(prefix=("creator", "model"), dataset="task/split")


def test_save_writes_capture_without_interpreting_its_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "DATA_DIRECTORY", tmp_path)
    wrapped = WrappedModel(ProbeModel())

    def keep_latest(_current, _module, _inputs, output):
        return output.detach().cpu().clone()

    wrapped.hook("probe", keep_latest)
    wrapped.model(torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]))

    identity = CaptureIdentity(
        prefix=("creator", "model", "base"),
        dataset="snli",
    )
    saved_paths = wrapped.save(identity)

    expected_path = (
        tmp_path
        / "creator"
        / "model"
        / "base"
        / "probe"
        / "snli_activation.pt"
    )
    assert saved_paths == {"probe": expected_path}
    assert torch.equal(
        torch.load(expected_path, weights_only=False),
        torch.tensor(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
            ]
        ),
    )


def test_save_preserves_complete_suffix_resolved_layer(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "DATA_DIRECTORY", tmp_path)
    wrapped = WrappedModel(PeftShapedModel())

    def keep_latest(_current, _module, _inputs, output):
        return output.detach().cpu().clone()

    resolved = wrapped.hook("model.layers.0.feed_forward", keep_latest)
    wrapped.model(torch.tensor([[[1.0], [2.0]]]))

    identity = CaptureIdentity(
        prefix=("creator", "model", "task", "checkpoint-10"),
        dataset="snli",
    )
    saved_paths = wrapped.save(identity)

    assert resolved == "base_model.model.model.layers.0.feed_forward"
    assert saved_paths[resolved] == (
        tmp_path
        / "creator"
        / "model"
        / "task"
        / "checkpoint-10"
        / resolved
        / "snli_activation.pt"
    )


def test_save_skips_unexecuted_hooks_without_creating_data(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(model_module, "DATA_DIRECTORY", tmp_path)
    wrapped = WrappedModel(ProbeModel())
    wrapped.hook("probe", append_output)

    identity = CaptureIdentity(prefix=("creator", "model", "base"), dataset="snli")
    assert wrapped.save(identity) == {}

    assert list(tmp_path.iterdir()) == []


def test_save_preserves_nested_capture_structure(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "DATA_DIRECTORY", tmp_path)
    wrapped = WrappedModel(ProbeModel())
    wrapped.captures["probe"] = {
        "first": [torch.tensor([[1.0]]), torch.tensor([[2.0]])],
        "second": [torch.tensor([[3.0]])],
    }
    identity = CaptureIdentity(prefix=("creator", "model", "base"), dataset="snli")

    saved_path = wrapped.save(identity)["probe"]
    saved = torch.load(saved_path, weights_only=False)

    assert list(saved) == ["first", "second"]
    assert torch.equal(saved["first"][1], torch.tensor([[2.0]]))
    assert torch.equal(saved["second"][0], torch.tensor([[3.0]]))


def test_hook_can_replace_the_layer_capture_on_every_execution():
    wrapped = WrappedModel(ProbeModel())

    def keep_latest(_current, _module, _inputs, output):
        return output.detach().cpu().clone()

    wrapped.hook("probe", keep_latest)
    wrapped.model(torch.tensor([[1.0]]))
    wrapped.model(torch.tensor([[2.0]]))

    assert torch.equal(wrapped.captures["probe"], torch.tensor([[2.0]]))


def test_hook_can_accumulate_every_execution():
    wrapped = WrappedModel(ProbeModel())
    wrapped.hook("probe", append_output)

    wrapped.model(torch.tensor([[1.0]]))
    wrapped.model(torch.tensor([[2.0]]))

    assert len(wrapped.captures["probe"]) == 2
    assert torch.equal(wrapped.captures["probe"][0], torch.tensor([[1.0]]))
    assert torch.equal(wrapped.captures["probe"][1], torch.tensor([[2.0]]))


def test_hooked_layers_keep_isolated_capture_values():
    wrapped = WrappedModel(TwoProbeModel())

    def count_executions(current, _module, _inputs, _output):
        return 1 if current is None else current + 1

    wrapped.hook("first", count_executions)
    wrapped.hook("second", count_executions)
    wrapped.model(torch.tensor([[1.0]]))
    wrapped.model(torch.tensor([[2.0]]))

    assert wrapped.captures == {"first": 2, "second": 2}


def test_print_hook_report_shows_nested_model_and_plain_hook_list(capsys):
    wrapped = WrappedModel(ArchitectureModel())
    wrapped.hook("model.layers.1.ffn", append_output)

    wrapped.print_hook_report()

    assert capsys.readouterr().out == (
        "Model architecture:\n"
        "└── model\n"
        "    ├── embedding\n"
        "    └── layers\n"
        "        ├── 0\n"
        "        │   ├── attention\n"
        "        │   └── ffn\n"
        "        └── 1\n"
        "            ├── attention\n"
        "            └── \033[32mffn\033[0m\n"
        "----------------------------------------\n"
        "Hooking:\n"
        "model.layers.1.ffn\n"
    )


def test_hook_can_group_captures_using_external_pipeline_state():
    wrapped = WrappedModel(ProbeModel())
    context = {"example": None}

    def group_by_example(current, _module, _inputs, output):
        if current is None:
            current = {}
        current.setdefault(context["example"], []).append(
            output.detach().cpu().clone()
        )
        return current

    wrapped.hook("probe", group_by_example)
    context["example"] = "first"
    wrapped.model(torch.tensor([[1.0]]))
    wrapped.model(torch.tensor([[2.0]]))
    context["example"] = "second"
    wrapped.model(torch.tensor([[3.0]]))

    assert len(wrapped.captures["probe"]["first"]) == 2
    assert len(wrapped.captures["probe"]["second"]) == 1


def test_capture_return_value_does_not_replace_module_output():
    wrapped = WrappedModel(ProbeModel())

    def replace_capture(_current, _module, _inputs, output):
        return torch.zeros_like(output)

    wrapped.hook("probe", replace_capture)
    model_output = wrapped.model(torch.tensor([[5.0]]))

    assert torch.equal(model_output, torch.tensor([[5.0]]))
    assert torch.equal(wrapped.captures["probe"], torch.tensor([[0.0]]))


def test_clear_captures_restarts_active_hooks_from_none():
    wrapped = WrappedModel(ProbeModel())
    observed_current_values = []

    def capture_current(current, _module, _inputs, output):
        observed_current_values.append(current)
        return output.detach().cpu().clone()

    wrapped.hook("probe", capture_current)
    wrapped.model(torch.tensor([[1.0]]))
    wrapped.clear_captures()
    wrapped.model(torch.tensor([[2.0]]))

    assert observed_current_values[0] is None
    assert observed_current_values[1] is None
    assert torch.equal(wrapped.captures["probe"], torch.tensor([[2.0]]))
