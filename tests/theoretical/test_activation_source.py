import ast
import json
from pathlib import Path

import pytest

import theoretical.activation_source as activation_source


MODEL_ID = "creator/model"
LAYER = "model.layers.8.feed_forward"


def model_directory(root):
    return root / "creator" / "model"


def save_placeholder(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"generated activation")
    return path


def test_resolve_latest_checkpoint_uses_numeric_generated_order(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(activation_source, "DATA_DIRECTORY", tmp_path)
    task_directory = model_directory(tmp_path) / "snli"
    (task_directory / "checkpoint-9").mkdir(parents=True)
    expected = task_directory / "checkpoint-10"
    expected.mkdir()
    (task_directory / "named-checkpoint").mkdir()

    assert (
        activation_source.resolve_latest_checkpoint(MODEL_ID, "snli")
        == expected
    )


def test_resolve_base_activation_uses_fixed_generated_data_root(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(activation_source, "DATA_DIRECTORY", tmp_path)
    expected = save_placeholder(
        model_directory(tmp_path)
        / "base"
        / LAYER
        / "snli_activation.pt"
    )

    assert activation_source.resolve_activation_path(
        MODEL_ID,
        "snli",
        LAYER,
    ) == expected


def test_resolve_checkpoint_activation_matches_wrapped_layer_suffix(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(activation_source, "DATA_DIRECTORY", tmp_path)
    checkpoint = model_directory(tmp_path) / "claim_task" / "checkpoint-12"
    expected = save_placeholder(
        checkpoint
        / f"base_model.model.{LAYER}"
        / "vitaminc_activation.pt"
    )

    assert activation_source.resolve_activation_path(
        MODEL_ID,
        "vitaminc",
        LAYER,
        checkpoint=checkpoint,
    ) == expected


def test_resolve_activation_rejects_checkpoint_outside_model_data(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(activation_source, "DATA_DIRECTORY", tmp_path / "data")
    outside = tmp_path / "other" / "snli" / "checkpoint-1"
    outside.mkdir(parents=True)

    with pytest.raises(ValueError, match="inside the generated model directory"):
        activation_source.resolve_activation_path(
            MODEL_ID,
            "snli",
            LAYER,
            checkpoint=outside,
        )


def test_resolve_activation_rejects_ambiguous_wrapped_layers(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(activation_source, "DATA_DIRECTORY", tmp_path)
    checkpoint = model_directory(tmp_path) / "snli" / "checkpoint-3"
    first = checkpoint / f"first.{LAYER}" / "snli_activation.pt"
    second = checkpoint / f"second.{LAYER}" / "snli_activation.pt"
    save_placeholder(first)
    save_placeholder(second)

    with pytest.raises(ValueError, match="matched multiple"):
        activation_source.resolve_activation_path(
            MODEL_ID,
            "snli",
            LAYER,
            checkpoint=checkpoint,
        )


def test_resolver_does_not_create_missing_generated_data(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(activation_source, "DATA_DIRECTORY", tmp_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        activation_source.resolve_activation_path(
            MODEL_ID,
            "snli",
            LAYER,
        )

    assert list(tmp_path.iterdir()) == []


def test_theoretical_code_does_not_import_experimental():
    theoretical_directory = Path(__file__).resolve().parents[2] / "theoretical"
    violations = []

    for source_path in theoretical_directory.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(
                module == "experimental" or module.startswith("experimental.")
                for module in modules
            ):
                violations.append(source_path)

    assert violations == []


def test_raw_activation_notebooks_use_theoretical_resolver():
    theoretical_directory = Path(__file__).resolve().parents[2] / "theoretical"
    analysis_names = (
        "compositional_explanations",
        "activation_diagnostics",
        "activation_regression",
        "activation_transfer",
        "graph_analysis",
    )

    for analysis_name in analysis_names:
        notebook_path = (
            theoretical_directory / analysis_name / "notebook.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "".join(
            line
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            for line in cell["source"]
        )
        assert "resolve_activation_path" in source
        assert "resolve_latest_checkpoint" in source
        assert "checkpoint-1000" not in source
