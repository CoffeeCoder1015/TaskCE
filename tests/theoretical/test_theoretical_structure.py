import ast
import json
from pathlib import Path
import re

import pytest

import theoretical.activations.source as activation_source


LOCAL_RESOLVER_ANALYSES = (
    "compositional_explanations",
    "graph_analysis",
)
ACTIVATION_PURPOSES = (
    "neuron_fate",
    "threshold_coverage",
    "task_separation",
)
MODEL_ID = "creator/model"
LAYER = "model.layers.8.feed_forward"


def load_activation_path(analysis_name, data_directory):
    """Load only the resolver, avoiding each analysis module's ML dependencies."""
    analysis_path = (
        Path(__file__).resolve().parents[2]
        / "theoretical"
        / analysis_name
        / "analysis.py"
    )
    tree = ast.parse(analysis_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "activation_path"
    )
    namespace = {
        "ACTIVATION_DATA_DIRECTORY": data_directory,
        "Path": Path,
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            analysis_path,
            "exec",
        ),
        namespace,
    )
    return namespace["activation_path"]


def save_placeholder(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"activation")
    return path


@pytest.mark.parametrize("analysis_name", LOCAL_RESOLVER_ANALYSES)
def test_direction_activation_paths_follow_experimental_writer_layout(
    analysis_name,
    tmp_path,
):
    activation_path = load_activation_path(analysis_name, tmp_path)
    model_directory = tmp_path / "creator" / "model"
    base_path = save_placeholder(
        model_directory
        / "base"
        / LAYER
        / "snli_activation.pt"
    )
    checkpoint_9_path = save_placeholder(
        model_directory
        / "snli"
        / "checkpoint-9"
        / LAYER
        / "snli_activation.pt"
    )
    checkpoint_10_path = save_placeholder(
        model_directory
        / "snli"
        / "checkpoint-10"
        / f"base_model.model.{LAYER}"
        / "snli_activation.pt"
    )
    (model_directory / "base" / f"base_model.model.{LAYER}").mkdir()

    assert activation_path(MODEL_ID, "snli", LAYER) == base_path
    assert activation_path(
        MODEL_ID,
        "snli",
        LAYER,
        task_name="snli",
    ) == checkpoint_10_path
    assert activation_path(
        MODEL_ID,
        "snli",
        LAYER,
        task_name="snli",
        checkpoint_name="checkpoint-9",
    ) == checkpoint_9_path


@pytest.mark.parametrize("analysis_name", LOCAL_RESOLVER_ANALYSES)
def test_direction_activation_paths_report_missing_layers(
    analysis_name,
    tmp_path,
):
    activation_path = load_activation_path(analysis_name, tmp_path)
    base_directory = tmp_path / "creator" / "model" / "base"
    base_directory.mkdir(parents=True)

    with pytest.raises(KeyError, match=re.escape(LAYER)):
        activation_path(MODEL_ID, "snli", LAYER)


def test_shared_activation_source_follows_experimental_writer_layout(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        activation_source,
        "ACTIVATION_DATA_DIRECTORY",
        tmp_path,
    )
    model_directory = tmp_path / "creator" / "model"
    base_path = save_placeholder(
        model_directory / "base" / LAYER / "snli_activation.pt"
    )
    save_placeholder(
        model_directory
        / "snli"
        / "checkpoint-9"
        / LAYER
        / "snli_activation.pt"
    )
    checkpoint_10_path = save_placeholder(
        model_directory
        / "snli"
        / "checkpoint-10"
        / f"base_model.model.{LAYER}"
        / "snli_activation.pt"
    )
    (model_directory / "snli" / "notes").mkdir()

    assert activation_source.activation_path(
        MODEL_ID,
        "snli",
        LAYER,
    ) == base_path
    assert activation_source.activation_path(
        MODEL_ID,
        "snli",
        LAYER,
        task_name="snli",
    ) == checkpoint_10_path


def test_shared_activation_source_rejects_ambiguous_suffix_layers(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        activation_source,
        "ACTIVATION_DATA_DIRECTORY",
        tmp_path,
    )
    base_directory = tmp_path / "creator" / "model" / "base"
    save_placeholder(
        base_directory
        / f"first.{LAYER}"
        / "snli_activation.pt"
    )
    save_placeholder(
        base_directory
        / f"second.{LAYER}"
        / "snli_activation.pt"
    )

    with pytest.raises(ValueError, match="multiple saved activation"):
        activation_source.activation_path(MODEL_ID, "snli", LAYER)


def test_theoretical_root_contains_only_analysis_directories():
    theoretical_directory = Path(__file__).resolve().parents[2] / "theoretical"

    assert [
        path.name
        for path in theoretical_directory.iterdir()
        if path.is_file()
    ] == []


def test_activations_are_grouped_by_research_purpose():
    activation_directory = (
        Path(__file__).resolve().parents[2]
        / "theoretical"
        / "activations"
    )

    assert {
        path.name
        for path in activation_directory.iterdir()
        if path.is_file()
    } == {"source.py"}
    assert {
        path.name
        for path in activation_directory.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    } == set(ACTIVATION_PURPOSES)


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


def test_activation_notebooks_use_shared_source():
    activation_directory = (
        Path(__file__).resolve().parents[2]
        / "theoretical"
        / "activations"
    )
    for purpose in ACTIVATION_PURPOSES:
        notebook = json.loads(
            (
                activation_directory
                / purpose
                / "notebook.ipynb"
            ).read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        assert "from theoretical.activations.source import activation_path" in source
        assert "checkpoint_name" in source
        assert re.search(r"checkpoint-\d+", source) is None


def test_other_activation_consumers_use_direction_local_resolvers():
    theoretical_directory = Path(__file__).resolve().parents[2] / "theoretical"
    for analysis_name in LOCAL_RESOLVER_ANALYSES:
        notebook_path = (
            theoretical_directory / analysis_name / "notebook.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        source = "\n".join(code_cells)
        trees = [ast.parse(code_cell) for code_cell in code_cells]
        local_module = f"theoretical.{analysis_name}.analysis"
        local_imports = [
            alias.name
            for tree in trees
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == local_module
            for alias in node.names
        ]

        assert "activation_path" in local_imports
        assert "theoretical.activation_source" not in source
        assert "checkpoint_name" in source
        assert re.search(r"checkpoint-\d+", source) is None
