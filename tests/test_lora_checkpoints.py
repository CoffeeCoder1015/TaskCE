import importlib.util
from pathlib import Path


module_path = Path(__file__).resolve().parents[1] / "capture" / "lora_checkpoints.py"
spec = importlib.util.spec_from_file_location("lora_checkpoints_under_test", module_path)
lora_checkpoints = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(lora_checkpoints)

latest_task_lora_checkpoint = lora_checkpoints.latest_task_lora_checkpoint
latest_task_lora_checkpoints = lora_checkpoints.latest_task_lora_checkpoints
resolve_lora_root = lora_checkpoints.resolve_lora_root


def test_latest_task_lora_checkpoint_uses_latest_local_checkpoint(tmp_path):
    task_dir = tmp_path / "snli"
    task_dir.mkdir()
    checkpoint_9 = task_dir / "checkpoint-9"
    checkpoint_10 = task_dir / "checkpoint-10"
    checkpoint_9.mkdir()
    checkpoint_10.mkdir()

    assert latest_task_lora_checkpoint(str(tmp_path), "snli") == str(checkpoint_10)


def test_latest_task_lora_checkpoints_returns_present_tasks(tmp_path):
    snli_dir = tmp_path / "snli" / "checkpoint-1"
    claim_dir = tmp_path / "claim" / "checkpoint-2"
    snli_dir.mkdir(parents=True)
    claim_dir.mkdir(parents=True)

    assert latest_task_lora_checkpoints(str(tmp_path)) == {
        "snli": str(snli_dir),
        "claim": str(claim_dir),
    }


def test_resolve_lora_root_uses_local_dir_by_default(tmp_path):
    assert resolve_lora_root(str(tmp_path)) == str(tmp_path)


def test_resolve_lora_root_snapshots_hub_repo_when_remote(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(repo_id, repo_type, token):
        calls.append((repo_id, repo_type, token))
        return str(tmp_path)

    monkeypatch.setattr(lora_checkpoints, "snapshot_download", snapshot_download)

    assert resolve_lora_root("org/private-loras", remote=True) == str(tmp_path)
    assert calls == [("org/private-loras", "model", None)]


def test_resolve_lora_root_passes_token_to_hub_download(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(repo_id, repo_type, token):
        calls.append((repo_id, repo_type, token))
        return str(tmp_path)

    monkeypatch.setattr(lora_checkpoints, "snapshot_download", snapshot_download)

    assert (
        resolve_lora_root("org/private-loras", remote=True, token="hf_test")
        == str(tmp_path)
    )
    assert calls == [("org/private-loras", "model", "hf_test")]
