from pathlib import Path

from capture.capturer import latest_checkpoint


def test_latest_checkpoint_uses_numeric_checkpoint_suffix(tmp_path):
    task_dir = tmp_path / "snli"
    task_dir.mkdir()
    checkpoint_9 = task_dir / "checkpoint-9"
    checkpoint_10 = task_dir / "checkpoint-10"
    checkpoint_9.mkdir()
    checkpoint_10.mkdir()

    assert latest_checkpoint([str(checkpoint_9), str(checkpoint_10)]) == str(checkpoint_10)


def test_latest_checkpoint_falls_back_to_lexicographic_for_unknown_names(tmp_path):
    task_dir = tmp_path / "snli"
    task_dir.mkdir()
    first = task_dir / "alpha"
    second = task_dir / "beta"
    first.mkdir()
    second.mkdir()

    assert latest_checkpoint([str(second), str(first)]) == str(second)
