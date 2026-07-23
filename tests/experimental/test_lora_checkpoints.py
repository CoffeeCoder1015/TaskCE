from experimental.lora_checkpoints import (
    latest_task_lora_checkpoint,
    latest_task_lora_checkpoints,
)


def test_latest_task_lora_checkpoint_uses_numeric_checkpoint_order(tmp_path):
    task_dir = tmp_path / "snli"
    (task_dir / "checkpoint-9").mkdir(parents=True)
    checkpoint_10 = task_dir / "checkpoint-10"
    checkpoint_10.mkdir()

    assert latest_task_lora_checkpoint(str(tmp_path), "snli") == str(checkpoint_10)


def test_latest_task_lora_checkpoints_returns_only_tasks_with_checkpoints(tmp_path):
    snli_checkpoint = tmp_path / "snli" / "checkpoint-2"
    snli_checkpoint.mkdir(parents=True)
    (tmp_path / "empty").mkdir()

    assert latest_task_lora_checkpoints(str(tmp_path)) == {
        "snli": str(snli_checkpoint)
    }
