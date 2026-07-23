from experimental.lora_checkpoints import latest_task_lora_checkpoints


def test_latest_task_lora_checkpoints_use_numeric_checkpoint_order(tmp_path):
    task_dir = tmp_path / "snli"
    (task_dir / "checkpoint-9").mkdir(parents=True)
    checkpoint_10 = task_dir / "checkpoint-10"
    checkpoint_10.mkdir()

    assert latest_task_lora_checkpoints(str(tmp_path)) == {
        "snli": str(checkpoint_10)
    }


def test_latest_task_lora_checkpoints_returns_each_task(tmp_path):
    snli_checkpoint = tmp_path / "snli" / "checkpoint-2"
    snli_checkpoint.mkdir(parents=True)
    claim_checkpoint = tmp_path / "claim" / "checkpoint-3"
    claim_checkpoint.mkdir(parents=True)

    assert latest_task_lora_checkpoints(str(tmp_path)) == {
        "snli": str(snli_checkpoint),
        "claim": str(claim_checkpoint),
    }
