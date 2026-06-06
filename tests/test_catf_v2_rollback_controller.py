from __future__ import annotations

from pathlib import Path

import torch

from AutoAugment.catf_v2.policy_matrix import initial_policy_matrix
from AutoAugment.catf_v2.rollback_controller import CATFRollbackController, annotate_policy_history_with_rollback


class _Trainer:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.ema = None


def _policy() -> dict:
    policy = initial_policy_matrix({0: "defect"})
    row = policy["classes"]["0"]
    row["status"] = "active"
    row["state"] = "pending"
    row["ops"]["sharpen_mild"]["prob"] = 0.02
    row["ops"]["sharpen_mild"]["strength"] = 0.2
    return policy


def test_rb_saves_checkpoint_at_adaptive_trigger_epoch(tmp_path: Path) -> None:
    trainer = _Trainer()
    controller = CATFRollbackController(probe_window=5)
    event = controller.start_candidate(trainer=trainer, output_dir=tmp_path, epoch=10)

    assert event["action"] == "save_safe_checkpoint"
    assert event["checkpoint_epoch"] == 10
    assert Path(event["checkpoint_path"]).exists()
    assert controller.candidate_active is True


def test_rb_rejects_bad_probe_and_restores_state(tmp_path: Path) -> None:
    trainer = _Trainer()
    controller = CATFRollbackController(probe_window=5)
    controller.start_candidate(trainer=trainer, output_dir=tmp_path, epoch=10)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    with torch.no_grad():
        for param in trainer.model.parameters():
            param.add_(10.0)

    event = controller.maybe_evaluate_probe(
        trainer=trainer,
        epoch=15,
        policy=_policy(),
        metrics={"precision": 0.60, "recall": 0.40, "map50": 0.49, "map50_95": 0.29},
        reference_metrics={"precision": 0.61, "recall": 0.45, "map50": 0.51, "map50_95": 0.31},
        per_class_diagnosis={},
        active_classes=[0],
        output_dir=tmp_path,
    )

    assert event is not None
    assert event["action"] == "rollback"
    assert event["trainer_state_restored"] is True
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, before[key])
    assert max(
        op["prob"]
        for row in event["policy"]["classes"].values()
        for op in row["ops"].values()
    ) == 0.0


def test_rollback_decision_is_written_to_policy_history() -> None:
    history = [{"epoch": 15, "action": "accept", "guard_triggered": []}]
    event = {
        "action": "rollback",
        "checkpoint_epoch": 10,
        "checkpoint_path": "checkpoint.pt",
        "reasons": ["probe_map50_95_drop"],
        "policy": _policy(),
    }
    annotate_policy_history_with_rollback(history, event, event["policy"])
    assert history[0]["catf_rollback_mode"] is True
    assert history[0]["action"] == "rollback"
    assert history[0]["rb_checkpoint_epoch"] == 10
    assert "probe_map50_95_drop" in history[0]["guard_triggered"]
