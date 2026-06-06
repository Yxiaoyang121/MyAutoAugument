from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from AutoAugment.catf_v2.gated_controller import gate_bad_patterns, positive_gain
from AutoAugment.catf_v2.safe_controller import compact_metrics, force_noop_policy, metric_delta


@dataclass
class CATFRollbackController:
    """First-branch rollback helper for CATF-v2 adaptive candidate probes."""

    probe_window: int = 5
    checkpoint_epoch: int | None = None
    checkpoint_path: str | None = None
    checkpoint_state: dict[str, Any] | None = None
    candidate_active: bool = False
    accepted: bool = False
    rolled_back: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    def start_candidate(self, *, trainer: Any, output_dir: Path, epoch: int) -> dict[str, Any]:
        self.checkpoint_epoch = int(epoch)
        self.candidate_active = True
        self.accepted = False
        self.rolled_back = False
        path = output_dir / "checkpoints" / f"catf_v2_rb_safe_epoch_{int(epoch):03d}.pt"
        state = capture_trainer_state(trainer)
        self.checkpoint_state = state
        self.checkpoint_path = str(path.resolve())
        save_checkpoint_payload(path, state)
        event = {
            "epoch": int(epoch),
            "catf_rollback_mode": True,
            "action": "save_safe_checkpoint",
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_path": self.checkpoint_path,
            "candidate_active": True,
            "probe_window": int(self.probe_window),
            "restorable_components": sorted(state.keys()),
        }
        self.events.append(deepcopy(event))
        return event

    def maybe_evaluate_probe(
        self,
        *,
        trainer: Any,
        epoch: int,
        policy: dict[str, Any],
        metrics: dict[str, Any],
        reference_metrics: dict[str, Any],
        per_class_diagnosis: dict[str, Any] | None,
        active_classes: list[int],
        output_dir: Path,
    ) -> dict[str, Any] | None:
        if not self.candidate_active or self.accepted or self.rolled_back or self.checkpoint_epoch is None:
            return None
        epoch = int(epoch)
        gate_epoch = self.checkpoint_epoch + max(1, int(self.probe_window))
        if epoch < gate_epoch:
            return None
        delta = metric_delta(metrics, reference_metrics)
        bad_patterns = gate_bad_patterns(
            delta,
            per_class_diagnosis=per_class_diagnosis,
            previous_per_class=None,
            active_classes=active_classes,
        )
        reject_reasons: list[str] = []
        if bad_patterns:
            reject_reasons.extend(bad_patterns)
        if delta.get("map50_95") is not None and float(delta["map50_95"]) < -0.008:
            reject_reasons.append("probe_map50_95_drop")
        if delta.get("map50") is not None and float(delta["map50"]) < -0.008:
            reject_reasons.append("probe_map50_drop")
        if delta.get("precision") is not None and float(delta["precision"]) < -0.010:
            reject_reasons.append("probe_precision_drop")
        if not positive_gain(delta) and (float(delta.get("recall") or 0.0) < 0.0 or float(delta.get("map50_95") or 0.0) < 0.0):
            reject_reasons.append("probe_no_positive_gain_with_global_drop")
        reject_reasons = list(dict.fromkeys(reject_reasons))

        if reject_reasons:
            restored = restore_trainer_state(trainer, self.checkpoint_state)
            rollback_dir = output_dir / "checkpoints"
            rollback_dir.mkdir(parents=True, exist_ok=True)
            event = {
                "epoch": epoch,
                "catf_rollback_mode": True,
                "action": "rollback",
                "checkpoint_epoch": self.checkpoint_epoch,
                "checkpoint_path": self.checkpoint_path,
                "probe_window": int(self.probe_window),
                "metrics": compact_metrics(metrics),
                "reference_metrics": compact_metrics(reference_metrics),
                "delta_metrics": delta,
                "bad_patterns": bad_patterns,
                "reasons": reject_reasons,
                "trainer_state_restored": bool(restored),
                "policy": force_noop_policy(policy, reason="catf_v2_rb_probe_rejected"),
            }
            self.rolled_back = True
            self.candidate_active = False
            self.events.append(_without_policy(event))
            return event

        event = {
            "epoch": epoch,
            "catf_rollback_mode": True,
            "action": "accept_candidate",
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_path": self.checkpoint_path,
            "probe_window": int(self.probe_window),
            "metrics": compact_metrics(metrics),
            "reference_metrics": compact_metrics(reference_metrics),
            "delta_metrics": delta,
            "bad_patterns": [],
            "reasons": ["probe_positive_or_constraint_safe"],
            "policy": deepcopy(policy),
        }
        self.accepted = True
        self.candidate_active = False
        self.events.append(_without_policy(event))
        return event


def capture_trainer_state(trainer: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    model = getattr(trainer, "model", None)
    if hasattr(model, "state_dict"):
        state["model"] = deepcopy(model.state_dict())
    ema = getattr(trainer, "ema", None)
    ema_model = getattr(ema, "ema", None)
    if hasattr(ema_model, "state_dict"):
        state["ema"] = deepcopy(ema_model.state_dict())
    optimizer = getattr(trainer, "optimizer", None)
    if hasattr(optimizer, "state_dict"):
        state["optimizer"] = deepcopy(optimizer.state_dict())
    scheduler = getattr(trainer, "scheduler", None)
    if hasattr(scheduler, "state_dict"):
        state["scheduler"] = deepcopy(scheduler.state_dict())
    return state


def restore_trainer_state(trainer: Any, state: dict[str, Any] | None) -> bool:
    if not state:
        return False
    restored = False
    model = getattr(trainer, "model", None)
    if "model" in state and hasattr(model, "load_state_dict"):
        model.load_state_dict(state["model"], strict=False)
        restored = True
    ema = getattr(trainer, "ema", None)
    ema_model = getattr(ema, "ema", None)
    if "ema" in state and hasattr(ema_model, "load_state_dict"):
        ema_model.load_state_dict(state["ema"], strict=False)
        restored = True
    optimizer = getattr(trainer, "optimizer", None)
    if "optimizer" in state and hasattr(optimizer, "load_state_dict"):
        optimizer.load_state_dict(state["optimizer"])
        restored = True
    scheduler = getattr(trainer, "scheduler", None)
    if "scheduler" in state and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state["scheduler"])
        restored = True
    return restored


def save_checkpoint_payload(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        torch.save(state, path)
    except Exception:
        path.with_suffix(".txt").write_text("checkpoint serialization unavailable\n", encoding="utf-8")


def annotate_policy_history_with_rollback(history: list[dict[str, Any]], event: dict[str, Any], new_policy: dict[str, Any]) -> None:
    if not history or not event:
        return
    latest = history[-1]
    event_for_history = _without_policy(event)
    latest["catf_rollback_mode"] = True
    latest["rollback_controller_event"] = event_for_history
    latest["rb_candidate_active"] = bool(event_for_history.get("candidate_active"))
    latest["rb_checkpoint_epoch"] = event_for_history.get("checkpoint_epoch")
    latest["rb_checkpoint_path"] = event_for_history.get("checkpoint_path")
    if event_for_history.get("action") == "rollback":
        latest["action"] = "rollback"
        latest["accepted_policy"] = deepcopy(new_policy)
        latest["new_policy"] = deepcopy(new_policy)
        latest["rollback_reason"] = ",".join(event_for_history.get("reasons") or [])
        latest["guard_triggered"] = list(
            dict.fromkeys(list(latest.get("guard_triggered", []) or []) + list(event_for_history.get("reasons", []) or []))
        )
    elif event_for_history.get("action") == "accept_candidate":
        latest["rb_candidate_accepted"] = True


def _without_policy(event: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(event)
    out.pop("policy", None)
    return out
