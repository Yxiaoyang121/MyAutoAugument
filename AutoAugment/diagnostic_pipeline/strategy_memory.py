from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from AutoAugment.diagnostic_pipeline.common import json_safe, write_json, write_markdown


VECTOR_KEYS = [
    "small_object_score",
    "low_contrast_score",
    "class_imbalance_score",
    "localization_score",
    "false_positive_score",
]


def memory_guided_rerank(
    *,
    diagnosis: dict[str, Any],
    proxy_payload: dict[str, Any],
    output_dir: str | Path,
    memory_path: str | Path = "outputs/strategy_memory.jsonl",
    top_similar: int = 3,
) -> dict[str, Any]:
    """Rerank proxy candidates using similar historical diagnosis cases."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vector = flatten_diagnosis_vector(diagnosis.get("diagnosis_vector", {}))
    history = load_strategy_memory(memory_path)
    similar = sorted(
        [
            {
                "similarity": cosine_similarity(vector, flatten_diagnosis_vector(item.get("diagnosis_vector", {}))),
                "record": item,
            }
            for item in history
        ],
        key=lambda item: item["similarity"],
        reverse=True,
    )[: max(1, top_similar)]
    ranking: list[dict[str, Any]] = []
    for row in proxy_payload.get("ranking", []):
        candidate = dict(row)
        base_score = float(candidate.get("combined_proxy_safety_score", candidate.get("proxy_score", 0.0)) or 0.0)
        memory_boost, evidence = memory_boost_for_candidate(candidate, similar)
        candidate["memory_base_score"] = base_score
        candidate["memory_boost"] = memory_boost
        candidate["memory_guided_score"] = float(np.clip(base_score + memory_boost, 0.0, 1.0))
        candidate["memory_evidence"] = evidence
        ranking.append(candidate)
    ranking = sorted(
        ranking,
        key=lambda item: (
            bool(item.get("hard_filter_pass")),
            float(item.get("memory_guided_score", 0.0) or 0.0),
            float(item.get("combined_proxy_safety_score", item.get("proxy_score", 0.0)) or 0.0),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
        row["memory_guided_rank"] = rank
    payload = dict(proxy_payload)
    payload["ranking"] = ranking
    payload["memory_guided"] = {
        "enabled": True,
        "memory_path": str(Path(memory_path).resolve()),
        "history_count": len(history),
        "similar_case_count": len([item for item in similar if item["similarity"] > 0.0]),
        "diagnosis_vector": vector,
    }
    write_json(output / "memory_guided_ranking.json", payload)
    write_memory_report(output / "strategy_memory_report.md", payload, similar)
    return payload


def append_strategy_memory(
    *,
    diagnosis: dict[str, Any],
    policies_payload: dict[str, Any],
    proxy_payload: dict[str, Any],
    short_payload: dict[str, Any],
    output_dir: str | Path,
    memory_path: str | Path = "outputs/strategy_memory.jsonl",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Append one completed short-training selection record to strategy memory."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_policy = short_payload.get("selected_policy", {})
    selected_id = selected_policy.get("policy_id", selected_policy.get("name"))
    selected_proxy = find_policy_row(proxy_payload, selected_id)
    selected_trial = find_selected_trial(short_payload, selected_id)
    final_score = float(
        selected_trial.get("selection_score", selected_proxy.get("combined_proxy_safety_score", selected_proxy.get("proxy_score", 0.0)))
        or 0.0
    )
    record = {
        "diagnosis_vector": diagnosis.get("diagnosis_vector", {}),
        "candidate_policies": summarize_candidate_policies(policies_payload, proxy_payload),
        "proxy_score": selected_proxy.get("proxy_score"),
        "safety_score": selected_proxy.get("safety_score"),
        "short_train_metrics": selected_trial.get("val_metrics", {}),
        "selected_policy": selected_policy,
        "final_score": final_score,
    }
    memory_file = Path(memory_path)
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    appended = False
    if not dry_run:
        with memory_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
        appended = True
    payload = {
        "stage": "strategy_memory_append",
        "status": "skipped" if dry_run else "completed",
        "memory_path": str(memory_file.resolve()),
        "appended": appended,
        "record": record,
    }
    write_json(output / "strategy_memory_append.json", payload)
    write_markdown(
        output / "strategy_memory_append_report.md",
        [
            "# Strategy Memory Report",
            "",
            f"- Memory path: {memory_file.resolve()}",
            f"- Appended: {appended}",
            f"- Selected policy: {selected_id}",
            f"- Proxy score: {selected_proxy.get('proxy_score')}",
            f"- Safety score: {selected_proxy.get('safety_score')}",
            f"- Final score: {final_score:.4f}",
        ],
    )
    return payload


def load_strategy_memory(memory_path: str | Path) -> list[dict[str, Any]]:
    path = Path(memory_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def flatten_diagnosis_vector(vector: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in VECTOR_KEYS:
        value = vector.get(key, 0.0)
        if isinstance(value, dict):
            value = value.get("score", 0.0)
        try:
            out[key] = float(np.clip(float(value), 0.0, 1.0))
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    a = np.asarray([left.get(key, 0.0) for key in VECTOR_KEYS], dtype=np.float32)
    b = np.asarray([right.get(key, 0.0) for key in VECTOR_KEYS], dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.clip(float(np.dot(a, b) / denom), 0.0, 1.0))


def memory_boost_for_candidate(candidate: dict[str, Any], similar: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    candidate_ops = policy_ops(candidate.get("policy", {}))
    evidence: list[dict[str, Any]] = []
    raw_boost = 0.0
    for item in similar:
        similarity = float(item.get("similarity", 0.0) or 0.0)
        if similarity <= 0.15:
            continue
        record = item.get("record", {})
        historical_policy = record.get("selected_policy", {})
        overlap = jaccard(candidate_ops, policy_ops(historical_policy))
        final_score = float(record.get("final_score", 0.0) or 0.0)
        contribution = similarity * overlap * final_score
        if contribution <= 0.0:
            continue
        raw_boost += contribution
        evidence.append(
            {
                "similarity": similarity,
                "operator_overlap": overlap,
                "historical_final_score": final_score,
                "historical_policy_id": historical_policy.get("policy_id", historical_policy.get("name")),
                "contribution": contribution,
            }
        )
    return float(np.clip(0.12 * raw_boost, 0.0, 0.20)), evidence


def policy_ops(policy: dict[str, Any]) -> set[str]:
    return {str(operation.get("name", "")).lower() for operation in policy.get("operations", []) if operation.get("name")}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def summarize_candidate_policies(policies_payload: dict[str, Any], proxy_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_id = {
        row.get("policy_id", row.get("policy", {}).get("policy_id")): row
        for row in proxy_payload.get("ranking", [])
    }
    out: list[dict[str, Any]] = []
    for policy in policies_payload.get("policies", []):
        policy_id = policy.get("policy_id", policy.get("name"))
        row = rows_by_id.get(policy_id, {})
        out.append(
            {
                "policy_id": policy_id,
                "source_issue": policy.get("source_issue"),
                "severity_score": policy.get("severity_score"),
                "operations": policy.get("operations", []),
                "proxy_score": row.get("proxy_score"),
                "safety_score": row.get("safety_score"),
                "combined_proxy_safety_score": row.get("combined_proxy_safety_score"),
                "memory_guided_score": row.get("memory_guided_score"),
            }
        )
    return out


def find_policy_row(proxy_payload: dict[str, Any], policy_id: Any) -> dict[str, Any]:
    for row in proxy_payload.get("ranking", []):
        if row.get("policy_id") == policy_id:
            return row
    return {}


def find_selected_trial(short_payload: dict[str, Any], policy_id: Any) -> dict[str, Any]:
    for trial in short_payload.get("trials", []):
        if trial.get("policy_id") == policy_id:
            return trial
    return {}


def write_memory_report(path: str | Path, payload: dict[str, Any], similar: list[dict[str, Any]]) -> None:
    metadata = payload.get("memory_guided", {})
    lines = [
        "# Strategy Memory Report",
        "",
        f"- Memory path: {metadata.get('memory_path')}",
        f"- Historical records: {metadata.get('history_count', 0)}",
        f"- Similar cases used: {metadata.get('similar_case_count', 0)}",
        "",
        "## Similar Cases",
    ]
    if not similar:
        lines.append("- No prior strategy memory records found.")
    for item in similar:
        record = item.get("record", {})
        selected = record.get("selected_policy", {})
        lines.append(
            f"- similarity={float(item.get('similarity', 0.0)):.4f} "
            f"policy={selected.get('policy_id', selected.get('name'))} "
            f"final_score={float(record.get('final_score', 0.0) or 0.0):.4f}"
        )
    lines.extend(["", "## Reranked Candidates"])
    for row in payload.get("ranking", []):
        lines.append(
            f"- rank={row.get('memory_guided_rank')} policy={row.get('policy_id')} "
            f"base={float(row.get('memory_base_score', 0.0) or 0.0):.4f} "
            f"boost={float(row.get('memory_boost', 0.0) or 0.0):.4f} "
            f"score={float(row.get('memory_guided_score', 0.0) or 0.0):.4f}"
        )
    write_markdown(path, lines)
