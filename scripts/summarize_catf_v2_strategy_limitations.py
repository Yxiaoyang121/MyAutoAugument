"""Summarize CATF-v2 strategy limitations from the seed2 audit.

This script is report-only. It reads completed audit artifacts and writes a
Markdown/JSON summary. It does not run training, validation, or prediction.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_REPORTS = PROJECT_ROOT / "outputs/experiments/seed2_failure_root_cause/reports"
FIXED_SUMMARY = (
    PROJECT_ROOT
    / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.json"
)
OUTPUT_MD = AUDIT_REPORTS / "catf_v2_strategy_limitation_analysis.md"
OUTPUT_JSON = AUDIT_REPORTS / "catf_v2_strategy_limitation_analysis.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def f4(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def fd(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.4f}"


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ("precision", "recall", "map50", "map50_95")
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def metric_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: float(a[key]) - float(b[key]) for key in ("precision", "recall", "map50", "map50_95")}


def class_list(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    def deltas(row: dict[str, Any]) -> dict[str, Any]:
        return row.get("delta") or row.get("final_delta") or {}

    return [
        {
            "class_id": row["class_id"],
            "class_name": row.get("class_name"),
            "delta_recall": deltas(row).get("recall", row.get("delta_recall")),
            "delta_ap50": deltas(row).get("ap50", row.get("delta_ap50")),
            "delta_ap50_95": deltas(row).get("ap50_95", row.get("delta_ap50_95")),
            "active": row.get("active_class", row.get("active", "final_delta" in row)),
            "roi_applied": row.get("roi_applied", 0),
        }
        for row in rows[:limit]
    ]


def format_class_items(rows: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"class {row['class_id']} "
        f"(dR={fd(row.get('delta_recall'))}, dAP50={fd(row.get('delta_ap50'))}, "
        f"dAP50-95={fd(row.get('delta_ap50_95'))}, active={str(row.get('active')).lower()}, "
        f"ROI={row.get('roi_applied', 0)})"
        for row in rows
    )


def build_payload() -> dict[str, Any]:
    root = read_json(AUDIT_REPORTS / "seed2_root_cause_summary.json")
    curve = read_json(AUDIT_REPORTS / "seed2_curve_degradation_analysis.json")
    per_class = read_json(AUDIT_REPORTS / "seed2_per_class_regression_analysis.json")
    operator = read_json(AUDIT_REPORTS / "seed2_augmentation_operator_attribution.json")
    active = read_json(AUDIT_REPORTS / "seed2_active_vs_regressed_class_analysis.json")
    prediction = read_json(AUDIT_REPORTS / "seed2_prediction_diff_analysis.json")
    fixed = read_json(FIXED_SUMMARY)

    fixed_seeds = fixed["seeds"]
    clean_avg = average_metrics([fixed_seeds[str(seed)]["clean_native"] for seed in (0, 1, 2)])
    fixed_avg = average_metrics([fixed_seeds[str(seed)]["fixed_catf_v2"] for seed in (0, 1, 2)])
    avg_delta = metric_delta(fixed_avg, clean_avg)

    fixed_vs_clean = curve["fixed_vs_clean"]
    first_lag = fixed_vs_clean["first_lag"]
    clean_seed2 = curve["clean_final_metrics"]
    fixed_seed2 = curve["fixed_final_metrics"]
    seed2_delta = metric_delta(fixed_seed2, clean_seed2)

    online = operator["online_aug_stats"]
    roi = operator["roi_aug_stats"]
    ops = online.get("ops") or {}

    recall_drop = class_list(per_class["top_recall_drop"])
    ap50_drop = class_list(per_class["top_ap50_drop"])
    ap95_drop = class_list(per_class["top_ap50_95_drop"])
    active_regressed = class_list(active["regressed_active_classes"], limit=5)
    non_active = class_list(active["non_active_regressions"], limit=8)

    limitations = [
        {
            "title": "诊断结果不能直接等价于增强收益",
            "evidence": (
                "epoch5 将 class 9 诊断为 texture_boundary_weak 且置信度高，但随后 Recall 在 epoch6 "
                "落后 clean，mAP50/mAP50-95 在 epoch8 明显落后。"
            ),
        },
        {
            "title": "active class 的局部问题并不保证该类受益",
            "evidence": (
                "class 9 是 epoch5 active class，ROI applied=25，但最终 Recall -0.1667、"
                "AP50 -0.0466、AP50-95 -0.0654，估算 FN +14。"
            ),
        },
        {
            "title": "缺少增强前因果验证",
            "evidence": (
                "当前策略在诊断后直接进入图像增强，未先验证候选增强是否能恢复 FN、是否增加 FP、"
                "是否伤害非 active 类。"
            ),
        },
        {
            "title": "只约束 active class 不足以控制 non-active regression",
            "evidence": (
                "class 10、12、6、5、3、2 等非 ROI 类出现 AP 或 Recall 退化；"
                "说明增强 A 类可能改变共享特征并伤害 B 类。"
            ),
        },
        {
            "title": "算子级风险没有被隔离",
            "evidence": (
                "seed2 中 sharpen_mild 与 local_contrast 共同启用，local_contrast applied=44、"
                "sharpen_mild applied=42，现有日志无法拆分单个算子的责任。"
            ),
        },
        {
            "title": "fallback/gate 发现问题时权重可能已经被影响",
            "evidence": (
                "Gated 在 epoch10 fallback，但 epoch6-10 已经经过候选增强训练；"
                "因此 strict no-op 后续路径不等价于 clean 权重轨迹。"
            ),
        },
        {
            "title": "强 clean baseline 下策略应更保守",
            "evidence": (
                "Safe 与 adaptive-RB strict no-op 完全复现 clean seed2；fixed/Gated 失败，"
                "说明 seed2 这类强 clean baseline 更适合 no-op 或 sampler-only。"
            ),
        },
    ]

    paper_text = (
        "反馈增强策略在工业缺陷检测中需要同时考虑收益与风险。我们的实验表明，"
        "类别感知反馈增强并非在所有随机种子上都产生一致收益：在 seed0 和 seed1 上，"
        "CATF-v2 能带来 mAP 或多指标提升；但在 seed2 上，候选增强使模型呈现更保守的预测行为，"
        "即 Precision 上升而 Recall 与定位相关 AP 下降。进一步分析发现，诊断出的类别问题并不能直接等价为"
        "图像增强收益。以 class 9 为例，该类被诊断为边界纹理相关问题，但 ROI texture 增强后该类 Recall 和 AP "
        "反而下降，并伴随若干非目标类别退化。这说明反馈增强策略需要在训练前引入因果验证，检查候选增强是否确实"
        "恢复漏检、是否引入新的误检，以及是否损害非 active 类。对于工业应用，增强策略不应只最大化 active class "
        "的局部收益，还应将全局 Recall、定位质量和非目标类稳定性纳入风险约束。"
    )

    improvements = [
        {
            "name": "增强前 causal probe",
            "description": "不更新权重，先验证候选增强是否恢复 FN、是否增加 FP、是否伤害非 active 类。",
            "why_minimal": "只增加干预前验证层，不改变现有 CATF-v2 增强算子或训练流程。",
        },
        {
            "name": "active / non-active 双重约束",
            "description": "候选策略不仅看 active class 是否改善，还要监控其他类别 AP/Recall 是否退化。",
            "why_minimal": "把已有 per-class diagnosis 与 prediction diff 转成 gate 条件，避免增强 A 类伤害 B 类。",
        },
        {
            "name": "高风险 class-op 组合黑名单",
            "description": "class 9 + ROI texture 这类已暴露风险的组合进入高风险候选池，必须通过 causal probe 才允许训练。",
            "why_minimal": "不是永久禁用算子，而是对有证据的风险组合提高准入门槛。",
        },
    ]

    conclusions = {
        "needs_large_ablation_to_prove_limitation": False,
        "existing_evidence_sufficient_for_strategy_limitation": True,
        "main_seed2_problem": "诊断到增强的策略选择缺少因果验证，并且 fallback/gate 在权重已受影响后才触发。",
        "next_best_implementation": "CP-CATF causal probe",
        "paper_innovation_conversion": (
            "将 seed2 暴露的问题转化为从 diagnosis-driven augmentation 到 causality-checked feedback augmentation "
            "的动机：增强动作必须先通过 FN/FP/non-active regression 风险验证。"
        ),
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": {
            "no_training": True,
            "no_ablation": True,
            "no_catf_v2_rule_change": True,
            "input_reports": [
                "seed2_root_cause_summary.md",
                "seed2_curve_degradation_analysis.md",
                "seed2_per_class_regression_analysis.md",
                "seed2_augmentation_operator_attribution.md",
                "seed2_active_vs_regressed_class_analysis.md",
                "seed2_prediction_diff_analysis.md",
            ],
        },
        "seed2_failure_chain": {
            "clean_seed2": clean_seed2,
            "fixed_seed2": fixed_seed2,
            "fixed_delta_vs_clean": seed2_delta,
            "degradation_epochs": {
                "recall_first_lag": first_lag["recall"]["first_negative_epoch"],
                "recall_first_clear_lag": first_lag["recall"]["first_delta_le_minus_0_01_epoch"],
                "map50_first_clear_lag": first_lag["map50"]["first_delta_le_minus_0_01_epoch"],
                "map50_95_first_negative": first_lag["map50_95"]["first_negative_epoch"],
                "map50_95_first_clear_lag": first_lag["map50_95"]["first_delta_le_minus_0_01_epoch"],
            },
            "most_suspicious_policy_update": root["most_suspicious_policy_update"],
            "most_suspicious_active_class": root["most_suspicious_active_class"],
            "most_suspicious_operator_family": root["most_suspicious_op"],
            "augmentation_stats": {
                "industrial_samples_augmented": online.get("samples_augmented"),
                "roi_applied": roi.get("roi_aug_applied"),
                "router_random_draw_count": online.get("router_random_draw_count"),
                "ops": ops,
                "roi_affected_classes": roi.get("affected_classes"),
            },
            "worst_recall_drop": recall_drop,
            "worst_ap50_drop": ap50_drop,
            "worst_ap50_95_drop": ap95_drop,
            "active_regressed_classes": active_regressed,
            "non_active_regression_exists": bool(non_active),
            "non_active_regressions": non_active,
            "prediction_diff": {
                "clean_detected_fixed_missed_count": prediction["clean_detected_fixed_missed_count"],
                "clean_high_iou_fixed_worse_count": prediction["clean_high_iou_fixed_worse_count"],
                "fixed_confidence_drop_count": prediction["fixed_confidence_drop_count"],
                "fixed_new_high_conf_fp_count": prediction["fixed_new_high_conf_fp_count"],
            },
        },
        "catf_v2_not_invalid": {
            "seed0_delta_fixed_vs_clean": fixed_seeds["0"]["delta_fixed_vs_clean"],
            "seed1_delta_fixed_vs_clean": fixed_seeds["1"]["delta_fixed_vs_clean"],
            "fixed_constraint_pass_seed0": not fixed_seeds["0"]["fixed_constraint_failed"],
            "fixed_constraint_pass_seed1": not fixed_seeds["1"]["fixed_constraint_failed"],
            "average_clean": clean_avg,
            "average_fixed": fixed_avg,
            "average_delta_fixed_vs_clean": avg_delta,
            "interpretation": "seed2 暴露的是策略选择与风险控制不足，不是类别感知反馈增强方向错误。",
        },
        "main_limitations": limitations,
        "paper_ready_text": paper_text,
        "minimal_improvements": improvements,
        "final_conclusions": conclusions,
    }


def write_report(payload: dict[str, Any]) -> None:
    chain = payload["seed2_failure_chain"]
    not_invalid = payload["catf_v2_not_invalid"]
    lines = [
        "# CATF-v2 策略不足分析与后续改进建议",
        "",
        "本报告基于已有 seed2 failure root-cause audit 结果整理。未训练新模型，未运行消融，未修改 CATF-v2 规则。",
        "",
        "## 一、Seed2 失败链条总结",
        "",
        "### 时间线",
        "",
        "| 阶段 | 证据 | 结论 |",
        "|---|---|---|",
        (
            "| clean seed2 | "
            f"P={f4(chain['clean_seed2']['precision'])}, R={f4(chain['clean_seed2']['recall'])}, "
            f"mAP50={f4(chain['clean_seed2']['map50'])}, mAP50-95={f4(chain['clean_seed2']['map50_95'])} | "
            "clean baseline 本身较强，尤其 Recall 和 mAP50-95 是当前 seed2 保护目标 |"
        ),
        (
            "| fixed CATF-v2 seed2 | "
            f"P={f4(chain['fixed_seed2']['precision'])}, R={f4(chain['fixed_seed2']['recall'])}, "
            f"mAP50={f4(chain['fixed_seed2']['map50'])}, mAP50-95={f4(chain['fixed_seed2']['map50_95'])}; "
            f"dP={fd(chain['fixed_delta_vs_clean']['precision'])}, dR={fd(chain['fixed_delta_vs_clean']['recall'])}, "
            f"dM50={fd(chain['fixed_delta_vs_clean']['map50'])}, dM95={fd(chain['fixed_delta_vs_clean']['map50_95'])} | "
            "不是 Precision 问题，而是更保守的漏检和定位质量下降 |"
        ),
        (
            "| epoch 5 | "
            f"{chain['most_suspicious_policy_update']} | "
            "最可疑的首次策略更新，发生在退化前 |"
        ),
        (
            "| epoch 6-8 | "
            f"Recall first lag={chain['degradation_epochs']['recall_first_lag']}; "
            f"mAP50 clear lag={chain['degradation_epochs']['map50_first_clear_lag']}; "
            f"mAP50-95 clear lag={chain['degradation_epochs']['map50_95_first_clear_lag']} | "
            "退化紧跟 epoch5 候选增强窗口出现 |"
        ),
        (
            "| final per-class | "
            f"{format_class_items(chain['active_regressed_classes'])} | "
            "active class 与退化类存在重叠，class 9 最关键 |"
        ),
        (
            "| non-active classes | "
            f"{format_class_items(chain['non_active_regressions'][:5])} | "
            "存在增强 A 类、伤害 B 类的非目标类退化 |"
        ),
        "",
        "### 增强执行证据",
        "",
        f"- Active class 最可疑：`{chain['most_suspicious_active_class']}`。",
        f"- 最可疑算子组合：`{chain['most_suspicious_operator_family']}`。",
        f"- Industrial samples augmented：`{chain['augmentation_stats']['industrial_samples_augmented']}`。",
        f"- ROI applied：`{chain['augmentation_stats']['roi_applied']}`。",
        f"- Router random draw count：`{chain['augmentation_stats']['router_random_draw_count']}`。",
        f"- ROI affected classes：`{chain['augmentation_stats']['roi_affected_classes']}`。",
        f"- Prediction diff：clean 检出但 fixed 漏检 `{chain['prediction_diff']['clean_detected_fixed_missed_count']}` 个 GT；clean 高 IoU 但 fixed 定位变差 `{chain['prediction_diff']['clean_high_iou_fixed_worse_count']}` 个 GT。",
        "",
        "## 二、当前 CATF-v2 策略的主要不足",
        "",
    ]
    for idx, item in enumerate(payload["main_limitations"], start=1):
        lines += [
            f"{idx}. **{item['title']}**",
            f"   - 证据：{item['evidence']}",
        ]
    lines += [
        "",
        "## 三、这不是“方法失败”",
        "",
        "CATF-v2 并非无效。fixed CATF-v2 在 seed0/seed1 上保留了有效收益，seed2 暴露的是策略选择与风险控制不足，而不是类别感知反馈增强方向错误。",
        "",
        "| seed | dP fixed-clean | dR fixed-clean | dM50 fixed-clean | dM95 fixed-clean | constraint pass |",
        "|---:|---:|---:|---:|---:|:---:|",
        (
            f"| 0 | {fd(not_invalid['seed0_delta_fixed_vs_clean']['precision'])} | "
            f"{fd(not_invalid['seed0_delta_fixed_vs_clean']['recall'])} | "
            f"{fd(not_invalid['seed0_delta_fixed_vs_clean']['map50'])} | "
            f"{fd(not_invalid['seed0_delta_fixed_vs_clean']['map50_95'])} | "
            f"{str(not_invalid['fixed_constraint_pass_seed0']).lower()} |"
        ),
        (
            f"| 1 | {fd(not_invalid['seed1_delta_fixed_vs_clean']['precision'])} | "
            f"{fd(not_invalid['seed1_delta_fixed_vs_clean']['recall'])} | "
            f"{fd(not_invalid['seed1_delta_fixed_vs_clean']['map50'])} | "
            f"{fd(not_invalid['seed1_delta_fixed_vs_clean']['map50_95'])} | "
            f"{str(not_invalid['fixed_constraint_pass_seed1']).lower()} |"
        ),
        (
            f"| avg 0/1/2 | {fd(not_invalid['average_delta_fixed_vs_clean']['precision'])} | "
            f"{fd(not_invalid['average_delta_fixed_vs_clean']['recall'])} | "
            f"{fd(not_invalid['average_delta_fixed_vs_clean']['map50'])} | "
            f"{fd(not_invalid['average_delta_fixed_vs_clean']['map50_95'])} | n/a |"
        ),
        "",
        "## 四、论文可用表述：反馈增强策略的局限性分析",
        "",
        payload["paper_ready_text"],
        "",
        "## 五、后续最小改进方向",
        "",
    ]
    for idx, item in enumerate(payload["minimal_improvements"], start=1):
        lines += [
            f"{idx}. **{item['name']}**",
            f"   - 做法：{item['description']}",
            f"   - 最小性：{item['why_minimal']}",
        ]
    conclusions = payload["final_conclusions"]
    lines += [
        "",
        "## 六、最终结论",
        "",
        f"1. 是否需要大量消融才能证明当前不足：`{str(conclusions['needs_large_ablation_to_prove_limitation']).lower()}`。已有 seed2 audit 已足够证明策略选择存在不足；大量消融是后续优化，不是证明不足的前提。",
        f"2. 当前证据是否足够说明 CATF-v2 策略选择存在不足：`{str(conclusions['existing_evidence_sufficient_for_strategy_limitation']).lower()}`。",
        f"3. seed2 最主要暴露的问题：{conclusions['main_seed2_problem']}",
        f"4. 下一步最值得实现：`{conclusions['next_best_implementation']}`。",
        f"5. 如何转化为论文创新点：{conclusions['paper_innovation_conversion']}",
    ]
    write_md(OUTPUT_MD, lines)


def main() -> None:
    payload = build_payload()
    write_json(OUTPUT_JSON, payload)
    write_report(payload)


if __name__ == "__main__":
    main()
