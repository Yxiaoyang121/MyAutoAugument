# Counterfactual-Adjusted Policy Ranking

- Formula: `counterfactual_adjusted_score = combined_proxy_safety_score + 0.20 * mean_direct_op_recovery_rate`
- copy_paste rule: copy_paste is not directly testable by prediction-only counterfactual diagnosis and is not hard rejected.

| rank | policy_id | original_rank | base_score | recovery_score | adjusted_score | direct_ops |
|---:|---|---:|---:|---:|---:|---|
| 1 | `diag_policy_001` | 1 | 0.9570 | 0.0400 | 0.9650 | clahe, contrast, gamma, brightness |
| 2 | `diag_policy_005` | 2 | 0.9561 | 0.0375 | 0.9636 | clahe, contrast, gamma, scale |
| 3 | `diag_policy_002` | 3 | 0.9536 | 0.0400 | 0.9616 | clahe, contrast, gamma, brightness |
| 4 | `diag_policy_003` | 5 | 0.9283 | 0.0350 | 0.9353 | scale, contrast |
| 5 | `diag_policy_004` | 4 | 0.9305 | 0.0100 | 0.9325 | scale |
