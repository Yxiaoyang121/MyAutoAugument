# Adaptive Burn-in CATF-v2 Retrospective Simulation

Generated: `2026-06-06T13:06:44`
Code commit used: `671a3e8f164dac4eff76193ba291004926667a4a`

## Seed Decisions

| seed | adaptive start epoch | candidate started | no-op fallback | no-op epoch | strong clean baseline protection | expected path |
|---:|---:|---|---|---:|---|---|
| 0 | 15 | true | false | None | false | adaptive_rb_candidate |
| 1 | 15 | true | false | None | false | adaptive_rb_candidate |
| 2 | None | false | true | 15 | true | clean_noop_fallback |

## Required Answers

1. Seed0 adaptive trigger epoch: `15`.
2. Seed1 adaptive trigger epoch: `15`.
3. Seed2 candidate trigger: `false`; strong clean baseline protection: `true`.
4. Avoids Safe epoch5 early no-op: `true`.
5. Avoids Gated seed2 epoch10 late fallback: `true`.
6. Start conditions are recorded per seed in the event trace below.
7. Key blocking/triggering conditions: metric stability, max-burnin low-risk force for seed0/1, and final strong clean-baseline protection for seed2.
8. Overfit assessment: The trigger uses metric stability, diagnosis evidence, support guards, no-aug/high-FP guards, and final clean-baseline protection. It does not branch on seed id or class id, but the default thresholds are still calibrated from the current three-seed audit and need larger validation.
9. Recommended to enter real training validation: `true`.

## Event Trace

### Seed 0

| epoch | action | start_condition | reasons | map50_range | recall_range | eligible_classes | medium_issue_classes |
|---:|---|---|---|---:|---:|---|---|
| 5 | burnin_observe | false | `['metric_unstable']` | 0.0540 | 0.1766 | `[4, 11]` | `[4, 11]` |
| 10 | burnin_observe | false | `['metric_unstable']` | 0.0376 | 0.0067 | `[11]` | `[11]` |
| 15 | start_candidate_low_risk | false | `['force_start_at_max_burnin_low_risk_issue']` | 0.0354 | 0.0742 | `[12]` | `[12]` |

### Seed 1

| epoch | action | start_condition | reasons | map50_range | recall_range | eligible_classes | medium_issue_classes |
|---:|---|---|---|---:|---:|---|---|
| 5 | burnin_observe | false | `['metric_unstable']` | 0.0515 | 0.0845 | `[4, 11]` | `[4, 5, 11]` |
| 10 | burnin_observe | false | `['metric_unstable', 'insufficient_diagnosis_evidence']` | 0.0290 | 0.0482 | `[]` | `[]` |
| 15 | start_candidate_low_risk | false | `['force_start_at_max_burnin_low_risk_issue']` | 0.0489 | 0.0775 | `[12]` | `[12]` |

### Seed 2

| epoch | action | start_condition | reasons | map50_range | recall_range | eligible_classes | medium_issue_classes |
|---:|---|---|---|---:|---:|---|---|
| 5 | burnin_observe | false | `['metric_unstable', 'strong_clean_baseline_protection']` | 0.0744 | 0.1673 | `[9]` | `[9]` |
| 10 | burnin_observe | false | `['metric_unstable', 'strong_clean_baseline_protection']` | 0.1045 | 0.0886 | `[6, 8, 11, 12]` | `[6, 8, 11, 12]` |
| 15 | no_op_fallback | false | `['adaptive_burnin_not_ready']` | 0.0397 | 0.0390 | `[6]` | `[6]` |
