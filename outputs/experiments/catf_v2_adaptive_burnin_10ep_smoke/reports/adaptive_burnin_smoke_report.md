# Adaptive Burn-in CATF-v2 Smoke Report

## Answers

- Adaptive burn-in enabled: `true`
- Epoch 5 start_condition checked: `true`
- Candidate branch started: `false`
- Adaptive start epoch: `None`
- No-start reason at epoch 5: `['metric_unstable', 'strong_clean_baseline_protection']`
- No-op fallback entered: `false`
- Strict no-op so far: `true`
- BBox/class legal: `true`
- Industrial samples augmented: `0`
- ROI applied: `0`
- CATF router random draw count: `0`
- Safe checkpoint saved: `false`

## Start Condition Trace

| epoch | action | checked | start_condition | reasons | map50_range | recall_range | eligible_classes | strong_baseline |
|---:|---|---|---|---|---:|---:|---|---|
| 5 | burnin_observe | true | false | `['metric_unstable', 'strong_clean_baseline_protection']` | 0.0874 | 0.1067 | `[4, 8, 12]` | `true` |

## Next Step

- Enter the seed2 50ep adaptive-RB run only if the smoke remains strict no-op before candidate start, bbox/class validation is clean, and CATF router random draws remain zero while burn-in is observing.
