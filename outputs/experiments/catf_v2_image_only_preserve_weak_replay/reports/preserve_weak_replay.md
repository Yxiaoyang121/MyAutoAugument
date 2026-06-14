# Preserve-Original + Weak-Only Image CATF Replay

Scope: offline replay only. No training was run. Sampler-only and weighted index list are not used.

## Decision Rule

1. Preserve the original fixed CATF-v2 image policy when the fixed seed already passed constraints and an executable conservative fixed image policy exists.
2. Use weak ROI texture only when the original fixed path is not preservable and attenuation 0.25 passes the replay secondary gate.
3. Use strict no-op for high or critical image risk, including no image evidence, failed attenuation gate, or critical FP/spillover risk.
4. Do not use sampler_only.

Note: the fixed CATF-v2 runs predate candidate-level causal risk metrics for the preserved original policies. For preserve decisions, low-risk status is therefore based on seed-level fixed constraint pass plus the surviving conservative executable image policy. Causal/precision/non-active replay metrics are applied to candidates that would otherwise replace or repair the fixed path.

## Replay Counts

| seed | preserve_original | weak_roi_texture | strict_noop | fixed classes | weak classes |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 9 | 0 | 0 | [4, 11, 12] | [] |
| 1 | 9 | 0 | 0 | [4, 11] | [] |
| 2 | 0 | 5 | 4 | [8, 9, 11] | [6, 7, 9] |

## Expected Behavior Checks

- Seed0 preserves fixed class 4/11/12 strategy: `True`.
- Seed0 avoids weak class9 replacement: `True`.
- Seed1 preserves fixed gain policy: `True`.
- Seed2 converts failed fixed policy to weak/no-op: `True`.
- Seed2 contains previous weak-safe candidates: `True`.
- Sampler-only absent: `True`.

## Seed-Level Interpretation

- Seed0: all 9 feedback epochs preserve the fixed original image policy. The replay keeps the fixed classes 4/11/12 and blocks the epoch25 weak class9 replacement that caused the seed0 weak failure.
- Seed1: all 9 feedback epochs preserve the fixed original image policy. This keeps the fixed seed1 gain path instead of globally switching to weak.
- Seed2: no epoch preserves the failed fixed image policy. Five epochs become weak ROI texture and four become strict no-op, matching the earlier weak-safe seed2 replay pattern.

## Answers

- Seed0 preserve/weak/noop counts: `9/0/0`.
- Seed1 preserve/weak/noop counts: `9/0/0`.
- Seed2 preserve/weak/noop counts: `0/5/4`.
- Seed0 keeps fixed original class4/11/12 policy: yes.
- Seed0 avoids weak class9 replacement: yes.
- Seed1 keeps fixed original benefit path: yes.
- Seed2 high-risk fixed path is converted to weak/no-op: yes.
- Sampler-only participation: none.
- Recommended first training validation: seed0 sanity first, because seed0 is exactly where global weak replacement failed.
- Three-seed training validation: not yet. Run seed0 sanity, then seed2, then seed1 if both pass.

## Output

- Decision records: `outputs\experiments\catf_v2_image_only_preserve_weak_replay\preserve_weak_decision_records.csv`
- JSON report: `outputs\experiments\catf_v2_image_only_preserve_weak_replay\reports\preserve_weak_replay.json`
- Training plan: `outputs\experiments\catf_v2_image_only_preserve_weak_replay\reports\preserve_weak_training_plan.md`
