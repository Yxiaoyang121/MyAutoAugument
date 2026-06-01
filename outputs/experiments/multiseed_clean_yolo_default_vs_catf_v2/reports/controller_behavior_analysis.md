# CATF-v2 Controller Behavior Analysis

Generated: `2026-06-01T13:08:26`

- Aggregate policy actions: `{'shrink': 14, 'accept': 2, 'observe': 5, 'freeze': 6}`.
- Aggregate class actions: `{'propose': 2, 'observe': 8}`.
- Rollback/cooldown/freeze policy-action totals: `0/0/6`.
- Freeze markers including `frozen=true` flags: `6`.

## Findings

- `shrink` dominates (`14`), while `accept` is rare (`2`). This means CATF-v2 is mostly avoiding harm, not reliably discovering helpful policies.
- Rollback/cooldown are zero because the current failure modes are not tied to a pending class policy in the controller state; they are global trajectory and non-active-class effects.
- Freeze at epoch 40 is reasonable for convergence, but when very few policies were accepted beforehand it also prevents late recovery.
- Seed 1 had zero industrial augmentation but still diverged from clean, which suggests in-loop diagnosis/callback RNG effects need a dedicated control.
- The next controller change should be negative-effect attribution: if a class or metric worsens after an update, rollback that class policy or reduce all risky policies that correlate with the degradation.
