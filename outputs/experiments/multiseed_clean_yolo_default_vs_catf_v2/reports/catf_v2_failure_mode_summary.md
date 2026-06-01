# CATF-v2 Failure Mode Summary

Generated: `2026-06-01T13:08:26`

## Main Failure Modes

- `intervention too weak`
- `intervention wrong target`
- `precision threshold issue`
- `over-conservative freeze`
- `per-class diagnosis trajectory sensitive`
- `in-loop diagnosis/callback side effect not isolated`

## Seed-Level Conclusions

- Seed 0: Precision threshold / FP control issue. Recall and AP improve, but FP increases across several classes, not only the active class.
- Seed 1: Conservative/no-op behavior. No ROI or industrial augmentation was applied, so the pass is not evidence that CATF-v2 augmentation helped; it may include in-loop diagnosis/callback RNG effects.
- Seed 2: Conservative confidence/recall collapse on a clean seed that already had high Recall. CATF-v2 did not target the final degraded classes.

## Mechanism Diagnosis

- CATF-v2 fixed the OK3 activation problem: OK2/OK3 were never active and OK3 ROI remained 0.
- The remaining instability is not mainly bad OK-class activation. It is sparse intervention plus unstable global confidence/threshold behavior.
- ROI-aware augmentation is too sparse to prove causal benefit: only 23 ROI applications total across seeds 0/1/2.
- Controller behavior is over-conservative: many shrink/freeze actions, no rollback/cooldown, and only two accepted/proposed active class policies.
- Seed 1 passes with zero industrial augmentation, so CATF-v2 gains are not yet causally attributable to ROI-aware augmentation.
- Some failures happen in non-active classes, so the controller lacks negative-effect attribution.
- No confusion/PR curve artifacts were found under the multiseed root; prediction JSON files were available for feedback-epoch diagnosis.

## Next Step

- Do not run more 50 epoch training before evaluating threshold calibration as a post-processing/validation layer.
- Add a diagnosis-only in-loop control that enables feedback callbacks/diagnosis but keeps industrial augmentation probabilities at zero, to isolate RNG/callback side effects.
- Next controller work should add per-class negative-effect attribution and rollback, not just more activation tuning.
- CATF-v2 should be positioned as a promising ablation/controller variant rather than the paper main method until it passes multiseed constraints.
