# Limitations And Next Step

## Current Limitation

Preserve-weak image-only CATF reaches `3/3` pass under the current hard constraints, but seed2 Recall remains below clean:

- seed2 clean Recall: `0.728600`;
- seed2 preserve-weak Recall: `0.694235`;
- delta: `-0.034365`.

This does not trigger `constraint_failed` because the current hard constraints protect:

- Precision drop greater than `0.01`;
- mAP50 drop greater than `0.01`;
- mAP50-95 drop greater than `0.01`.

Recall is treated as a warning rather than a hard failure.

## Why This Matters

Industrial detection often values Recall strongly. A method that improves Precision and mAP while allowing a noticeable Recall drop may still be operationally undesirable in some production settings. Therefore, the seed2 Recall warning should be reported clearly rather than hidden behind the hard pass count.

## Recommended Next Step

The next improvement should remain inside image augmentation. It should not use sampler-only or weighted sampling to repair Recall.

Possible image-only extensions:

- recall-aware gate: reject or attenuate image augmentation when probe evidence predicts a large Recall drop;
- FN-focused weak image augmentation: prioritize weak augmentations that reduce false negatives without increasing non-active false positives;
- class-specific but non-hardcoded recall protection: use diagnostic risk signals rather than fixed seed or class IDs;
- more conservative no-op for recall-risk candidates: if a candidate has Precision/mAP benefit but unacceptable Recall risk, prefer strict no-op.

## Explicit Non-Goal

Do not use sampler-only to repair the Recall warning in the main method. Sampler-only changes the training sample distribution and is better treated as a separate hard-example mining ablation, not as automatic image data augmentation.
