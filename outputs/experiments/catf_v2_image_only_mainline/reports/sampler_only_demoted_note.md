# Sampler-Only Demoted Note

`sampler_only` has been implemented and verified through the weighted index-list dataloader path.

Observed sampler-only multiseed status:

- seed0 showed positive deltas versus its paper clean baseline.
- seed2 showed positive deltas versus its paper clean baseline.
- seed1 failed the requested Precision, mAP50, and mAP50-95 constraints.
- The dataloader intervention was effective: weighted index lists changed the sampled train_core distribution.

Method positioning:

- `sampler_only` is a training sampling intervention.
- It is closer to hard example mining / weighted sampling than image data augmentation.
- It changes which original samples appear more often during training.
- It does not synthesize or transform image content.

Decision:

- `sampler_only` is not a CP-CATF paper main method.
- `sampler_only` results are kept only as engineering exploration and possible ablation evidence.
- Unless explicitly requested as an ablation, do not continue pushing sampler-only as the main CP-CATF direction.
- Future paper main results must come from image augmentation: accepted, weakened, or rejected image-space policies.
