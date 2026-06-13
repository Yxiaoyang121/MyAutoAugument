# Image-Only CATF-v2 Mainline Summary

## Position

The paper mainline is restored to image augmentation based CATF. `sampler_only` is not a paper main method because it is a training sampling intervention, not an image data augmentation method.

Current image-only mainline baseline:

- fixed CATF-v2 is the current image augmentation mainline.
- CP-CATF remains relevant only as an image-only causal-probe controller: image candidates may be accepted, weakened, or blocked, but not converted into sampling reweighting for the main result.
- Future main results must come from image augmentation behavior, not weighted sampling or hard-example mining.

## Fixed CATF-v2 Results

| seed | clean P | clean R | clean mAP50 | clean mAP50-95 | fixed P | fixed R | fixed mAP50 | fixed mAP50-95 | dP | dR | dM50 | dM95 | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | -0.0061 | -0.0068 | +0.0090 | +0.0136 | false |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | +0.0127 | +0.0528 | +0.0284 | +0.0390 | false |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | +0.0675 | -0.0423 | -0.0110 | -0.0257 | true |
| mean | 0.7511 | 0.6843 | 0.7527 | 0.4927 | 0.7758 | 0.6855 | 0.7615 | 0.5017 | +0.0247 | +0.0012 | +0.0088 | +0.0090 | n/a |

Constraint pass count: `2/3`.

## Interpretation

Seed0 and seed1 show that CATF image augmentation has real potential:

- seed0 improves mAP50 and mAP50-95 while staying within the Precision constraint.
- seed1 improves all four headline metrics.

Seed2 remains the blocking case:

- fixed CATF-v2 improves Precision but reduces Recall, mAP50, and mAP50-95.
- The failure should be described as high-risk image augmentation causing non-active regression and unsafe transfer across classes.
- Earlier audits localized the risk to image-space texture/ROI intervention, including class-9 texture-path risk, rather than to the absence of sampler-only weighting.

## Seed2 Repair Direction

Seed2 must be fixed inside the image augmentation mainline:

- causal probe before applying image augmentation;
- weak image augmentation / attenuation when an image candidate has partial benefit but elevated risk;
- strict image no-op when risk is too high;
- explicit non-active regression constraints.

Do not use `sampler_only`, weighted index lists, or hard-example mining to repair seed2 for the paper main result.

## Sampler-Only Status

`sampler_only` has been implemented and verified, but it is demoted to engineering exploration / ablation only. It changes the training sample distribution and is not equivalent to image data augmentation. It must not be used as the paper's main CP-CATF result.
