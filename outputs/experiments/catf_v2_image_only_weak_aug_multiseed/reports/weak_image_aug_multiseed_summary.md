# Image-Only Weak Augmentation Multiseed Summary

## Scope

- Method: image-only weak CP-CATF / weak ROI texture attenuation
- Attenuation ratio: `0.25`
- Seeds: `0, 1, 2`
- Seed0/seed1 were run in `catf_v2_image_only_weak_aug_multiseed`.
- Seed2 reuses the completed `catf_v2_image_only_weak_aug_seed2_50ep` result.
- Sampler-only and weighted index list are not part of this mainline.

## Per-Seed Metrics

| seed | method | P | R | mAP50 | mAP50-95 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 clean | 0.784554 | 0.676457 | 0.734696 | 0.475894 |
| 0 fixed CATF-v2 | 0.778506 | 0.669654 | 0.743657 | 0.489541 |
| 0 weak image aug | 0.679043 | 0.711821 | 0.722170 | 0.476005 |
| 1 clean | 0.772525 | 0.647680 | 0.754191 | 0.479919 |
| 1 fixed CATF-v2 | 0.785199 | 0.700509 | 0.782618 | 0.518879 |
| 1 weak image aug | 0.765605 | 0.712074 | 0.776844 | 0.506815 |
| 2 clean | 0.696239 | 0.728630 | 0.769203 | 0.522371 |
| 2 fixed CATF-v2 | 0.763658 | 0.686345 | 0.758246 | 0.496684 |
| 2 weak image aug | 0.753254 | 0.694235 | 0.772718 | 0.515138 |

## Per-Seed Deltas

| seed | comparison | dP | dR | d mAP50 | d mAP50-95 | constraint_failed | recall_warning |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | weak - clean | -0.105511 | +0.035364 | -0.012525 | +0.000111 | true | false |
| 0 | weak - fixed | -0.099463 | +0.042167 | -0.021487 | -0.013536 | true | false |
| 1 | weak - clean | -0.006920 | +0.064394 | +0.022653 | +0.026896 | false | false |
| 1 | weak - fixed | -0.019594 | +0.011564 | -0.005773 | -0.012064 | false | false |
| 2 | weak - clean | +0.057015 | -0.034395 | +0.003515 | -0.007233 | false | true |
| 2 | weak - fixed | -0.010404 | +0.007890 | +0.014472 | +0.018454 | false | true |

## Means

| item | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| mean clean | 0.751106 | 0.684255 | 0.752696 | 0.492728 |
| mean fixed CATF-v2 | 0.775788 | 0.685503 | 0.761507 | 0.501701 |
| mean weak image aug | 0.732634 | 0.706043 | 0.757244 | 0.499319 |
| mean weak - clean | -0.018472 | 0.021788 | 0.004548 | 0.006591 |
| mean weak - fixed | -0.043154 | 0.020541 | -0.004263 | -0.002382 |

## Coverage

- Constraint pass count: `2/3`
- 3/3 pass: `false`
- Industrial images augmented total: `128`
- ROI applied total: `151`
- Sampler-only involved: `false`
- Weighted index list involved: `false`
- Sampled distribution changed any seed: `false`
- Final val used for policy selection any seed: `false`
- Recall warning any seed: `true`
- Main method candidate: `false`
- Recommend recall-aware constraint: `true`

## Interpretation

- Seed2 was repaired by image-only weak augmentation without sampler-only or sampling changes.
- Seed1 remains constraint-pass and keeps positive mAP gains vs clean, but does not fully retain fixed CATF-v2 mAP50-95.
- Seed0 fails the current hard constraints because precision and mAP50 drop below clean by more than 0.01.
- Therefore this exact weak augmentation setting is not yet a 3-seed paper main result.
- Seed2 recall remains below clean and should be reported as a recall warning; a recall-aware constraint is the next image-only safety direction.
