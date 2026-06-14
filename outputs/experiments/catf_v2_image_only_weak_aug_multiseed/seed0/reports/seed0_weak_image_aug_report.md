# Seed0 Image-Only Weak Augmentation Sanity

## Scope

- Run root: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_image_only_weak_aug_multiseed\seed0`
- Seed: `0`
- Method: image-only weak ROI texture attenuation
- Attenuation ratio: `0.25`
- Sampler-only: `disabled`
- Weighted index list: `disabled`
- Sampled distribution changed: `false`

## Execution

- 50ep completed: `true`
- Weak image augmentation executed: `true`
- Industrial images augmented: `16`
- ROI applied: `18`
- Router random draw count: `341`
- Invalid bbox count: `0`
- BBox out-of-bounds count: `0`
- Class-id out-of-bounds count: `0`
- Final val used for policy selection: `false`

## Metrics

| baseline | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.784554 | 0.676457 | 0.734696 | 0.475894 |
| fixed CATF-v2 | 0.778506 | 0.669654 | 0.743657 | 0.489541 |
| weak image aug | 0.679043 | 0.711821 | 0.722170 | 0.476005 |

## Deltas

| comparison | dP | dR | d mAP50 | d mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| weak - clean | -0.105511 | 0.035364 | -0.012525 | 0.000111 |
| weak - fixed | -0.099463 | 0.042167 | -0.021487 | -0.013536 |

## Constraint

- Constraint failed: `true`
- Failure reasons: `precision_drop_gt_0.01, map50_drop_gt_0.01`
- Recall warning: `false`
- Fixed CATF-v2 benefit retained: `false`
- New non-active regression vs fixed: `false`

## Candidate Decisions

| epoch | decision | class | retained op | weak prob | weak strength | gates |
| ---: | --- | ---: | --- | ---: | ---: | --- |
| 25 | weak_roi_texture | 9 | local_contrast | 0.045000 | 0.050000 | precision=true, non_active=true |

## Augmentation Counts

- Weak interval counts: `{"9:25": 16}`
- ROI affected classes: `{"9": 18}`
- Strict no-op count: `8`
- Sampler-only effective: `false`
- Sample weight map generated: `false`
- Weighted train-core images: `0`

## Non-Active Regression

- Mean AP50-95 delta vs fixed: `0.482786`
- Mean AP50-95 delta vs clean: `0.000131`
- Regressed vs fixed count (>0.01): `0`
- Regressed vs clean count (>0.01): `3`

## Conclusion

This seed does not pass the current hard constraints. The weak image augmentation path executed, but this result should not be treated as retaining the fixed CATF-v2 benefit for this seed.
