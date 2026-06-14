# Seed1 Image-Only Weak Augmentation Sanity

## Scope

- Run root: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_image_only_weak_aug_multiseed\seed1`
- Seed: `1`
- Method: image-only weak ROI texture attenuation
- Attenuation ratio: `0.25`
- Sampler-only: `disabled`
- Weighted index list: `disabled`
- Sampled distribution changed: `false`

## Execution

- 50ep completed: `true`
- Weak image augmentation executed: `true`
- Industrial images augmented: `32`
- ROI applied: `38`
- Router random draw count: `775`
- Invalid bbox count: `0`
- BBox out-of-bounds count: `0`
- Class-id out-of-bounds count: `0`
- Final val used for policy selection: `false`

## Metrics

| baseline | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.772525 | 0.647680 | 0.754191 | 0.479919 |
| fixed CATF-v2 | 0.785199 | 0.700509 | 0.782618 | 0.518879 |
| weak image aug | 0.765605 | 0.712074 | 0.776844 | 0.506815 |

## Deltas

| comparison | dP | dR | d mAP50 | d mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| weak - clean | -0.006920 | 0.064394 | 0.022653 | 0.026896 |
| weak - fixed | -0.019594 | 0.011564 | -0.005773 | -0.012064 |

## Constraint

- Constraint failed: `false`
- Failure reasons: `none`
- Recall warning: `false`
- Fixed CATF-v2 benefit retained: `false`
- New non-active regression vs fixed: `false`

## Candidate Decisions

| epoch | decision | class | retained op | weak prob | weak strength | gates |
| ---: | --- | ---: | --- | ---: | ---: | --- |
| 25 | weak_roi_texture | 6 | local_contrast | 0.045000 | 0.050000 | precision=true, non_active=true |
| 40 | weak_roi_texture | 9 | local_contrast | 0.045000 | 0.050000 | precision=true, non_active=true |

## Augmentation Counts

- Weak interval counts: `{"6:25": 16, "9:40": 16}`
- ROI affected classes: `{"6": 18, "9": 20}`
- Strict no-op count: `7`
- Sampler-only effective: `false`
- Sample weight map generated: `false`
- Weighted train-core images: `0`

## Non-Active Regression

- Mean AP50-95 delta vs fixed: `0.533254`
- Mean AP50-95 delta vs clean: `0.031980`
- Regressed vs fixed count (>0.01): `0`
- Regressed vs clean count (>0.01): `2`

## Conclusion

This seed passes constraints and improves over clean, but it does not fully retain the fixed CATF-v2 mAP50-95 gain within a 0.01 tolerance.
