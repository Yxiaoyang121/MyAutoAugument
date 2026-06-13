# Seed2 Image-Only Weak Augmentation Validation

## Scope

- Run: `catf_v2_image_only_weak_aug_seed2_50ep`
- Seed: `2`
- Method: image-only CP-CATF weak ROI texture attenuation
- Attenuation ratio: `0.25`
- Sampler-only: `disabled`
- Weighted index list: `disabled`
- Sampled distribution changed: `false`
- Final-val policy-selection flag in weak decisions: `false`

## Execution

- 50ep completed: `true`
- Weak image augmentation executed: `true`
- Industrial images augmented: `80`
- ROI applied: `95`
- Router random draw count: `1922`
- Weak interval counts: `{"7:20": 16, "9:25": 16, "6:30": 16, "9:35": 16, "9:45": 16}`
- sample_weight_map generated: `false`
- weighted_train_core_images_count: `0`

## Metrics

| run | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 |
| fixed CATF-v2 seed2 | 0.7637 | 0.6863 | 0.7582 | 0.4967 |
| weak image aug seed2 | 0.7533 | 0.6942 | 0.7727 | 0.5151 |

## Delta

- vs clean seed2: P `+0.0570`, R `-0.0344`, mAP50 `+0.0035`, mAP50-95 `-0.0072`
- vs fixed CATF-v2 seed2: P `-0.0104`, R `+0.0079`, mAP50 `+0.0145`, mAP50-95 `+0.0185`
- constraint_failed: `false`

## Weak Candidates

| epoch | op | original prob | original strength | weak prob | weak strength | cap | router |
|---:|---|---:|---:|---:|---:|---:|---|
| 20 | `local_contrast` | 0.1800 | 0.2000 | 0.0450 | 0.0500 | 16 | `true` |
| 25 | `local_contrast` | 0.1800 | 0.2000 | 0.0450 | 0.0500 | 16 | `true` |
| 30 | `local_contrast` | 0.1800 | 0.2000 | 0.0450 | 0.0500 | 16 | `true` |
| 35 | `local_contrast` | 0.1800 | 0.2000 | 0.0450 | 0.0500 | 16 | `true` |
| 45 | `local_contrast` | 0.1800 | 0.2000 | 0.0450 | 0.0500 | 16 | `true` |

## Class9 And Non-Active Regression

- Class9 recovered vs fixed: `true`
- Class9 fully recovered vs clean: `false`
- Class9 delta vs fixed: recall `+0.1086`, AP50 `+0.0342`, AP50-95 `+0.0099`
- Class9 delta vs clean: recall `-0.0581`, AP50 `-0.0124`, AP50-95 `-0.0556`
- Non-active AP50-95 improved vs fixed count: `7/10`
- Mean non-active AP50-95 delta vs fixed: `+0.0187`
- Mean non-active AP50-95 delta vs clean: `-0.0094`

## Conclusion

- Seed2 weak image augmentation passes constraints: `true`
- It is a viable image-only CP-CATF repair direction for seed2: `true`
- Recommend seed0/seed1 sanity next: `true`
