# Top3 Policy Short-Training Report

## Scope

- Top3 policies come from the same baseline diagnosis.
- This is not multi-round closed-loop optimization.
- These are proxy/safety top3 candidates.
- Each policy was trained for 5 epochs only.
- This is not a final model result; it validates strategy selection and informs whether to rerun formal 50 epoch DiagAug.

## Comparison

| policy_id | source_issue | operations | copy_paste | proxy_rank | Precision | Recall | mAP50 | mAP50-95 | short_train_score |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| diag_policy_001 | low_contrast_missed_defect | clahe(p=0.411, s=0.382), contrast(p=0.407, s=0.372), gamma(p=0.324, s=0.311), brightness(p=0.229, s=0.241) | False | 1 | 0.709 | 0.672 | 0.714 | 0.479 | 0.609204 |
| diag_policy_005 | combined | copy_paste(p=0.717, s=0.696), clahe(p=0.459, s=0.422), contrast(p=0.464, s=0.422), gamma(p=0.447, s=0.402), scale(p=0.344, s=0.251), gaussian_noise(p=0.167, s=0.120) | True | 2 | 0.714 | 0.583 | 0.683 | 0.453 | 0.578862 |
| diag_policy_002 | low_contrast_missed_defect | clahe(p=0.391, s=0.382), contrast(p=0.423, s=0.372), gamma(p=0.321, s=0.311), brightness(p=0.232, s=0.241) | False | 3 | 0.726 | 0.613 | 0.710 | 0.460 | 0.595412 |

## Selection

- Short-training best policy: `diag_policy_001`
- Matches current formal DiagAug policy `diag_policy_001`: `true`
- Score formula used: `0.30*mAP50 + 0.35*mAP50-95 + 0.20*Recall + 0.15*small_object_recall`

## copy_paste Policy

- Policy: `diag_policy_005`
- Beats best low-contrast policy: `false`
- Bbox safety issue: `false`
- Worth formal 50 epoch: `false`

## Recommendation

Short-training top1 matches the current formal DiagAug policy. A rerun is not strictly required for strategy selection, but can be used for confirmation.
