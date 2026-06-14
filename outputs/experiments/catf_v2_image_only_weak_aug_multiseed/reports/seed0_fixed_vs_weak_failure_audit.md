# Seed0 Fixed vs Weak Image Augmentation Failure Audit

Scope: offline audit only. No training was run. Sampler-only and weighted index list paths are not involved.

## Aggregate Metrics

| run | P | R | mAP50 | mAP50-95 | constraint |
| --- | ---: | ---: | ---: | ---: | --- |
| clean seed0 | 0.784554 | 0.676457 | 0.734696 | 0.475894 | baseline |
| fixed CATF-v2 seed0 | 0.778506 | 0.669654 | 0.743657 | 0.489541 | pass |
| weak image aug seed0 | 0.679043 | 0.711821 | 0.722170 | 0.476005 | fail |

Delta vs clean:
- fixed: P -0.006048, R -0.006803, mAP50 0.008962, mAP50-95 0.013647.
- weak: P -0.105511, R 0.035364, mAP50 -0.012525, mAP50-95 0.000111.

## Execution Difference

- Fixed seed0 passed with conservative image augmentation on classes 4, 11, and 12: 41 industrial images augmented, 45 ROI applications.
- Weak seed0 executed image augmentation only at epoch25 on class 9: 16 industrial images augmented and 18 ROI applications.
- Weak seed0 did not use sampler-only: sampler_only_effective=false, weighted_index_list_enabled=false, sampled_distribution_changed=false.
- The weak epoch25 candidate kept only local_contrast at prob=0.045 and strength=0.05, but it replaced the fixed path's active safe policies instead of preserving them.

## Epoch-Level Policy Diff

| epoch | fixed candidate / ops | weak candidate / ops | fixed policy replaced | weak new policy | weak applied | no-op reason |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 5 | fixed_catf_v2_roi_texture c4:油污:low_recall:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c11:锡尖:texture_boundary_weak:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02 | candidate_policy_0_noop | True | False | 0 img / 0 ROI | no_prior_image_causal_evidence |
| 10 | fixed_catf_v2_rollback_policy c4:油污:low_recall:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01 | candidate_policy_0_noop | True | False | 0 img / 0 ROI | no_prior_image_causal_evidence |
| 15 | fixed_catf_v2_roi_texture c4:油污:low_recall:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:锡膏:weak_localization:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02 | candidate_policy_0_noop | True | False | 0 img / 0 ROI | no_prior_image_causal_evidence |
| 20 | fixed_catf_v2_shrink_policy c4:油污:low_recall:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:锡膏:weak_localization:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02 | candidate_policy_0_noop | True | False | 0 img / 0 ROI | no_prior_image_causal_evidence |
| 25 | fixed_catf_v2_shrink_policy c4:油污:low_recall:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:锡膏:weak_localization:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02 | candidate_policy_1b_weak_roi_texture c9:轮廓划伤:causal_probe_accepted:local_contrast@p=0.045/s=0.05 | True | True | 16 img / 18 ROI |  |
| 30 | fixed_catf_v2_observe_policy c4:油污:low_recall:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:锡膏:weak_localization:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02 | candidate_policy_0_noop | True | False | 0 img / 0 ROI | no_prior_image_causal_evidence |
| 35 | fixed_catf_v2_observe_policy c4:油污:low_recall:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:锡膏:weak_localization:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02 | candidate_policy_0_noop | True | False | 0 img / 0 ROI | no_prior_image_causal_evidence |
| 40 | fixed_catf_v2_freeze_policy c4:油污:low_recall:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:锡膏:weak_localization:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02 | candidate_policy_0_noop | True | False | 0 img / 0 ROI | no_prior_image_causal_evidence |
| 45 | fixed_catf_v2_freeze_policy c4:油污:low_recall:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:锡膏:weak_localization:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02 | candidate_policy_0_noop | True | False | 0 img / 0 ROI | no_prior_image_causal_evidence |

## Per-Class Regression

The TP/FP/FN columns in the CSV are metric-derived estimates from precision, recall, and instance count; the run artifacts do not expose raw prediction-level TP/FP/FN tables.

Largest weak precision drops vs clean:

| class | name | delta P | delta FP vs fixed | delta AP50 | delta AP50-95 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 5 | 浅划伤 | -0.449913 | 10.259378 | -0.097174 | -0.030836 |
| 4 | 油污 | -0.251566 | -0.991920 | -0.066599 | -0.036138 |
| 3 | 开裂 | -0.227702 | 2.474973 | -0.165833 | -0.103515 |
| 11 | 锡尖 | -0.198785 | 1.631544 | 0.002558 | 0.000636 |
| 9 | 轮廓划伤 | -0.082185 | 11.627821 | 0.002719 | -0.000140 |

Largest weak FP-estimate increases vs fixed:

| class | name | delta FP vs fixed | delta P vs clean | delta AP50 | delta AP50-95 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | 碰伤 | 32.357575 | -0.046465 | -0.010474 | 0.006054 |
| 9 | 轮廓划伤 | 11.627821 | -0.082185 | 0.002719 | -0.000140 |
| 5 | 浅划伤 | 10.259378 | -0.449913 | -0.097174 | -0.030836 |
| 12 | 锡膏 | 6.005689 | -0.036408 | -0.026161 | -0.001604 |
| 6 | 漏背锡 | 4.998724 | -0.053762 | 0.020632 | 0.006559 |

Largest AP50 drops vs clean:

| class | name | delta AP50 | delta P vs clean | delta FP vs fixed |
| ---: | --- | ---: | ---: | ---: |
| 3 | 开裂 | -0.165833 | -0.227702 | 2.474973 |
| 5 | 浅划伤 | -0.097174 | -0.449913 | 10.259378 |
| 4 | 油污 | -0.066599 | -0.251566 | -0.991920 |
| 12 | 锡膏 | -0.026161 | -0.036408 | 6.005689 |
| 7 | 碰伤 | -0.010474 | -0.046465 | 32.357575 |

Largest AP50-95 drops vs clean:

| class | name | delta AP50-95 | delta P vs clean | delta FP vs fixed |
| ---: | --- | ---: | ---: | ---: |
| 3 | 开裂 | -0.103515 | -0.227702 | 2.474973 |
| 4 | 油污 | -0.036138 | -0.251566 | -0.991920 |
| 5 | 浅划伤 | -0.030836 | -0.449913 | 10.259378 |
| 2 | 加强筋打伤 | -0.008395 | -0.006298 | 0.515325 |
| 12 | 锡膏 | -0.001604 | -0.036408 | 6.005689 |

## Interpretation

1. Fixed seed0 passed because it kept very conservative ROI texture augmentation on classes 4, 11, and 12. The probabilities and strengths stayed small, and the run still improved mAP50 and mAP50-95 relative to clean.
2. Weak seed0 failed because the weak replay path globally replaced the fixed policy behavior. It suppressed the fixed seed0 safe policies and then executed a different class9 weak local_contrast candidate at epoch25.
3. The precision collapse is FP-driven. Weak seed0 precision dropped by 0.105511 vs clean while recall increased by 0.035364, consistent with recall gain bought by broad FP spillover.
4. The FP increase is not isolated to class9. The largest FP-estimate increase vs fixed is class7, and several non-active classes also regress, so this is non-active regression rather than only a target-class tradeoff.
5. The epoch25 weak candidate was allowed after attenuation=0.25, but its original replay metadata still carried precision and non-active risk flags. In the full run, attenuation was not sufficient for seed0.
6. Direct prediction-level high-confidence FP histograms are not present in the final metrics artifacts. However, the replay row for the original epoch25 candidate reported high_confidence_fp_delta=0.05, and the final per-class metrics show broad FP-estimate growth.

## Required Answers

- Seed0 weak failure main cause: global weak replacement did not preserve fixed seed0's safe original image policies and introduced a class9 weak local_contrast policy that produced broad FP/non-active regression.
- Precision drop source: largest weak precision drops vs clean are class5, class4, class3, class11, class9, and class7; largest FP-estimate increases vs fixed are led by class7, class9, class5, class12, and class6.
- Did weak break fixed strategy: yes. Fixed's effective classes were 4/11/12; weak's only executed class was 9 at epoch25.
- Non-active regression: yes. Most FP growth appears in classes not targeted by weak augmentation.
- Should seed0 preserve fixed original policy: yes. Seed0 should keep fixed original policy when probe risk is low.
- Recommended next strategy: implement preserve-safe-original plus weak-only-for-moderate-risk and strict no-op for high or critical risk.
- Continue training now: no. First replay/validate the preserve-safe-original decision logic offline.
- Image-only mainline: yes. This audit does not use sampler-only and does not recommend reintroducing it.

## Output Tables

- Epoch policy diff: `outputs\experiments\catf_v2_image_only_weak_aug_multiseed\seed0_fixed_vs_weak_epoch_policy_diff.csv`
- Per-class regression: `outputs\experiments\catf_v2_image_only_weak_aug_multiseed\seed0_fixed_vs_weak_per_class_regression.csv`
