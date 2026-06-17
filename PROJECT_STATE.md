# Project State

Last updated: 2026-06-16

## Current Position

The repository is centered on diagnosis-driven augmentation for industrial defect detection. The active path still keeps YOLO network architecture unchanged and focuses on dataset construction, validation-error diagnosis, policy generation, proxy safety, short training, and auditable reporting.

## Preserve-Original Execution Parity Fix (2026-06-16)

- Scope: audit, dry-run, and code/test fix only. No seed0/seed1/seed2 50ep training, multiseed run, sampler_only run, weighted sampling, data split change, gate change, or attenuation change was performed.
- The seed0 preserve-weak sanity failure was traced to an execution parity bug, not to the three-stage image-only decision policy itself.
- Root cause: `preserve_original` recorded the replay decision but returned the current runtime controller policy via `deepcopy(policy)`; it did not install the replayed fixed CATF-v2 class/op policy into the executable `policy_matrix`.
- Failure symptom:
  - replay expected seed0 preserve classes `4/11/12`;
  - failed runtime executed only class `11`;
  - fixed seed0 had ROI affected classes `{4:9, 11:22, 12:14}`, while preserve sanity had `{11:20}`.
- Fix:
  - `preserve_original` now parses `original_fixed_active_class`, `original_fixed_op_list`, and `original_fixed_prob_strength`;
  - it overlays the fixed class list, op list, op probability, and op strength into the runtime policy matrix;
  - it clears stale active ops from non-preserved classes, preventing weak class9 replacement;
  - it bypasses weak attenuation and sampler-only paths.
- Dry-run after the fix:
  - expected class union=`[4, 11, 12]`;
  - runtime policy_matrix class union=`[4, 11, 12]`;
  - sample_router eligible class union=`[4, 11, 12]`;
  - final executable class union=`[4, 11, 12]`;
  - weak class9 replacement=false;
  - sampler_only=false; weighted_index_list=false.
- Reports:
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity/reports/preserve_original_execution_audit.md`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity/reports/preserve_original_execution_audit.json`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity/reports/preserve_execution_dryrun.md`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity/reports/preserve_execution_dryrun.json`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity/preserve_execution_parity_epoch_diff.csv`
- Next step: rerun seed0 preserve-weak sanity first. Do not proceed to seed2 until seed0 confirms execution parity and avoids the prior Precision collapse.

## Seed0 Preserve-Weak Sanity Rerun After Parity Fix (2026-06-17)

- Scope: ran only seed0 50ep. Seed1, seed2, multiseed, sampler_only, weighted index list, sampling changes, data-split changes, gate changes, causal-score changes, and attenuation-ratio changes were not used.
- Output:
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/reports/seed0_preserve_weak_sanity_rerun_report.md`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/reports/seed0_preserve_weak_sanity_rerun_report.json`
- Execution:
  - completed_50ep=true;
  - preserve_original=9, weak_roi_texture=0, strict_noop=0;
  - expected class union=`[4, 11, 12]`;
  - runtime policy_matrix union=`[4, 11, 12]`;
  - sample_router eligible union=`[4, 11, 12]`;
  - final executable union=`[4, 11, 12]`;
  - weak class9 replacement=false;
  - sampler_only=false; weighted_index_list=false; sampled_distribution_changed=false.
- Augmentation:
  - industrial images augmented=155;
  - ROI applied=187;
  - ROI affected classes=`{4:66, 11:22, 12:99}`;
  - fixed CATF-v2 seed0 reference was industrial=41, ROI=45, affected=`{4:9, 11:22, 12:14}`.
- Metrics:
  - rerun seed0: P=0.700514, R=0.623545, mAP50=0.712481, mAP50-95=0.456610;
  - vs clean seed0: dP=-0.084086, dR=-0.052955, dM50=-0.022219, dM95=-0.019290;
  - vs fixed CATF-v2 seed0: dP=-0.077986, dR=-0.046155, dM50=-0.031219, dM95=-0.032890;
  - constraint_failed=true.
- Interpretation:
  - the original class4/class12 missing bug is fixed;
  - seed0 still fails because preserve now over-applies the replay policy compared with fixed CATF-v2;
  - this is no longer weak class9 replacement and not sampler_only, but a remaining preserve policy lifetime / application-volume parity mismatch;
  - do not run seed2 yet. Next step should audit fixed-vs-preserve policy lifetime and applied augmentation volume.

## Preserve-Original Volume / Lifetime Parity Fix (2026-06-17)

- Scope: audit, dry-run, code/test fix, and report generation only. No seed0/seed1/seed2 50ep training, multiseed run, sampler_only run, weighted sampling, data split change, gate change, causal-score change, or attenuation-ratio change was performed.
- Root cause:
  - the preserve schedule used cumulative replay active classes after epoch15 instead of epoch-exact fixed CATF-v2 router-executable policy rows;
  - fixed CATF-v2 retained some nonzero old ops in guarded/frozen class policies, but those rows were not router-executable;
  - preserve cleared guards and treated those historical rows as active, so class4 stayed active for 9 feedback epochs and class12 for 7 feedback epochs.
- Volume mismatch before fix:
  - fixed seed0: industrial=41, ROI=45;
  - preserve rerun seed0: industrial=155, ROI=187;
  - class4 ROI ratio=66/9=7.33x;
  - class12 ROI ratio=99/14=7.07x;
  - op probabilities and strengths were not numerically amplified; the policy lifetime/router eligibility was too long.
- Fix:
  - `scripts/build_preserve_weak_decision_schedule.py` now loads fixed CATF-v2 `policy_history.json` and emits `epoch_exact_fixed_active_class`, `epoch_exact_fixed_op_list`, and `epoch_exact_fixed_prob_strength`;
  - only router-executable fixed rows are preserved; guarded/frozen rows are not treated as active policy;
  - `apply_preserve_original_policy()` now prefers explicit epoch-exact fields and treats an explicit empty epoch as no active policy instead of falling back to replay class union or `probe_set.class_id`;
  - each feedback update clears stale ops before installing only that epoch's fixed executable policy.
- Dry-run after fix:
  - epoch5 executable classes=`[4, 11]`;
  - epoch15 executable classes=`[12]`;
  - epochs10/20/25/30/35/40/45 executable classes=`[]`;
  - expected/runtime policy_matrix/sample_router/final executable paths are epoch-exact;
  - fixed expected volume=41 industrial / 45 ROI;
  - preserve expected volume after fix=41 industrial / 45 ROI;
  - old volume ratios were industrial=3.780488 and ROI=4.155556; post-fix dry-run ratios are 1.0/1.0.
- Reports:
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/reports/preserve_volume_lifetime_audit.md`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/reports/preserve_volume_lifetime_audit.json`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/reports/preserve_volume_parity_dryrun.md`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/reports/preserve_volume_parity_dryrun.json`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/fixed_vs_preserve_volume_parity_epoch.csv`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/fixed_vs_preserve_volume_parity_by_class.csv`
- Next step: rerun seed0 preserve-weak sanity before seed2. The mainline remains image-only CATF; sampler_only remains outside the main method.

## Seed0 Preserve-Weak Volume-Fixed Sanity Rerun (2026-06-17)

- Scope: ran only seed0 50ep. Seed1, seed2, multiseed, sampler_only, weighted index list, sampling changes, data-split changes, gate changes, causal-score changes, and attenuation-ratio changes were not used.
- Environment: rerun used `D:\Anaconda\envs\pytorch\python.exe` because the base Python had Ultralytics `8.4.48`; the project guard requires the historical training API `8.3.221`.
- Output:
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_volume_fixed/`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_volume_fixed/reports/seed0_preserve_weak_volume_fixed_report.md`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_volume_fixed/reports/seed0_preserve_weak_volume_fixed_report.json`
- Execution:
  - completed_50ep=true;
  - preserve_original=9, weak_roi_texture=0, strict_noop=0;
  - epoch5 executable classes=`[4, 11]`;
  - epoch15 executable classes=`[12]`;
  - epochs10/20/25/30/35/40/45 executable classes=`[]`;
  - stale ops cleared=true;
  - seed-level union avoided=true;
  - weak class9 replacement=false;
  - sampler_only=false; weighted_index_list=false; sampled_distribution_changed=false.
- Volume:
  - fixed CATF-v2 seed0 expected: industrial=41, ROI=45;
  - volume-fixed preserve rerun: industrial=41, ROI=45;
  - ROI affected classes=`{4:9, 11:22, 12:14}`;
  - router_random_draw_count=3050;
  - volume parity matches fixed CATF-v2.
- Metrics:
  - seed0 volume-fixed preserve: P=0.778506, R=0.669654, mAP50=0.743657, mAP50-95=0.489541;
  - vs requested clean seed0: dP=-0.006094, dR=-0.006846, dM50=+0.008957, dM95=+0.013641;
  - vs fixed CATF-v2 seed0: dP=+0.000006, dR=-0.000046, dM50=-0.000043, dM95=+0.000041;
  - vs pre-fix preserve rerun: dP=+0.077992, dR=+0.046109, dM50=+0.031176, dM95=+0.032931;
  - constraint_failed=false under the requested clean seed0 thresholds.
- Interpretation:
  - preserve volume/lifetime parity now holds in training, not just dry-run;
  - the prior seed0 collapse was caused by over-extended preserve policy lifetime and router eligibility;
  - seed0 now reproduces fixed CATF-v2 seed0 behavior and passes constraints;
  - next recommended action is seed2 validation, still image-only and without sampler_only.

## Current Mainline: Image Augmentation CATF (2026-06-13)

- The paper mainline is restored to image augmentation based CATF.
- `sampler_only` has been implemented and verified, but it is demoted from the paper main method.
- `sampler_only` is a training sampling intervention closer to hard example mining / weighted sampling; it changes the training distribution and is not equivalent to image data augmentation.
- `sampler_only` results are retained only as engineering exploration and possible ablation evidence.
- Current image-augmentation mainline baseline: fixed CATF-v2.
- CP-CATF remains relevant only as an image-only controller: causal probe can accept, attenuate, or reject image-space candidates, but main results must not rely on sampling reweighting.
- Future main results must come from image augmentation behavior, not weighted index lists or sampler-only gains.
- Reports:
  - `outputs/experiments/catf_v2_image_only_mainline/reports/image_only_catf_v2_mainline_summary.md`
  - `outputs/experiments/catf_v2_image_only_mainline/reports/image_only_catf_v2_mainline_summary.json`
  - `outputs/experiments/catf_v2_image_only_mainline/reports/sampler_only_demoted_note.md`
- Fixed CATF-v2 image-only baseline:
  - seed0: clean `0.7846/0.6765/0.7347/0.4759`, fixed CATF-v2 `0.7785/0.6697/0.7437/0.4895`, delta `-0.0061/-0.0068/+0.0090/+0.0136`, `constraint_failed=false`;
  - seed1: clean `0.7725/0.6477/0.7542/0.4799`, fixed CATF-v2 `0.7852/0.7005/0.7826/0.5189`, delta `+0.0127/+0.0528/+0.0284/+0.0390`, `constraint_failed=false`;
  - seed2: clean `0.6962/0.7286/0.7692/0.5224`, fixed CATF-v2 `0.7637/0.6863/0.7582/0.4967`, delta `+0.0675/-0.0423/-0.0110/-0.0257`, `constraint_failed=true`.
- Interpretation:
  - seed0/seed1 show CATF image augmentation has real potential;
  - seed2 remains the blocking case because high-risk image augmentation caused non-active regression and mAP drops;
  - seed2 must be fixed inside the image augmentation mainline through causal probe, weak image augmentation / attenuation, strict no-op safety, and non-active regression constraints;
  - do not use `sampler_only`, weighted index lists, or hard-example mining as the paper main result.

## Image-Only Weak Augmentation Replay (2026-06-13)

- Scope: offline replay and method design only. No training was run, no seed was rerun, no clean baseline was rerun, and no gate/sampler/augmentation training logic was changed.
- Added `scripts/replay_weak_image_aug.py`.
- Outputs:
  - `outputs/experiments/catf_v2_image_only_weak_aug_replay/reports/weak_image_aug_replay.md`
  - `outputs/experiments/catf_v2_image_only_weak_aug_replay/reports/weak_image_aug_replay.json`
  - `outputs/experiments/catf_v2_image_only_weak_aug_replay/weak_candidate_records.csv`
  - `outputs/experiments/catf_v2_image_only_weak_aug_replay/reports/weak_image_aug_training_plan.md`
- Replay method:
  - `candidate_policy_3_sampler_only` is disabled and not used;
  - `candidate_policy_1b_weak_roi_texture` is introduced as an image-only downgrade from original ROI texture;
  - weak policy keeps one lower-risk op, uses `prob_multiplier=strength_multiplier=attenuation_ratio`, and caps augmented samples per feedback interval;
  - ratios tested: `0.5` and `0.25`;
  - `0.5` remains too risky, while `0.25` passes the replay gates for legacy image-evidence rows;
  - replay uses no seed-id, class-id, or dataset-class-name hard rule.
- Replay coverage:
  - total image candidates=54;
  - original ROI texture candidates=27;
  - legacy image-evidence candidates=8;
  - weak ROI texture accepted by replay=8;
  - strict no-op=19;
  - all no-op=false;
  - final_val_leakage=false;
  - sampler_only involved=false.
- Seed-level replay:
  - seed0: 1 weak candidate at epoch25, 8 strict no-op;
  - seed1: 2 weak candidates at epochs25/40, 7 strict no-op;
  - seed2: 5 weak candidates at epochs20/25/30/35/45, 4 strict no-op.
- Interpretation:
  - seed2 has offline gate-safe weak image candidates, but this is not a training result;
  - because fixed CATF-v2 seed2 failed through high-risk image augmentation and non-active regression, the next validation should run seed2 image-only weak augmentation 50ep first;
  - if seed2 passes constraints, then run seed0/seed1 sanity;
  - continue to avoid `sampler_only`, weighted index lists, and sampling reweighting in the paper mainline.
- Verification:
  - `python scripts/replay_weak_image_aug.py`
  - `python -m py_compile scripts/replay_weak_image_aug.py`

## Seed2 Image-Only Weak Augmentation Validation (2026-06-13)

- Scope: ran only seed2 for 50 epochs. Seed0/seed1 and multiseed were not run.
- Output:
  - `outputs/experiments/catf_v2_image_only_weak_aug_seed2_50ep/`
  - `outputs/experiments/catf_v2_image_only_weak_aug_seed2_50ep/reports/seed2_weak_image_aug_report.md`
  - `outputs/experiments/catf_v2_image_only_weak_aug_seed2_50ep/reports/seed2_weak_image_aug_report.json`
- Method:
  - image-only CP-CATF weak ROI texture attenuation;
  - `candidate_policy_1b_weak_roi_texture`;
  - attenuation ratio fixed at `0.25`;
  - retained op: `local_contrast`;
  - weak prob/strength: `0.045/0.05`;
  - max augmented samples per feedback interval: `16`;
  - sampler_only disabled; weighted index list disabled; sampled distribution unchanged.
- Execution:
  - 50ep completed=true;
  - weak image augmentation executed=true;
  - industrial images augmented=80;
  - ROI applied=95;
  - router random draw count=1922;
  - weak epochs: 20, 25, 30, 35, 45;
  - strict no-op epochs: 5, 10, 15, 40.
- Metrics:
  - clean seed2: P=0.6962, R=0.7286, mAP50=0.7692, mAP50-95=0.5224;
  - fixed CATF-v2 seed2: P=0.7637, R=0.6863, mAP50=0.7582, mAP50-95=0.4967, constraint_failed=true;
  - weak image aug seed2: P=0.7533, R=0.6942, mAP50=0.7727, mAP50-95=0.5151, constraint_failed=false.
- Delta:
  - vs clean seed2: dP=+0.0570, dR=-0.0344, dM50=+0.0035, dM95=-0.0072;
  - vs fixed CATF-v2 seed2: dP=-0.0104, dR=+0.0079, dM50=+0.0145, dM95=+0.0185.
- Interpretation:
  - weak image augmentation fixes the fixed CATF-v2 seed2 constraint failure under the requested Precision/mAP constraints;
  - class9 is recovered relative to fixed CATF-v2 but not fully recovered relative to clean;
  - non-active regression is mitigated relative to fixed CATF-v2;
  - because seed2 now passes with image-only augmentation, the next validation step is seed0/seed1 sanity, still without sampler_only.
- Verification:
  - requested py_compile passed;
  - requested targeted pytest set passed: `55 passed`.

## Seed0 Fixed-vs-Weak Failure Audit (2026-06-14)

- Scope: analysis only. No training, seed1/seed2 run, multiseed run, sampler change, gate change, attenuation-ratio change, or data-split change was performed.
- Script:
  - `scripts/analyze_seed0_fixed_vs_weak.py`
- Reports:
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/reports/seed0_fixed_vs_weak_failure_audit.md`
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/reports/seed0_fixed_vs_weak_failure_audit.json`
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed0_fixed_vs_weak_epoch_policy_diff.csv`
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed0_fixed_vs_weak_per_class_regression.csv`
- Current weak image augmentation status:
  - seed2 was repaired by image-only weak augmentation;
  - seed1 passes constraints;
  - seed0 fails constraints, so the current global weak replacement is not a final main-method candidate.
- Seed0 mechanism:
  - fixed CATF-v2 seed0 passed with conservative image augmentation on classes `4`, `11`, and `12`, totaling `41` industrial images augmented and `45` ROI applications;
  - weak seed0 executed image augmentation only at epoch `25` on class `9`, totaling `16` industrial images augmented and `18` ROI applications;
  - sampler_only remained disabled, weighted index list remained disabled, and sampled distribution did not change.
- Failure interpretation:
  - weak seed0 did not preserve the original fixed CATF-v2 safe policies for classes `4/11/12`;
  - it introduced a new class `9` weak local-contrast policy instead;
  - Precision collapse is FP-driven and broad, not isolated to class `9`;
  - largest weak precision drops vs clean include classes `5`, `4`, `3`, `11`, `9`, and `7`;
  - largest estimated FP increases vs fixed are led by classes `7`, `9`, `5`, `12`, and `6`;
  - this is non-active regression/spillover and indicates attenuation `0.25` is not automatically safe for seed0.
- Mainline direction:
  - keep the project image-only;
  - do not restore sampler_only as a main-method repair;
  - next design should be `preserve-safe-original + weak-only-for-moderate-risk + strict no-op for high/critical risk`;
  - do not continue training until that image-only decision logic is replayed and reviewed offline.
- Verification:
  - `python -m py_compile scripts/analyze_seed0_fixed_vs_weak.py`

## Preserve-Original + Weak-Only Replay (2026-06-14)

- Scope: offline replay and training-plan generation only. No training, seed rerun, sampler change, gate change, attenuation-ratio change, causal-score change, augmentation-strategy change, or data-split change was performed.
- Script:
  - `scripts/replay_preserve_weak_image_catf.py`
- Outputs:
  - `outputs/experiments/catf_v2_image_only_preserve_weak_replay/reports/preserve_weak_replay.md`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_replay/reports/preserve_weak_replay.json`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_replay/preserve_weak_decision_records.csv`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_replay/reports/preserve_weak_training_plan.md`
- Decision policy:
  - `preserve_original` when fixed CATF-v2 already passed constraints and has an executable conservative image policy;
  - `weak_roi_texture` only when the fixed path is not preservable and attenuation `0.25` passes the secondary replay gate;
  - `strict_noop` for high/critical image risk;
  - `sampler_only` remains absent from the main method.
- Replay counts:
  - seed0: `preserve_original=9`, `weak_roi_texture=0`, `strict_noop=0`;
  - seed1: `preserve_original=9`, `weak_roi_texture=0`, `strict_noop=0`;
  - seed2: `preserve_original=0`, `weak_roi_texture=5`, `strict_noop=4`.
- Checks:
  - seed0 preserves fixed class `4/11/12` strategy and avoids weak class `9` replacement;
  - seed1 preserves the fixed gain path;
  - seed2 converts the failed fixed path to weak/no-op and retains the previously weak-safe candidates;
  - sampler_only and weighted index list are not involved.
- Next training plan, not executed:
  - run seed0 sanity first because fixed seed0 passed and global weak seed0 failed;
  - then run seed2 to confirm the failed fixed seed2 path is still repaired by image-only weak/no-op;
  - run seed1 sanity last;
  - only then consider a three-seed validation.
- Verification:
  - `python -m py_compile scripts/replay_preserve_weak_image_catf.py`

## Seed0 Preserve-Weak Sanity (2026-06-15)

- Scope: ran seed0 only for 50 epochs. Seed1/seed2 and multiseed were not run; clean was not rerun; sampler_only and weighted index list were disabled.
- Output:
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity/`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity/reports/seed0_preserve_weak_sanity_report.md`
  - `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity/reports/seed0_preserve_weak_sanity_report.json`
- Implementation note:
  - added CLI support for `--preserve-original-enabled` and `--weak-only-for-moderate-risk`;
  - added `scripts/build_preserve_weak_decision_schedule.py`;
  - added `scripts/summarize_preserve_weak_seed0_sanity.py`.
- Decision outcome:
  - preserve_original=`9`;
  - weak_roi_texture=`0`;
  - strict_noop=`0`;
  - weak class9 replacement avoided=`true`;
  - sampler_only_enabled=`false`;
  - weighted_index_list_enabled=`false`;
  - sampled_distribution_changed=`false`.
- Execution outcome:
  - image augmentation executed=`true`;
  - industrial images augmented=`20`;
  - ROI applied=`20`;
  - ROI affected classes=`{"11": 20}`;
  - fixed class `4/11/12` executable strategy retained=`false`.
- Metrics:
  - preserve-weak seed0: P/R/mAP50/mAP50-95 `0.666585/0.730019/0.699934/0.466744`;
  - delta vs requested clean seed0: `-0.118015/+0.053519/-0.034766/-0.009156`;
  - delta vs fixed CATF-v2 seed0: `-0.111915/+0.060319/-0.043766/-0.022756`;
  - delta vs previous weak global seed0: `-0.012458/+0.018198/-0.022236/-0.009261`;
  - constraint_failed=`true` with `precision_drop_gt_0.01` and `map50_drop_gt_0.01`.
- Interpretation:
  - event-level replay selection worked, but preserve_original did not replay/install the fixed seed0 class `4/11/12` policy exactly;
  - the run avoided the weak class9 replacement, but only class11 received ROI augmentation;
  - this seed0 sanity failed and does not validate the three-stage strategy;
  - do not proceed to seed2 until preserve_original execution can reproduce the fixed original class-op policy and seed0 passes.
- Verification:
  - requested py_compile checks passed;
  - requested targeted pytest suite passed before the run: `55 passed`.

## CP-CATF Paper-Mode Sampler-Only Multiseed (2026-06-12)

- Scope: continued from the completed seed0 sampler-only run and ran only seed1/seed2. Clean paper baselines and seed0 CP-CATF results were reused; clean and seed0 were not rerun.
- Current positioning: demoted to engineering exploration / ablation only. This run is not a paper main-method candidate because it is a sampling intervention, not image data augmentation.
- Output root:
  - `outputs/experiments/multiseed_cp_catf_paper_mode_sampler_only/`
- Report:
  - `outputs/experiments/multiseed_cp_catf_paper_mode_sampler_only/reports/multiseed_sampler_only_summary.md`
  - `outputs/experiments/multiseed_cp_catf_paper_mode_sampler_only/reports/multiseed_sampler_only_summary.json`
- Sampler-only status:
  - weighted index list was effective for all seeds;
  - sampler_only_effective seed count=3/3;
  - sampled distribution changed for all seeds;
  - final weighted train_core images: seed0=72, seed1=123, seed2=176, total=371.
- Leakage and augmentation status:
  - paper-mode split remains valid: train_core=2071, probe=230, final val=677;
  - final_val_used_for_policy_selection=false and final_val_leakage=false for all seeds;
  - image augmentation remained 0 for all seeds;
  - ROI applied=0 for all seeds;
  - router random draw count=0 for all seeds.
- Metrics versus requested clean paper baselines:
  - seed0 delta: dP=+0.0075, dR=+0.0115, dM50=+0.0213, dM95=+0.0090, constraint_failed=false;
  - seed1 delta: dP=-0.0135, dR=-0.0324, dM50=-0.0365, dM95=-0.0139, constraint_failed=true;
  - seed2 delta: dP=+0.0772, dR=+0.0127, dM50=+0.0253, dM95=+0.0046, constraint_failed=false;
  - mean delta: dP=+0.0237, dR=-0.0027, dM50=+0.0034, dM95=-0.0001.
- Conclusion:
  - 3/3 pass=false; pass_count=2/3.
  - CP-CATF paper-mode sampler-only is not a paper main-method result.
  - The current result proves sampler_only is an effective dataloader intervention, but it is a sampling reweighting method rather than image data augmentation.
  - Do not continue sampler_only as the CP-CATF main direction; keep it only as engineering exploration / ablation evidence.
- Verification:
  - `python -m py_compile AutoAugment/catf_v2/causal_probe.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py scripts/train_yolo_default_with_inloop_feedback.py scripts/summarize_cp_catf_sampler_only_multiseed.py`
  - `python -m pytest -q tests/test_cp_catf_sampler_only.py tests/test_cp_catf_accept_to_execution.py tests/test_cp_catf_paper_mode.py tests/test_catf_v2_causal_probe.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_sample_router.py tests/test_online_augmentation.py`
  - Result: `60 passed`.

## CP-CATF Effective Sampler-Only Dataloader Intervention (2026-06-12)

- Scope: implemented sampler-only dataloader support. No seed1/seed2 training and no multiseed run were started.
- Dataloader entry audit:
  - `outputs/debug/cp_catf_sampler_only_dataloader_impl/dataloader_entry_audit.md`
- Implementation:
  - added `AutoAugment/catf_v2/sampler_only.py` for probe-only train_core sample weighting and before/after distribution audit;
  - added `--sampler-only-enabled` to `scripts/train_yolo_default_with_inloop_feedback.py`;
  - changed `scripts/train_yolo_online_aug.py` so `OnlineYOLODataset` supports weighted index-list mapping and `OnlineAugDetectionTrainer` exposes the active train dataset/loader;
  - chose weighted index list rather than `WeightedRandomSampler` because Ultralytics' active `build_dataloader()` path does not expose a sampler injection argument.
- Behavior:
  - image augmentation remains strict no-op when the precision-aware gate rejects image candidates;
  - sampler-only now changes train_core sampling probability by installing weighted indices and resetting the Ultralytics `InfiniteDataLoader`;
  - original image files, label files, final validation split, and label/Instances content are not rewritten.
- Smoke run:
  - output: `outputs/debug/cp_catf_sampler_only_execution_smoke/`;
  - paper-mode=true, final_val_used_for_policy_selection=false;
  - sample_weight_map generated=true;
  - weighted train_core images=178;
  - weighted_index_list_enabled=true;
  - sampler_only_effective=true;
  - sampled_distribution_changed=true;
  - industrial image augmented=0, ROI applied=0, router random draw count=0;
  - bbox/class legal.
- Seed0 50ep sampler-only run:
  - output: `outputs/experiments/cp_catf_paper_mode_sampler_only_seed0/`;
  - feedback epochs: 5/10/15/20/25/30/35/40/45;
  - epoch5 and epoch10 had sampler-only selected but no train_core image weights >1, so they stayed pending with an explicit blocker;
  - epochs 15/20/25/30/35/40/45 were effective sampler-only with weighted index lists and changed sampled distribution;
  - final sampler-only status=true/effective, weighted train_core images at the final sampler update=72;
  - industrial image augmented=0, ROI applied=0, router random draw count=0.
- Seed0 final metrics against the requested paper clean seed0 baseline:
  - sampler-only seed0: P=0.7588, R=0.6878, mAP50=0.7779, mAP50-95=0.5204;
  - clean paper seed0: P=0.7513, R=0.6763, mAP50=0.7566, mAP50-95=0.5114;
  - delta: dP=+0.0075, dR=+0.0115, dM50=+0.0213, dM95=+0.0090;
  - constraint_failed=false.
- Required audit artifacts:
  - `outputs/debug/cp_catf_sampler_only_dataloader_impl/sample_weight_map.json`
  - `outputs/debug/cp_catf_sampler_only_dataloader_impl/weighted_train_indices.json`
  - `outputs/debug/cp_catf_sampler_only_dataloader_impl/sampled_distribution_before_after.json`
- Verification:
  - `python -m py_compile AutoAugment/catf_v2/sampler_only.py scripts/train_yolo_online_aug.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/causal_probe.py AutoAugment/catf_v2/policy_matrix.py AutoAugment/catf_v2/sample_router.py`
  - `python -m pytest tests/test_cp_catf_sampler_only.py tests/test_cp_catf_accept_to_execution.py tests/test_catf_v2_causal_probe.py -q`
  - Result: `29 passed`.

## CP-CATF Paper-Mode Decision Coverage Audit (2026-06-12)

- Scope: offline decision coverage audit only. No 50ep training, no seed1/seed2 execution, no multiseed run, no clean rerun, and no augmentation strategy change was made.
- Added `scripts/analyze_cp_catf_decision_coverage.py`.
- Audit inputs:
  - `outputs/experiments/multiseed_cp_catf_paper_mode/`
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/`
  - `outputs/experiments/cp_catf_precision_gate_dry_run_seed0/`
  - `outputs/experiments/cp_catf_paper_mode_precision_gate_seed0_rerun/`
  - paper-mode probe split under `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/`
- Paper-mode leakage status remains complete:
  - train_core=2071, probe=230, final val=677;
  - final-val leakage=false;
  - final val was not used for policy selection.
- Accept-to-execution status remains complete:
  - seed0 execution-fixed proved executable ROI image augmentation at epoch25;
  - ROI applied=312, industrial image augmented=226, router random draw count=1330.
- Precision-aware gate status:
  - it blocks the known seed0 Precision-risk `roi_texture` candidate;
  - seed0 precision-gate rerun became sampler-only/strict no-op and matched clean metrics.
- Decision coverage across seeds 0/1/2 and epochs 5/10/15/20/25/30/35/40/45:
  - total_candidates=81;
  - total_image_candidates=54;
  - logged original image causal accepts=8;
  - precision-gate rejects after logged original accept=8;
  - final image accepts=0;
  - sampler-only selected=27;
  - effective sampler-only=0;
  - strict no-op=27.
- Main rejection buckets under current replay:
  - no_positive_benefit=54;
  - high_fp_spillover_rate_too_high=40;
  - non_active_regression_too_high=40;
  - estimated_precision_drop_too_high=20;
  - non_active_fp_delta_too_high=20;
  - high_confidence_fp_delta_too_high=20.
- Interpretation:
  - current precision-aware decision stack is over-conservative for image-augmentation coverage;
  - high_confidence_fp_delta > 0.0 is not the sole blocker, and relaxing any single precision threshold admits zero candidates under current replay;
  - 8 legacy roi_texture accepts are plausible attenuation candidates, but none is safe at original strength/probability under the current gate;
  - sampler_only was later implemented and verified, but is now demoted to engineering exploration / ablation only;
  - paper-mainline work should design causal-probe image acceptance and graded image-augmentation attenuation before any image-rerun.
- Reports:
  - `outputs/experiments/cp_catf_decision_coverage_audit/reports/decision_coverage_audit.md`
  - `outputs/experiments/cp_catf_decision_coverage_audit/reports/decision_coverage_audit.json`
  - `outputs/experiments/cp_catf_decision_coverage_audit/decision_records.csv`
  - `outputs/experiments/cp_catf_decision_coverage_audit/rejection_reason_summary.csv`
  - `outputs/experiments/cp_catf_decision_coverage_audit/seed_level_coverage.csv`
- Verification:
  - `python scripts/analyze_cp_catf_decision_coverage.py`
  - `python -m py_compile scripts/analyze_cp_catf_decision_coverage.py`
  - `python -m py_compile AutoAugment/catf_v2/causal_probe.py`
  - `python -m py_compile AutoAugment/catf_v2/policy_matrix.py`
  - `python -m py_compile AutoAugment/catf_v2/sample_router.py`

## CP-CATF Precision-Aware Gate Dry Run and Seed0 Rerun (2026-06-12)

- Scope: seed0 only. No seed1, seed2, clean rerun, or multiseed run was started.
- Added `--precision-aware-accept-gate` CLI support in `scripts/train_yolo_default_with_inloop_feedback.py`.
- Added dry-run replay script:
  - `scripts/run_cp_catf_precision_gate_dry_run_seed0.py`
  - Output: `outputs/experiments/cp_catf_precision_gate_dry_run_seed0/`
- Dry run replayed the existing paper-mode seed0 execution-fixed epoch25 accept:
  - original candidate: `candidate_policy_1_roi_texture`;
  - active class: `9`;
  - ops: `sharpen_mild` and `local_contrast`;
  - original execution had ROI applied=312, industrial image augmented=226, router random draw count=1330.
- New precision-aware gate rejects the epoch25 roi_texture candidate before training:
  - `estimated_precision_drop=0.0300`;
  - `non_active_fp_delta=0.0500`;
  - `high_confidence_fp_delta=0.0500`;
  - selected fallback: `candidate_policy_3_sampler_only`;
  - sampler weighting is still pending dataloader support, so the image path is strict no-op.
- Because the dry run rejected the known high-risk candidate, seed0 was rerun for 50 epochs at:
  - `outputs/experiments/cp_catf_paper_mode_precision_gate_seed0_rerun/`
- Rerun behavior:
  - feedback epochs 5/10/15/20/25/30/35/40/45 all selected `candidate_policy_3_sampler_only`;
  - no image candidate was accepted;
  - ROI applied=0, industrial image augmented=0, router random draw count=0;
  - bbox/class checks stayed legal;
  - final validation leakage=false and final validation was not used for policy selection.
- Seed0 rerun metrics against the requested paper clean seed0 baseline:
  - CP-CATF precision-gate seed0: P=0.7513, R=0.6763, mAP50=0.7566, mAP50-95=0.5114.
  - Delta vs paper clean seed0: all `0.0000` at report precision.
  - `constraint_failed=false` against the requested paper clean seed0 baseline.
- Important interpretation:
  - Precision-aware gate successfully blocks the known seed0 FP-spillover image candidate.
  - This rerun is not evidence of paper-mode image-augmentation benefit, because all image candidates were rejected.
  - Sampler-only was later connected to the dataloader and verified, but is now demoted from the paper mainline because it is sampling reweighting, not image augmentation.
  - Do not proceed to formal seed1/seed2 image-augmentation validation until an image candidate can pass the precision gate or be safely attenuated inside the image path.
- Reports:
  - `outputs/experiments/cp_catf_precision_gate_dry_run_seed0/reports/precision_gate_dry_run_report.md`
  - `outputs/experiments/cp_catf_precision_gate_dry_run_seed0/reports/precision_gate_dry_run_report.json`
  - `outputs/experiments/cp_catf_paper_mode_precision_gate_seed0_rerun/reports/seed0_precision_gate_rerun_report.md`
  - `outputs/experiments/cp_catf_paper_mode_precision_gate_seed0_rerun/reports/seed0_precision_gate_rerun_report.json`
- Verification:
  - `python -m py_compile AutoAugment/catf_v2/causal_probe.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py scripts/run_cp_catf_precision_gate_dry_run_seed0.py scripts/summarize_cp_catf_precision_gate_seed0_rerun.py`
  - `pytest -q tests/test_catf_v2_causal_probe.py tests/test_cp_catf_accept_to_execution.py tests/test_cp_catf_paper_mode.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_policy_matrix.py tests/test_catf_v2_sample_router.py tests/test_catf_v2_roi_augmentation.py tests/test_inloop_feedback_training.py tests/test_online_augmentation.py`
  - Result: `70 passed`.

## CP-CATF Precision-Aware Accept Gate (2026-06-11)

- No training, seed1/seed2 run, or multiseed run was started for this update.
- The paper-mode seed0 execution-fixed run already proved the accept-to-execution path: `candidate_policy_1_roi_texture` accepted at epoch 25, ROI applied=312, industrial image augmented=226, router random draw count=1330, and final validation leakage=false.
- Seed0 still failed the industrial constraint because Precision changed from 0.7513 to 0.7399 (delta=-0.0114), while mAP50 improved slightly and mAP50-95 remained within the 0.01 tolerance.
- The seed0 precision-risk audit attributed the failure mainly to non-active false-positive spillover, not to the active class itself; class 9 improved locally while non-active classes drove the operating-point Precision loss.
- Added a generic precision-aware reject gate to `AutoAugment/catf_v2/causal_probe.py`:
  - `estimated_precision_drop > 0.005` rejects image candidates.
  - `non_active_fp_delta > 0.005` rejects image candidates.
  - `high_confidence_fp_delta > 0.0` rejects image candidates.
- These precision-gate fields do not change the causal score formula; they are required accept conditions for image-modifying candidates.
- Paper-mode risk estimation in `scripts/train_yolo_default_with_inloop_feedback.py` now uses the full probe-split per-class context for non-active FP risk, while candidate selection still comes from active rows.
- Added tests covering precision-gate rejection, causal-score stability, and paper-mode non-active context use.
- Verification:
  - `python -m py_compile AutoAugment/catf_v2/causal_probe.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py`
  - `pytest -q tests/test_cp_catf_accept_to_execution.py tests/test_cp_catf_paper_mode.py tests/test_catf_v2_causal_probe.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_policy_matrix.py tests/test_catf_v2_sample_router.py tests/test_catf_v2_roi_augmentation.py tests/test_inloop_feedback_training.py tests/test_online_augmentation.py`
  - Result: `70 passed`.
- Report:
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_precision_aware_gate_update.md`
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_precision_aware_gate_update.json`

## CP-CATF Multiseed Training Validation (2026-06-09)

- CP-CATF development-mode validation remains feasibility evidence for image-space causal control, not the current paper main result.
- The current paper mainline is image augmentation based CATF, with fixed CATF-v2 as the image-only baseline and CP-CATF image-only as the repair direction.
- RiskGuard has been downgraded to an audit/debug prior and is not used as the final accept/reject rule.
- Added offline probe decision integration to `scripts/train_yolo_default_with_inloop_feedback.py`:
  - `--causal-probe-mode true`
  - `--use-offline-probe-decisions true`
  - `--offline-probe-decisions-file`
  - `--offline-probe-decisions-dir`
- Added `scripts/summarize_catf_v2_cp_catf_multiseed.py` and additional causal-probe tests for offline decision application.
- Full multiseed CP-CATF training validation completed at:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_cp_catf/`
- Probe decisions used in training:
  - seed0: `candidate_policy_1_roi_texture`, accepted, image augmentation allowed.
  - seed1: `candidate_policy_1_roi_texture`, accepted, image augmentation allowed.
  - seed2: `candidate_policy_3_sampler_only`, image augmentation rejected; sample weighting is pending dataloader support, so the actual path is strict image no-op.
- Final CP-CATF metrics:
  - seed0: P=0.7785, R=0.6697, mAP50=0.7437, mAP50-95=0.4895, `constraint_failed=false`.
  - seed1: P=0.7852, R=0.7005, mAP50=0.7826, mAP50-95=0.5189, `constraint_failed=false`.
  - seed2: P=0.6962, R=0.7286, mAP50=0.7692, mAP50-95=0.5224, `constraint_failed=false`.
- Outcome:
  - constraint pass count: `3/3`.
  - seed0 retained fixed CATF-v2 mAP gains.
  - seed1 retained fixed CATF-v2's clear gains.
  - seed2 rejected image augmentation and preserved clean parity.
  - seed2 industrial image samples augmented=0, ROI applied=0, router random draw count=0.
  - OK3 was never active and OK3 ROI applied remained 0 across all seeds.
- Mean CP-CATF delta vs clean: dP=+0.0022, dR=+0.0153, dM50=+0.0125, dM95=+0.0175.
- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_cp_catf/reports/multiseed_cp_catf_summary.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_cp_catf/reports/multiseed_cp_catf_summary.json`
- Verification:
  - `python -m py_compile AutoAugment/catf_v2/causal_probe.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py scripts/summarize_catf_v2_cp_catf_multiseed.py`
  - Requested pytest suite result: `134 passed`.
- Paper-mode caveat: this training validation still uses development-mode offline probe decisions from existing validation diagnostics. If CP-CATF is used as the final paper method, the next step must replace this with a train/probe split or train hard-example probe set; do not claim a leakage-free final result until paper-mode CP-CATF also passes.

## CATF-v2 RiskGuard Seed2 Validation (2026-06-08)

- Implemented `--catf-riskguard true` as a seed-agnostic high-risk class-op guard.
- Added high-risk registry module `AutoAugment/catf_v2/high_risk_class_ops.py`.
- Current registry blocks class 9 with `sharpen_mild` and `local_contrast` unless a future causal probe explicitly clears the pair.
- The guard runs at policy-matrix time and the sample router has a runtime fallback before op accounting or random draw, so blocked ops do not modify images/labels/Instances and do not consume CATF random draws.
- Added `tests/test_catf_v2_riskguard.py`; targeted regression suite passed `121 passed`.
- Seed2 RiskGuard 50ep run:
  - Output: `outputs/experiments/catf_v2_riskguard_seed2_50ep/`
  - Correct seed2 clean reference/control was used.
  - Epoch5 class 9 `texture_boundary_weak` was reproduced.
  - RiskGuard blocked class 9 `local_contrast` and `sharpen_mild` at epoch5.
  - Sampler-only fallback was recorded, but sample weighting remains pending dataloader integration.
  - Final metrics: P=0.6394, R=0.7231, mAP50=0.7552, mAP50-95=0.5073.
  - Delta vs clean seed2: P=-0.0569, R=-0.0056, mAP50=-0.0140, mAP50-95=-0.0150.
  - `constraint_failed=true`; failure reasons are Precision, mAP50, and mAP50-95 drops beyond 0.01.
  - Industrial samples augmented dropped from fixed CATF-v2 seed2 `86` to `13`.
  - ROI applied dropped from fixed `90` to `15`; RiskGuard ROI affected classes were only class 12.
  - OK3 remained inactive and OK3 ROI applied stayed 0.
- Class 9 local result improved after blocking:
  - class 9 Recall clean/fixed/RiskGuard: 0.6310 / 0.4643 / 0.7960.
  - class 9 AP50 clean/fixed/RiskGuard: 0.7440 / 0.6973 / 0.7944.
  - class 9 AP50-95 clean/fixed/RiskGuard: 0.4139 / 0.3484 / 0.4198.
- Interpretation: RiskGuard is useful and should remain as a targeted safety mechanism, but it is necessary and not sufficient. The overall seed2 run still fails due to residual global Precision/mAP degradation and class12-only ROI activity.
- Because seed2 did not pass, seed0/seed1 sanity check was not run.
- Reports:
  - `outputs/experiments/catf_v2_riskguard_seed2_50ep/reports/final_report.md`
  - `outputs/experiments/catf_v2_riskguard_seed2_50ep/reports/final_metrics.json`
  - `outputs/experiments/catf_v2_riskguard_seed2_50ep/reports/riskguard_events.json`
  - `outputs/experiments/catf_v2_riskguard_seed2_50ep/reports/compare_with_clean_and_fixed_catf_v2.md`
  - `outputs/experiments/catf_v2_riskguard_seed2_50ep/reports/riskguard_seed2_summary.json`

## CATF-v2 Strategy Limitation Analysis (2026-06-07)

- Added a strategy-level limitation report based on the completed seed2 root-cause audit:
  - `outputs/experiments/seed2_failure_root_cause/reports/catf_v2_strategy_limitation_analysis.md`
  - `outputs/experiments/seed2_failure_root_cause/reports/catf_v2_strategy_limitation_analysis.json`
- No training, no 10ep/50ep ablation, and no CATF-v2 rule changes were made.
- Main conclusion: fixed CATF-v2 is not a failed method. It has valid gains on seed0/seed1 and improves average metrics, but seed2 exposes immature strategy selection and risk control.
- Current strategy limitations supported by seed2 evidence:
  - diagnosis does not directly imply augmentation benefit;
  - class 9 was diagnosed as `texture_boundary_weak`, but ROI texture enhancement led to class 9 Recall/AP regression;
  - candidate augmentation lacks pre-training causal validation;
  - active-class-only monitoring cannot prevent non-active class regression;
  - `sharpen_mild` and `local_contrast` are co-enabled, so operator-level risk is not isolated;
  - fallback/gate can fire after weights have already been affected by candidate augmentation;
  - strong clean-baseline seeds should prefer strict image no-op until image-augmentation causal evidence is positive.
- Minimal next improvements are CP-CATF causal probe, active/non-active dual constraints, and high-risk class-op candidate gating for combinations such as class 9 + ROI texture.

## Seed2 CATF-v2 Failure Root-Cause Audit (2026-06-07)

- Completed a seed2 root-cause audit at `outputs/experiments/seed2_failure_root_cause/`.
- No training was run. The audit only read existing clean/fixed/gated/Safe/adaptive-RB artifacts and ran predict-only validation on existing clean/fixed best weights to cache prediction JSON and debug images.
- Curve localization:
  - fixed CATF-v2 and Gated remain identical to clean through epoch 5.
  - Recall first lags clean at epoch 6.
  - mAP50 first clearly lags clean at epoch 8.
  - mAP50-95 first turns negative at epoch 6 and clearly lags by epoch 8.
- Most suspicious policy update: epoch 5 `accept/propose` for class 9 with `texture_boundary_weak`, enabling `sharpen_mild` and `local_contrast`.
- Fixed seed2 augmentation audit:
  - industrial samples augmented=86.
  - ROI applied=90.
  - router random draw count=5610.
  - ops: `local_contrast` applied 44, `sharpen_mild` applied 42.
  - ROI affected classes: class 9=25, class 11=59, class 8=6.
- Class-level finding:
  - class 9 is active/ROI-affected and regresses strongly: Recall -0.1667, AP50 -0.0466, AP50-95 -0.0654, estimated FN +14.
  - class 8 is active/ROI-affected and loses AP: AP50 -0.0559, AP50-95 -0.0502.
  - class 11 is active/ROI-affected but improves slightly, so texture ops are not uniformly harmful.
  - non-active regression exists: classes 10, 12, 6, 5, 3, and 2 show AP or Recall regressions without ROI application.
- Prediction diff on existing best weights found clean-detected/fixed-missed objects, fixed localization degradation, and fixed confidence drops; pre-NMS NMS ordering cannot be audited from saved post-NMS predictions.
- Root-cause ranking:
  1. fallback/gate too late after epoch5 candidate activation;
  2. class 9 texture ROI intervention;
  3. missing causal validation for active issue attribution;
  4. non-active class regression;
  5. seed2 clean baseline is strong enough that default no-op is justified unless causal evidence is positive.
- Recommendation: seed2 should default to strict image no-op until CP-CATF causal probe or a short image-only ablation proves class-9 texture intervention is safe.
- Reports:
  - `outputs/experiments/seed2_failure_root_cause/reports/seed2_curve_degradation_analysis.md`
  - `outputs/experiments/seed2_failure_root_cause/reports/seed2_per_class_regression_analysis.md`
  - `outputs/experiments/seed2_failure_root_cause/reports/seed2_augmentation_operator_attribution.md`
  - `outputs/experiments/seed2_failure_root_cause/reports/seed2_active_vs_regressed_class_analysis.md`
  - `outputs/experiments/seed2_failure_root_cause/reports/seed2_prediction_diff_analysis.md`
  - `outputs/experiments/seed2_failure_root_cause/reports/seed2_root_cause_summary.md`

## CATF-v2 Adaptive-RB Full Multiseed Validation (2026-06-07)

- Full multiseed adaptive burn-in + RB validation completed at `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_adaptive_rb/`.
- Fixed epoch5 has been upgraded conceptually to adaptive burn-in. `feedback_start_epoch=5` remains only the empirical earliest burn-in check point, not a claimed optimal trigger.
- Seed2 strong clean-baseline protection remains effective:
  - adaptive start epoch: `None`
  - candidate branch: false
  - rollback: false
  - no-op fallback epoch: 15
  - final metrics exactly match clean seed2: P=0.6962, R=0.7286, mAP50=0.7692, mAP50-95=0.5224
  - industrial samples=0, ROI applied=0, router random draw count=0
  - `constraint_failed=false`
- Full multiseed adaptive-RB metrics:
  - seed0: P=0.7846, R=0.6765, mAP50=0.7347, mAP50-95=0.4759, `constraint_failed=false`.
  - seed1: P=0.7550, R=0.7261, mAP50=0.7653, mAP50-95=0.4938, `constraint_failed=true` due to `precision_drop_gt_0.01`.
  - seed2: P=0.6962, R=0.7286, mAP50=0.7692, mAP50-95=0.5224, `constraint_failed=false`.
- Gate behavior:
  - seed0 did not start candidate and fell back to strict no-op at epoch 15; it passed constraints but lost fixed CATF-v2's mAP gains.
  - seed1 started a low-risk RB candidate at epoch 15, saved RB checkpoint, accepted at epoch 20, applied 17 industrial samples and 20 ROI operations on class 12, and improved Recall/mAP versus clean, but failed the precision constraint.
  - seed2 stayed strict no-op and preserved clean parity.
- OK3 remained inactive and OK3 ROI applied remained 0 across all adaptive-RB runs.
- Outcome: adaptive-RB achieved `2/3` constraint pass, not `3/3`. It should not be claimed as the final paper main method in its current form.
- Interpretation: adaptive burn-in + RB is a useful safety architecture and fixes the seed2 failure mode before augmentation affects weights, but the current rule is still too conservative for seed0 and insufficiently precision-aware for seed1.
- Next tuning priorities:
  - accumulate burn-in evidence across epoch 5/10/15 so seed0 is not lost when the epoch15 diagnosis alone has insufficient evidence;
  - make RB accept/rollback compare against clean/reference industrial constraints, not only the immediate probe reference;
  - add a precision floor or threshold-calibration step before accepting low-risk candidates with recall/mAP gains.
- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_adaptive_rb/reports/multiseed_adaptive_rb_summary.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_adaptive_rb/reports/multiseed_adaptive_rb_summary.json`

## CATF-v2 Adaptive Burn-in + RB Seed2 Validation (2026-06-06)

- Implemented adaptive burn-in for CATF-v2 with `--adaptive-burnin true`; the old fixed `feedback_start_epoch=5` is now treated only as an empirical minimum burn-in point, not as a theoretically optimal augmentation trigger.
- Added configurable burn-in parameters: `min_burnin_epoch`, `max_burnin_epoch`, `burnin_check_interval`, `metric_stability_window`, `map50_stability_threshold`, `recall_stability_threshold`, `min_diagnosis_evidence`, `min_active_class_evidence`, and `allow_force_start_at_max_burnin`.
- Added `--catf-rollback-mode true` and a first-branch rollback helper that saves an auditable safe checkpoint when adaptive burn-in starts a candidate branch. If a candidate is rejected, RB can restore trainer model/EMA/optimizer/scheduler state and force strict no-op policy.
- Adaptive burn-in states:
  - `burnin_observe`: YOLO default only, diagnosis allowed, no industrial/ROI augmentation, no CATF random draws.
  - `candidate_branch`: starts only after the model is diagnostically usable and evidence is sufficient.
  - `no_op_fallback`: strict clean/no-op path when max burn-in is reached without readiness.
- Fixed epoch5 should not be claimed as optimal in the paper. The method should be described as an adaptive burn-in framework that triggers candidate augmentation when validation curves are stable and diagnostic evidence is credible.
- Retrospective simulation:
  - Report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/adaptive_burnin_retrospective_simulation.md`
  - JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/adaptive_burnin_retrospective_simulation.json`
  - Result: seed0 would start a low-risk RB candidate at epoch 15, seed1 would start a low-risk RB candidate at epoch 15, seed2 would not start a candidate and would no-op fallback at epoch 15 under strong clean-baseline protection.
  - The rule avoids Safe's epoch5 early no-op and avoids Gated seed2's epoch10 fallback after already-damaging tentative augmentation.
- 10ep adaptive burn-in smoke:
  - Run: `outputs/experiments/catf_v2_adaptive_burnin_10ep_smoke/`
  - Seed: 2, epochs=10, YOLO default augmentation enabled, train images=2301.
  - Epoch5 checked `start_condition`; candidate branch did not start.
  - Reasons: `metric_unstable` and `strong_clean_baseline_protection`.
  - Industrial samples augmented=0, ROI applied=0, router random draw count=0, bbox/class legal=true.
  - Report: `outputs/experiments/catf_v2_adaptive_burnin_10ep_smoke/reports/adaptive_burnin_smoke_report.md`
- Seed2 adaptive-burnin + RB 50ep:
  - Run: `outputs/experiments/catf_v2_adaptive_rb_seed2_50ep/`
  - Adaptive start epoch: `None`; candidate branch did not start.
  - No-op fallback epoch: 15, reason `adaptive_burnin_not_ready`.
  - Rollback was not needed because no candidate branch was entered.
  - Strict no-op audit: industrial samples augmented=0, ROI applied=0, router random draw count=0.
  - Final metrics exactly match clean seed2: P=0.6962, R=0.7286, mAP50=0.7692, mAP50-95=0.5224.
  - `constraint_failed=false`, epochs 1..50 continuous, bbox/class legal=true.
  - Reports:
    - `outputs/experiments/catf_v2_adaptive_rb_seed2_50ep/reports/adaptive_rb_seed2_50ep_report.md`
    - `outputs/experiments/catf_v2_adaptive_rb_seed2_50ep/reports/adaptive_rb_seed2_50ep_summary.json`
- Interpretation: adaptive burn-in + RB is more methodologically defensible than fixed epoch5 triggering. Seed2 is protected before any CATF augmentation affects weights. Full multiseed adaptive-RB has since been completed; see the 2026-06-07 section above for the final multiseed result.

## CATF-v2-Gated Multiseed Validation (2026-06-06)

- Implemented `--catf-gated-mode true` as a new CATF-v2 controller mode; CATF-v2-Safe remains available as `--catf-safe-mode true`.
- Retrospective simulation over fixed CATF-v2 artifacts predicted the desired behavior: seed0 no fallback, seed1 no fallback, seed2 fallback at epoch 10, expected `3/3` constraint pass.
- Full 50ep multiseed CATF-v2-Gated was then run at `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/`.
- Configuration stayed within the required protocol: `yolo11n.pt`, safe tiled no-OK-position dataset, epochs=50, imgsz=1024, batch=2, workers=0, device=0, YOLO default augmentation enabled, no fixed augmented dataset, no copy-paste, train images=2301, and continuous single-run `results.csv` epochs 1..50.
- CATF-v2-Gated final metrics:
  - seed0: P=0.7785, R=0.6697, mAP50=0.7437, mAP50-95=0.4895, constraint_failed=false.
  - seed1: P=0.7852, R=0.7005, mAP50=0.7826, mAP50-95=0.5189, constraint_failed=false.
  - seed2: P=0.7850, R=0.6795, mAP50=0.7521, mAP50-95=0.5083, constraint_failed=true.
- Gated retained fixed CATF-v2 gains on seed0 and seed1 exactly, with real augmentation applied:
  - seed0: industrial samples=41, ROI applied=45.
  - seed1: industrial samples=55, ROI applied=56.
- Seed2 triggered strict no-op fallback at epoch 10 with `epoch10_bad_pattern_A` and all op probabilities zero from `active_policy_epoch_010.json` onward, but still failed mAP constraints because tentative augmentation between epochs 6 and 10 had already changed the training trajectory.
- Seed2 Gated totals: industrial samples=22, router draws=1540, ROI applied=25 before fallback; strict no-op after fallback was verified from saved active policies.
- OK3 remained inactive and OK3 ROI applied remained 0 across all Gated runs.
- Real outcome: CATF-v2-Gated is `2/3` constraint pass, not `3/3`. The retrospective simulation was overly optimistic because it assumed an epoch10 fallback could select clean fallback, but the real single-run training cannot rewind weights after tentative augmentation.
- Interpretation: CATF-v2-Gated is valuable diagnostically because it preserves seed0/seed1 benefits and exposes the need for pre-application gating or checkpoint rollback, but it is not ready as the paper main method.
- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/gated_controller_retrospective_simulation.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/gated_controller_retrospective_simulation.json`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/reports/multiseed_catf_v2_gated_summary.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/reports/multiseed_catf_v2_gated_summary.json`

## CATF-v2-Safe Multiseed Validation (2026-06-05)

- Run group: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe/`.
- Newly completed CATF-v2-Safe seed0 and seed1 50ep runs; reused the completed seed2 Safe 50ep run via `seed_2/catf_v2_safe/` summary/link artifacts.
- Configuration stayed within the industrial protocol: `yolo11n.pt`, tiled safe no-OK-position dataset, epochs=50, imgsz=1024, batch=2, workers=0, device=0, YOLO default augmentation enabled, no fixed augmented dataset, no copy-paste, train images=2301, and single-run continuous results.csv epochs 1..50.
- CATF-v2-Safe final metrics:
  - seed0: P=0.7846, R=0.6765, mAP50=0.7347, mAP50-95=0.4759, constraint_failed=false.
  - seed1: P=0.7725, R=0.6477, mAP50=0.7542, mAP50-95=0.4799, constraint_failed=false.
  - seed2: P=0.6962, R=0.7286, mAP50=0.7692, mAP50-95=0.5224, constraint_failed=false.
- Safe vs clean native YOLO default deltas are exactly 0.0000 for all four metrics on all three seeds.
- Constraint status improved from fixed CATF-v2 `1/3` failed to CATF-v2-Safe `0/3` failed, so Safe achieves `3/3` industrial constraint pass.
- Important tradeoff: Safe triggered early-abstention no-op fallback at epoch 5 on all seeds. Industrial samples augmented=0, ROI applied=0, and router random draws=0 for seed0/1/2.
- Active class proposals before fallback:
  - seed0: class 11 and class 4.
  - seed1: class 11 and class 4.
  - seed2: class 9.
- OK3 remained inactive and OK3 ROI applied remained 0 across all Safe runs.
- Seed0 passed constraints but lost fixed CATF-v2's mAP gains; seed1 also passed constraints but did not retain fixed CATF-v2's clear improvement. Seed2 was protected exactly as intended through clean parity.
- Interpretation: CATF-v2-Safe is effective as a conservative safety/protection layer and fixes seed2, but it is too conservative to serve as the sole paper main augmentation method because it also abstains on seed0/seed1 and removes valid fixed CATF-v2 gains.
- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe/reports/multiseed_catf_v2_safe_summary.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe/reports/multiseed_catf_v2_safe_summary.json`

## CATF-v2-Safe Seed2 Validation (2026-06-04)

- Implemented CATF-v2-Safe as a conservative protection layer over CATF-v2:
  - high-recall baseline protection,
  - negative-effect attribution,
  - early abstention,
  - safe accept guards,
  - strict no-op fallback.
- No seed0/seed1 retraining was run.
- Seed2 failure localization report:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed2_safe_controller_design.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed2_safe_controller_design.json`
- Seed2 curve finding: fixed CATF-v2 first lags clean Recall by >0.015 at epoch 6, first lags clean mAP50-95 by >0.008 at epoch 8, and shows a mAP50-95 guard by feedback epoch 10. Epoch 5 proposed class 9 (`轮廓划伤`) without positive Recall/mAP gain, so Safe should abstain.
- 10ep smoke:
  - Output: `outputs/experiments/catf_v2_safe_10ep_smoke/`.
  - Training completed, train images=2301, no fixed augmented dataset, bbox/class legal.
  - Because the smoke uses a 10ep schedule against a 50ep reference curve, Safe recorded `safe_accept_blocked` rather than no-op freeze; this validates control wiring but is not the formal seed2 behavior.
- Seed2 50ep Safe run:
  - Output: `outputs/experiments/catf_v2_safe_seed2_50ep/`.
  - Single-run continuous training, epochs 1..50, YOLO default augmentation enabled.
  - Safe triggered no-op freeze at epoch 5 via `early_abstention_no_recall_or_map_gain`.
  - Industrial samples augmented=0, ROI applied=0, router random draws=0.
  - Final metrics match clean seed2 exactly: Precision=0.6962, Recall=0.7286, mAP50=0.7692, mAP50-95=0.5224.
  - Constraint failed: false.
  - Compared with fixed CATF-v2 seed2, Safe recovers Recall by +0.0423, mAP50 by +0.0110, and mAP50-95 by +0.0257 while accepting lower Precision because the clean high-recall baseline is protected.
- Reports:
  - `outputs/experiments/catf_v2_safe_seed2_50ep/reports/final_report.md`
  - `outputs/experiments/catf_v2_safe_seed2_50ep/reports/final_metrics.json`
  - `outputs/experiments/catf_v2_safe_seed2_50ep/reports/safe_controller_events.json`
  - `outputs/experiments/catf_v2_safe_seed2_50ep/reports/compare_with_clean_and_fixed_catf_v2.md`
- Interpretation: CATF-v2-Safe fixes the known seed2 failure by abstaining on a strong clean baseline rather than forcing augmentation. Next step should be full multiseed CATF-v2-Safe validation only, not further threshold tuning.

## Official-Path CATF-v2 Threshold Re-optimization (2026-06-04)

- No training was run.
- Analysis path: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/`.
- Script: `scripts/reoptimize_catf_v2_thresholds_official_path.py`.
- Evaluation source: existing official Ultralytics `YOLO.predict(conf=0.10)` outputs plus the official-path post-processing evaluator; the old cached post-hoc evaluator is not used as the final judge.
- Pass counts under industrial constraints:
  - fixed CATF-v2 without RC: `2/3`.
  - old saved unified RC thresholds (`catf_v2_rc_per_class_thresholds.json`): `1/3`.
  - reoptimized unified precision-guard RC: `2/3`.
  - per-seed RC: `3/3`.
  - conservative default RC: `2/3`.
- Reoptimized unified thresholds cannot make all three seeds pass. Seed0 remains just outside the mAP50-95 guard (`delta mAP50-95=-0.0108`), so unified CATF-v2-RC should not be claimed as a stable 3/3 main method.
- Per-seed calibration can pass `3/3`, so RC is best framed as deployment/model-specific threshold calibration, not as the core training method.
- The old RC failed because it lowered many classes to `0.10`, including high-FP-prior or FP-sensitive classes, causing Precision failures on seed0 and seed2.
- Recommended paper line: fixed CATF-v2 remains the training/augmentation method; threshold calibration is a deployment-time companion and an official-path ablation, not the headline method claim.
- Outputs:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/official_threshold_reoptimization.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/official_threshold_reoptimization.json`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/unified_precision_guard_thresholds.json`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/per_seed_thresholds.json`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/conservative_default_thresholds.json`

## Fixed CATF-v2 Multiseed Validation (2026-06-04)

- Run group: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/`.
- Code commit used for training: `dfcd177fa058046073e9b8e87dc8052b7b705f9a`.
- Reused clean native YOLO default seed0/1/2 and the fixed seed1 CATF-v2 run; newly ran fixed CATF-v2 seed0 and seed2.
- Dataset and training configuration stayed unchanged: `yolo11n.pt`, `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`, epochs=50, imgsz=1024, batch=2, workers=0, official YOLO default augmentation enabled, no fixed augmented dataset generated, train images=2301.
- Fixed CATF-v2 per-seed results:
  - seed0: P=0.7785, R=0.6697, mAP50=0.7437, mAP50-95=0.4895, constraint_failed=false.
  - seed1: P=0.7852, R=0.7005, mAP50=0.7826, mAP50-95=0.5189, constraint_failed=false.
  - seed2: P=0.7637, R=0.6863, mAP50=0.7582, mAP50-95=0.4967, constraint_failed=true.
- Fixed CATF-v2 aggregate vs clean:
  - constraint_failed improved from old CATF-v2 `2/3` to fixed CATF-v2 `1/3`.
  - OK3 was never activated and OK3 ROI applied remained 0 across all fixed seeds.
  - Active class counts: class 11 x3, class 4 x2, class 12 x1, class 9 x1, class 8 x1.
  - ROI affected class counts: class 11=132, class 9=25, class 4=14, class 12=14, class 8=6.
- Interpretation: strict no-augmentation bypass repair materially improved multiseed stability, but seed2 still fails mAP constraints; CATF-v2 is a stronger candidate, not yet a fully stable sole main method.
- Reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.json`

## CATF-v2 Fixed Seed1 50 Epoch Rerun (2026-06-03)

- Run: `outputs/experiments/catf_v2_fixed_seed1_50ep/`.
- Purpose: rerun the critical CATF-v2 seed1 case after the strict no-augmentation bypass fix, because the old seed1 result had `ROI applied=0` and `industrial samples augmented=0`.
- Configuration: `yolo11n.pt`, safe no-OK-position tiled dataset, epochs=50, imgsz=1024, batch=2, workers=0, device=0, seed=1.
- Training mode: single-run continuous YOLO default + CATF-v2; official YOLO default augmentation remained enabled; no fixed augmented dataset was generated.
- Train images remained 2301; `results.csv` epochs are 1..50; final `best.pt` is from the single global run.
- Clean native seed1 reference: Precision=0.7725, Recall=0.6477, mAP50=0.7542, mAP50-95=0.4799.
- Fixed CATF-v2 seed1: Precision=0.7852, Recall=0.7005, mAP50=0.7826, mAP50-95=0.5189.
- Delta vs clean native seed1: Precision +0.0127, Recall +0.0528, mAP50 +0.0284, mAP50-95 +0.0390.
- Delta vs old CATF-v2 seed1: Precision +0.0161, Recall +0.0055, mAP50 +0.0277, mAP50-95 +0.0291.
- Constraint status: `constraint_failed=false`.
- This fixed run did apply augmentation: industrial samples augmented=55, ROI applied=56, affected classes class 11 (51 times) and class 4 (5 times).
- OK3 was not activated and OK3 ROI applied remained 0.
- Policy actions: shrink=5, freeze=2, accept=1, observe=1.
- Interpretation: the fixed seed1 improvement is no longer explained by no-op label/Instances rewrite; this run recorded actual CATF-v2 ROI/industrial augmentation.
- Reports:
  - `outputs/experiments/catf_v2_fixed_seed1_50ep/reports/final_report.md`
  - `outputs/experiments/catf_v2_fixed_seed1_50ep/reports/final_metrics.json`
  - `outputs/experiments/catf_v2_fixed_seed1_50ep/reports/compare_with_clean_native_seed1.md`

## CATF-v2 Strict No-Augmentation Bypass Fix (2026-06-03)

- No training was run.
- Fixed CATF-v2 formal transform no-augmentation paths so they return the native YOLO label object unchanged unless an industrial/ROI op is actually applied.
- Added `tests/test_catf_v2_transform_bypass.py`.
- Audit path: `outputs/audits/catf_v2_transform_parity/`.
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`.
- Re-ran `scripts/audit_catf_v2_transform_parity.py` on 100 deterministic train samples across:
  - clean native YOLO default transform output,
  - CATF-v2 noop transform output,
  - CATF-v2 formal path with all industrial/ROI ops force-skipped by zero probability.
- Result:
  - clean vs CATF-v2 noop final output: exact match.
  - clean vs CATF-v2 formal-force-skip final output: exact match.
  - bbox hash mismatch: 0/100.
  - formal-force-skip image/cls/Instances rewrites: 0/100.
  - router random draw count: 0.
  - applied industrial/ROI ops: 0.
  - router bbox_oob/invalid/class-oob count under force-skip: 0.
- The previous root cause was confirmed and removed: CATF-v2 formal transform used to convert, validate, clip, and rebuild labels/Instances even when no augmentation was selected. It now bypasses rewrite/validation when `applied_any_aug=false`.
- Validation:
  - `tests/test_catf_v2_transform_bypass.py` passed.
  - Full targeted suite passed: 81 tests.
- Reports:
  - `outputs/audits/catf_v2_transform_parity/transform_parity_report.md`
  - `outputs/audits/catf_v2_transform_parity/transform_parity.json`

## Formal Safe Tiled Baseline Result (2026-05-18)

- Run ID: `20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Safe tiled dataset: true; safe source config used `tile_size=1024`, `overlap=0.2`, `min_visibility=0.7`, `large_object_min_visibility=0.9`, `drop_border_truncated=True`, `border_margin=2.0`, `require_box_center_inside=True`, `keep_empty_ratio=0.1`.
- Removed classes: `OK`, `瀹氫綅`; retained classes: `OK2`, `OK3`, `鍔犲己绛嬫墦浼, `寮€瑁俙, `娌规薄`, `娴呭垝浼, `婕忚儗閿, `纰颁激`, `鑴忔薄`, `杞粨鍒掍激`, `閿′笣娈嬬暀`, `閿″皷`, `閿¤啅`.
- Train/val tiles: 2301 / 677; bboxes: 4087.
- Model: `yolo11n.pt`; epochs: 50; imgsz: 1024; batch: 2; device: 0; workers: 0.
- YOLO augmentation switches requested for shutdown were all set to `0`: `mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0`.
- Completed 50 epochs: true; OOM: false; training time: 2.826 hours.
- Validation metrics from `best.pt`: Precision=0.690, Recall=0.615, mAP50=0.669, mAP50-95=0.434.
- Lowest per-class Recall: `寮€瑁俙 (0.000).
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_report.md`.
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_metrics.json`.
- best.pt: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/best.pt`.
- last.pt: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/last.pt`.

## Implemented In This Update

- Added `scripts/filter_tiled_dataset.py` and built the filtered dataset at `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/`.
- The filtered dataset removes only `OK` and `瀹氫綅`, keeps `OK2` and `OK3`, and remaps class ids to `0..12` with Chinese names preserved.
- The filtered dataset reports:
  - source train/val images: 2452 / 677
  - filtered train/val images: 2301 / 677
  - source bbox count: 4269
  - filtered bbox count: 4087
  - class id out of range: false
  - Chinese class names damaged: false
  - formal baseline ready: true
- Added `scripts/audit_tiling_quality.py` and audited `outputs/datasets/tiled/tiled_1024_ov20_full/` for tile-boundary truncation risk.
- Marked `outputs/datasets/tiled/tiled_1024_ov20_full/` as unsafe for formal baseline use because it retains partial-object bboxes.
- Hardened `scripts/build_yolo_tiled_dataset.py` with safe tiling controls:
  - `--min-visibility` default changed to `0.7`
  - `--large-object-min-visibility` added with default `0.9`
  - `--drop-border-truncated` added with default `True`
  - `--border-margin` added with default `2`
  - `--require-box-center-inside` added with default `True`
  - `--classwise-visibility-config` added for optional per-class overrides
- Built the new safe full tiled dataset at `outputs/datasets/tiled/tiled_1024_ov20_full_safe/`.
- Safe dataset debug visualizations now include retained boxes and dropped border-truncated/visibility-risk boxes; 100 images were written under `outputs/datasets/tiled/tiled_1024_ov20_full_safe/debug_tiling/`.
- Built the full tiled dataset from `E:\TJGY\DataSet2_fixed` at `outputs/datasets/tiled/tiled_1024_ov20_full/` without `--max-images-per-split`.
- Expanded `scripts/build_yolo_tiled_dataset.py` reporting for full baseline readiness:
  - original and tiled train/val image counts
  - original and tiled bbox totals
  - per-class original/tiled train/val instance counts
  - bbox drop reasons
  - empty tile retention
  - `data.yaml` nc/names, class id range, class id out-of-range flags, and Chinese-name damage flags
- Changed full dataset debug output to 30 random tile-level bbox visualizations under `outputs/datasets/tiled/tiled_1024_ov20_full/debug_tiling/`.
- Updated `scripts/audit_dataset_mapping.py` to audit the full tiled dataset and write:
  - `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.md`
  - `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.json`
- Added `scripts/audit_dataset_mapping.py` and generated dataset mapping audit under `outputs/audits/dataset_mapping/`.
- Repaired `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml` so it fully inherits original class names from `E:\TJGY\DataSet2_fixed\data.yaml`.
- Updated `scripts/build_yolo_tiled_dataset.py` to write tiled `data.yaml` with `yaml.safe_dump(..., allow_unicode=True)` so non-ASCII class names are preserved.
- Added `scripts/audit_artifacts.py` and generated artifact inventory under `outputs/audits/artifact_inventory/`.
- Added `docs/output_convention.md` and normalized active artifacts into `outputs/datasets/`, `outputs/experiments/`, `outputs/audits/`, and `outputs/snapshots/`.
- Archived legacy root-level outputs to `outputs/archive/old_outputs_20260517/`.
- Archived legacy Ultralytics auto outputs from `runs/detect/*` to `outputs/archive/old_runs_20260517/runs_detect/`.
- Updated `scripts/run_gpu_preflight.py` to capture the active conda env, `sys.executable`, PyTorch/CUDA, Ultralytics, `yolo checks`, optional `nvidia-smi`, and conditional YOLO GPU smoke results without falling back to base PATH.
- Generated auditable GPU preflight reports:
  - `outputs/audits/gpu_preflight/gpu_preflight_report.md`
  - `outputs/audits/gpu_preflight/gpu_preflight_report.json`
- Added `scripts/build_yolo_tiled_dataset.py` for YOLO sliding-window tiled dataset construction.
- Added `diagnosis_vector` to `diagnosis.json` with normalized small-object, low-contrast, class-imbalance, localization, and false-positive scores plus calculation basis.
- Reworked `AutoAugment/diagnostic_pipeline/policy_mapping.py` from fixed issue rules to severity-score dynamic probability/strength formulas.
- Split proxy safety into explicit `original_bbox_retention`, `new_bbox_valid_rate`, `total_bbox_valid_rate`, `small_object_retention`, `SafetyScore`, and soft safety penalties.
- Added copy-paste filter audit artifacts and debug images under the proxy stage.
- Added `AutoAugment/diagnostic_pipeline/strategy_memory.py` with JSONL append and cosine-similarity reranking.
- Added metric consistency audit output under `metric_consistency/`.

## Verified Locally

- Formal safe tiled YOLO11n baseline training and validation completed for `20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep`.
- Filtered no-OK/no-position dataset build completed with no training:
  - Output dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/`
  - Deleted classes: `OK`, `瀹氫綅`
  - Kept classes: `OK2`, `OK3`, `鍔犲己绛嬫墦浼, `寮€瑁俙, `娌规薄`, `娴呭垝浼, `婕忚儗閿, `纰颁激`, `鑴忔薄`, `杞粨鍒掍激`, `閿′笣娈嬬暀`, `閿″皷`, `閿¤啅`
  - New class id mapping: `0..12` in the order above
  - Source train/val images: 2452 / 677
  - Filtered train/val images: 2301 / 677
  - Source bbox count: 4269
  - Filtered bbox count: 4087
  - Train bbox count before/after: 3337 / 3182
  - Val bbox count before/after: 932 / 905
  - Train empty tiles retained: 210
  - Val empty tiles retained: 83
  - Debug samples: 50
  - Class id out of range: false
  - Chinese class names damaged: false
  - Formal baseline readiness: true
- Tiling quality audit for the old full tiled dataset:
  - Report: `outputs/audits/tiling_quality/tiling_quality_audit.md`
  - JSON: `outputs/audits/tiling_quality/tiling_quality_audit.json`
  - Debug truncated bbox images: `outputs/audits/tiling_quality/debug_truncated_bboxes/` (50 images)
  - Total old full tiled bboxes: 7465
  - Border-truncated bboxes: 3197 (42.83%)
  - Visibility < 0.5: 1091
  - Visibility < 0.7: 1956
  - Visibility < 0.8: 2405
  - Visibility < 0.9: 2874
  - Bboxes touching tile boundary: 3249 (43.52%)
- Safe full tiled dataset build completed with no training:
  - Output dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/`
  - Parameters: `tile_size=1024`, `overlap=0.2`, `min_visibility=0.7`, `large_object_min_visibility=0.9`, `drop_border_truncated=True`, `border_margin=2`, `require_box_center_inside=True`, `keep_empty_ratio=0.1`, `seed=42`
  - Tiled images: 2452 train, 677 val
  - Original bboxes: 3084
  - Safe tiled bboxes: 4269
  - Dropped bbox candidates after tile intersection: 13826
  - Visibility-failed dropped candidates: 13346
  - Border-truncated dropped candidates: 3961
  - Center-outside dropped candidates: 9855
  - Obvious half-target bbox remains: false
  - Debug tile visualizations: 100
  - Class ids remain in range and Chinese names remain intact.
  - Note: class `瀹氫綅` has 0 retained bboxes under the strict large-structure rule; use this as an explicit audit caveat if that class must be evaluated.
- Full tiled dataset build completed with no training:
  - Command used no `--max-images-per-split`.
  - Source dataset: `E:\TJGY\DataSet2_fixed`
  - Output dataset: `outputs/datasets/tiled/tiled_1024_ov20_full/`
  - Parameters: `tile_size=1024`, `overlap=0.2`, `min_visibility=0.3`, `keep_empty_ratio=0.1`, `seed=42`
  - Original images: 461 train, 116 val
  - Tiled images: 4155 train, 1098 val
  - Original bboxes: 3084
  - Tiled bboxes: 7465
  - Empty tiles retained: 478
  - Dropped bboxes in retained tiles: 22202 (`below_min_visibility=6212`, `outside_tile=15990`)
  - Debug tile bbox visualizations: 30
  - `data.yaml`: `nc=15`, names inherited from original with Chinese names intact
  - Class ids: tiled min 0, max 14, no class id >= nc
  - Superseded status: this dataset is now marked unsafe for formal baseline use after the tiling quality audit found retained partial-object bboxes.
- Full tiled dataset mapping audit:
  - Report: `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.md`
  - JSON: `outputs/audits/dataset_mapping/full_tiled_dataset_mapping_audit.json`
  - Full source coverage confirmed: 461 train and 116 val source images.
  - Tiled names match original names exactly.
  - No class id >= nc and no negative class id found.
  - Chinese class names are not damaged.
- `python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['scripts/build_yolo_tiled_dataset.py','scripts/audit_dataset_mapping.py']]; print('syntax ok')"` passed.
- `pytest -q tests\test_build_yolo_tiled_dataset.py` passed: 2 tests.
- GPU environment confirmed in conda env `pytorch`:
  - Python executable: `D:\Anaconda\envs\pytorch\python.exe`
  - Python version: 3.9.19
  - PyTorch: 2.4.1
  - `torch.cuda.is_available()`: True
  - `torch.version.cuda`: 12.4
  - CUDA device count: 1
  - GPU: NVIDIA GeForce RTX 3060 Laptop GPU
  - Ultralytics: 8.3.221
  - `yolo checks`: passed
  - YOLO GPU smoke: passed with `model=yolo11n.pt`, `data=outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml`, `epochs=1`, `imgsz=640`, `batch=1`, `workers=0`, `device=0`, and YOLO built-in augmentations disabled
  - Formal training environment: conda env `pytorch`, YOLO `device=0`
- Latest GPU preflight reports are `outputs/audits/gpu_preflight/gpu_preflight_report.md` and `outputs/audits/gpu_preflight/gpu_preflight_report.json`.
- Earlier base-env preflight showed CPU-only PyTorch; base must not be used for formal training.
- Dataset mapping audit for the earlier smoke dataset:
  - Report: `outputs/audits/dataset_mapping/dataset_mapping_audit.md`
  - JSON: `outputs/audits/dataset_mapping/dataset_mapping_audit.json`
  - Original dataset: 461 train images, 116 val images, 2433 train bboxes, 651 val bboxes, class ids 0..14.
  - Tiled smoke dataset: 107 train tiles, 86 val tiles, 232 train bboxes, 143 val bboxes, class ids 0..14.
  - No class id >= nc found in original or tiled smoke labels.
  - Tiled smoke names now match original names exactly.
  - Tiled smoke is not a formal baseline dataset because it uses only 16 source images, matching the capped smoke build.
  - The 20 epoch `blank-or-unrendered` rows came from the earlier corrupted/non-renderable tiled class names in the saved val log/metrics, not from out-of-range class IDs.
  - Low mAP is mainly driven by the smoke split and class imbalance: OK and OK3 have high AP50, while 寮€瑁? 婕忚儗閿? 纰颁激, 杞粨鍒掍激, 閿′笣娈嬬暀, and 閿¤啅 have recall 0 in the smoke val metrics.
- Ran tiled baseline 20 epoch on GPU:
  - Command target: `outputs/experiments/20260517_tiled_baseline_20epoch`
  - Dataset: `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml`
  - Model: `yolo11n.pt`
  - Epochs: 20
  - imgsz: 1024
  - batch: 2
  - workers: 0
  - device: 0
  - YOLO built-in augmentations disabled: `mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0`
  - OOM: false
  - best.pt: `outputs/experiments/20260517_tiled_baseline_20epoch/train/weights/best.pt`
  - last.pt: `outputs/experiments/20260517_tiled_baseline_20epoch/train/weights/last.pt`
  - Precision: 0.828
  - Recall: 0.213
  - mAP50: 0.247
  - mAP50-95: 0.181
  - Report: `outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_report.md`
  - Metrics JSON: `outputs/experiments/20260517_tiled_baseline_20epoch/reports/baseline_20epoch_metrics.json`
- `pytest -q tests/test_build_yolo_tiled_dataset.py tests/test_copy_paste.py tests/test_proxy_prefilter.py tests/test_yolo_error_analysis.py` passed.
- `pytest -q` passed: 89 tests.
- Built tiled smoke dataset:
  - `outputs/datasets/tiled/tiled_1024_ov20_smoke`
  - `outputs/datasets/tiled/tiled_1024_ov20_smoke/data.yaml`
  - `outputs/datasets/tiled/tiled_1024_ov20_smoke/tiled_dataset_report.md`
  - `outputs/datasets/tiled/tiled_1024_ov20_smoke/debug_tiling/`
- Historical real tiled smoke pipeline was archived:
  - `outputs/archive/old_outputs_20260517/diagnostic_aug_tiled_smoke`
  - baseline epochs: 1
  - short epochs: 1
  - final epochs: 1
  - top-k: 1
  - workers: 0
  - device: cpu
  - final training skipped by request

## Smoke Conclusions

- `copy_paste` was not hard filtered. Three copy-paste policies passed the hard filter; ordinary bbox risks were recorded as soft safety penalties.
- Metric consistency audit found no precision/recall delta for this smoke because both YOLO val and diagnosis were zero-recall at the chosen thresholds, while also documenting why mAP and single-threshold TP/FP/FN are not expected to match in general.
- Strategy memory appended the smoke record to the archived `outputs/archive/old_outputs_20260517/strategy_memory.jsonl`; no prior similar cases existed for rerank boost on this run.

## Operational Notes

- Formal training must use conda env `pytorch` with `D:\Anaconda\envs\pytorch\python.exe`.
- Formal YOLO training must use GPU `device=0`.
- Formal runs must use explicit `--run-id` and `project=outputs/experiments/<run_id>`; do not rely on `runs/detect`.
- Unsafe full tiled dataset path: `outputs/datasets/tiled/tiled_1024_ov20_full/`; do not use it for formal baseline because partial-object bboxes were retained.
- Active safe full tiled dataset path: `outputs/datasets/tiled/tiled_1024_ov20_full_safe/`.
- Active filtered formal baseline dataset path: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/`.
- Formal baseline training has completed on `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`; primary result is `outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_report.md`.
- CPU is allowed only for smoke/debug runs and must not be treated as formal experiment output.
- Do not use the base conda environment for formal training; it previously resolved to CPU-only PyTorch.
- Windows YOLO commands should keep `workers=0`.
- This machine needed `KMP_DUPLICATE_LIB_OK=TRUE` for the CPU YOLO smoke because the active Anaconda environment initialized duplicate OpenMP runtimes.
- The smoke used a capped tiled dataset (`--max-images-per-split 8`) to keep runtime bounded; it is not a final benchmark.

<!-- DIAGAUG_50EP_START -->
## Diagnosis-Driven Augmentation 50 Epoch Result

- Run ID: `20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Baseline best.pt: `E:/TJGY/MinPaper/MyAutoAugument/outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/train/weights/best.pt`
- Selected policy: `diag_policy_001` from `low_contrast_missed_defect`
- Selected policy contains copy_paste: `false`
- Copy-paste candidates retained in proxy ranking: `2`
- Copy-paste hard rejected: `false`
- Augmented train images / bboxes: `4602` / `6364`
- Precision: `0.686` (-0.004 vs baseline)
- Recall: `0.688` (+0.073 vs baseline)
- mAP50: `0.717` (+0.048 vs baseline)
- mAP50-95: `0.496` (+0.062 vs baseline)
- OOM: `false`
- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep\train\weights\best.pt`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/diagaug_50ep_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/diagaug_50ep_metrics.json`
- Baseline comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/baseline_vs_diagaug.md`
<!-- DIAGAUG_50EP_END -->

<!-- YOLO_DEFAULT_AUG_50EP_START -->
## YOLO Default Augmentation 50 Epoch Control

- Run ID: `20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Model/settings: `yolo11n.pt epochs=50 imgsz=1024 batch=2 workers=0 device=0`
- YOLO default augmentations enabled; actual args recorded from `train/args.yaml`.
- Precision: `0.785` (+0.095 vs baseline, +0.099 vs DiagAug)
- Recall: `0.676` (+0.061 vs baseline, -0.012 vs DiagAug)
- mAP50: `0.735` (+0.066 vs baseline, +0.018 vs DiagAug)
- mAP50-95: `0.476` (+0.042 vs baseline, -0.020 vs DiagAug)
- OOM: `false`
- Training wall time: `13154.0s (3.65h)`
- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep\train\weights\best.pt`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/yolo_default_aug_50ep_report.md`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/compare_baseline_yolo_default_diagaug.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/yolo_default_aug_50ep_metrics.json`
<!-- YOLO_DEFAULT_AUG_50EP_END -->

<!-- BASELINE_POLICY_TOP3_START -->
## Baseline Diagnosis Top3 Candidate Policies

- Run ID: `20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline`
- Baseline best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep\train\weights\best.pt`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Triggered issues: `low_contrast_missed_defect, low_contrast_missed_defect, class_imbalance, localization_bias`
- Candidate policies generated: `5`
- Top3 proxy policies: `diag_policy_001, diag_policy_005, diag_policy_002`
- Top3 containing copy_paste: `diag_policy_005`
- Scope: single-round baseline diagnosis only; not multi-round optimization.
- Training status: no YOLO train, no final 50 epoch train, no top3 short-training.
- Next step for final strategy selection: run short-training for all top3 and select by short_train_score.
- Trace report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/reports/policy_selection_trace.md`
- Top3 report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/top3_policies/top3_policies.md`
- Proxy ranking: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/proxy/proxy_ranking.json`
<!-- BASELINE_POLICY_TOP3_END -->

<!-- TOP3_POLICY_SHORTTRAIN_START -->
## Top3 Policy Short-Training Validation

- Run ID: `20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain`
- Source top3 run: `20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline`
- Scope: top3 short-training strategy validation only; not a final model result.
- Each policy trained for `5` epochs with YOLO built-in augmentations disabled.
- Top3 came from one baseline diagnosis and proxy/safety ranking, not multi-round closed-loop search.
- Short-training scores: `diag_policy_001=0.609204, diag_policy_005=0.578862, diag_policy_002=0.595412`
- Best short-training policy: `diag_policy_001`
- Matches current formal DiagAug policy: `true`
- Result should decide whether a formal DiagAug 50 epoch rerun is needed.
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain/reports/top3_policy_shorttrain_report.md`
- Results JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain/reports/top3_policy_shorttrain_results.json`
<!-- TOP3_POLICY_SHORTTRAIN_END -->

<!-- COUNTERFACTUAL_DIAGNOSIS_START -->
## Counterfactual Diagnosis for Baseline Missed Defects

- Run ID: `20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis`
- Scope: prediction-only counterfactual diagnosis; no training, no 50 epoch run, no top3 short-training.
- Purpose: validate whether baseline FN cases respond to low-contrast/brightness-style transforms.
- Baseline FN count: `215`
- Tested FN count: `200`
- Highest recovery transform: `sharpen_mild` recovery_rate=`0.0700`
- diag_policy_001 unique photometric FN recovery rate: `0.1050`
- Supports low_contrast_missed_defect -> diag_policy_001: `True`
- copy_paste is not directly testable by prediction-only counterfactual inference.
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/reports/counterfactual_diagnosis_report.md`
- Summary JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/counterfactual_summary.json`
- Instance table: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis/counterfactual_instances.csv`
<!-- COUNTERFACTUAL_DIAGNOSIS_END -->

<!-- CLASS_AWARE_DIAGNOSIS_START -->
## Class-Aware Per-Class Diagnosis and Policy Generation

- Run ID: `20260518_tiled1024_safe_no_ok_position_per_class_diagnosis`
- Scope: upgraded diagnosis and policy generation only; no training, no 50 epoch run, no short-training.
- Method upgrade: global policy selection -> class-aware error attribution policy generation.
- Inputs: baseline 50 epoch metrics, baseline diagnosis, and counterfactual diagnosis.
- Policy generated: `class_aware_policy_001` with branches `photometric_branch, copy_paste_branch, texture_branch, localization_branch`
- final_policy_score: `0.570452`
- Next step: run short-training for `class_aware_policy_001` before any formal 50 epoch rerun.
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_per_class_diagnosis/reports/per_class_diagnosis_report.md`
- Policy: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_per_class_diagnosis/policies/class_aware_mixed_policy.json`
- Score: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_per_class_diagnosis/policies/class_aware_policy_score.json`
<!-- CLASS_AWARE_DIAGNOSIS_END -->

<!-- CLASS_AWARE_POLICY_SHORTTRAIN_START -->
## Class-Aware Policy Short-Training Validation

- Run ID: `20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain`
- Scope: top1 class-aware mixed policy short-training only; no formal 50 epoch training.
- Policy: `class_aware_policy_001`
- Train images / bboxes: `4602` / `6412`
- Precision: `0.626`
- Recall: `0.680`
- mAP50: `0.685`
- mAP50-95: `0.456`
- short_train_score: `0.603477`
- Beats diag_policy_001 short-training score `0.609204`: `false`
- Recommend formal 50 epoch rerun: `false`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain/reports/class_aware_shorttrain_report.md`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain/reports/class_aware_vs_diag_policy_001_shorttrain.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain/metrics/class_aware_shorttrain_metrics.json`
<!-- CLASS_AWARE_POLICY_SHORTTRAIN_END -->

<!-- RANDOM_EXTERNAL_AUG_50EP_START -->
## Random External Augmentation 50 Epoch Control

- Run ID: `20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep`
- Scope: random external augmentation control group; not diagnosis-driven.
- Training set: original train images + 1x random augmented train images.
- YOLO built-in augmentations: disabled to match DiagAug final training.
- Random policy: `random_external_policy_seed42` with ops `sharpen(p=0.238,s=0.343), brightness(p=0.681,s=0.414), cutout(p=0.151,s=0.140), horizontal_flip(p=0.448,s=1.000)`
- Train images / bboxes: `4602` / `6364`
- Safety: hard_filter_pass=`true`, bbox_valid_rate=`1.000000`
- Precision: `0.750`
- Recall: `0.668`
- mAP50: `0.734`
- mAP50-95: `0.501`
- Delta vs DiagAug: P `+0.064`, R `-0.020`, mAP50 `+0.017`, mAP50-95 `+0.005`
- OOM: `false`
- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep\train\weights\best.pt`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/reports/random_external_aug_50ep_report.md`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/reports/compare_baseline_yolo_default_diagaug_random.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/reports/random_external_aug_50ep_metrics.json`
<!-- RANDOM_EXTERNAL_AUG_50EP_END -->

<!-- DIAGNOSIS_GUIDED_POLICY_SEARCH_START -->
## Diagnosis-Guided Policy Search

- Run ID: `20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search`
- Scope: diagnosis-guided sampled policy search plus 5 epoch short-training; no formal 50 epoch training.
- Change in method: diagnosis adjusts operation sampling probabilities instead of directly selecting a fixed policy.
- Candidate policies: `30`
- Proxy pass count: `20`
- Short-training trials: `10`
- Best balanced-score policy: `search_policy_017` balanced=0.632550 P/R/mAP50/mAP50-95=0.697/0.691/0.720/0.468
- Beats diag_policy_001 short-training balanced score: `true`
- Recommend formal 50 epoch: `true`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/reports/policy_search_report.md`
- Results JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/reports/policy_search_results.json`
- Best summary: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search/reports/best_policy_summary.md`
<!-- DIAGNOSIS_GUIDED_POLICY_SEARCH_END -->

<!-- SEARCH_POLICY_017_50EP_START -->
## search_policy_017 Formal 50 Epoch Result

- Run ID: `20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep`
- Scope: diagnosis-guided policy search winner promoted to formal 50 epoch YOLO training.
- Training set: original train images + 1x `search_policy_017` augmented train images.
- YOLO built-in augmentations: disabled.
- Policy: `gaussian_noise(p=0.189, s=0.104), gamma(p=0.475, s=0.456), local_contrast(p=0.371, s=0.179), cutout(p=0.142, s=0.163), copy_paste(p=0.621, s=0.273)`
- Train images / bboxes: `4602` / `6408`
- Safety: bbox_valid_rate=`0.999380`, image_failures=`0`, label_failures=`0`
- Precision: `0.710`
- Recall: `0.616`
- mAP50: `0.681`
- mAP50-95: `0.474`
- Balanced score: `0.608450`
- Delta vs baseline: P `+0.020`, R `+0.001`, mAP50 `+0.012`, mAP50-95 `+0.040`
- Delta vs diag_policy_001: P `+0.024`, R `-0.072`, mAP50 `-0.036`, mAP50-95 `-0.022`
- Delta vs random external: P `-0.040`, R `-0.052`, mAP50 `-0.053`, mAP50-95 `-0.027`
- Exceeds random external by mAP50-95: `false`
- Exceeds diag_policy_001 by mAP50-95: `false`
- Current best formal by mAP50-95: `false`
- Current best formal by balanced score: `false`
- OOM: `false`
- Training wall seconds: `34573.7`
- best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep\train\weights\best.pt`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/reports/search_policy_017_50ep_report.md`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/reports/compare_baseline_diagaug_random_yolo_search017.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep/reports/search_policy_017_50ep_metrics.json`
<!-- SEARCH_POLICY_017_50EP_END -->

<!-- ONLINE_AUG_SMOKE_START -->
## Online Policy Augmentation Smoke

- Run ID: `online_yolo_like_base_smoke`
- New entrypoint: `scripts/train_yolo_online_aug.py`.
- Method change: custom policy is applied dynamically inside the YOLO training dataloader instead of building a fixed offline augmented dataset.
- Train image count: `2301`; no train image doubling.
- Fixed augmented dataset generated: `false`
- YOLO built-in augmentation mode for this smoke: disabled, so this is `only_custom_online_aug`.
- Online copy-paste: pending object-bank implementation; copy_paste ops are skipped safely for now.
- 1 epoch smoke train success: `true`
- 1 epoch smoke val success: `true`
- Val P/R/mAP50/mAP50-95: `0.4922/0.2364/0.1763/0.0986`
- Preview dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\online_yolo_like_base_smoke\previews`
- Report: `outputs/experiments/online_yolo_like_base_smoke/reports/online_aug_smoke_report.md`
- Stats JSON: `outputs/experiments/online_yolo_like_base_smoke/reports/online_aug_stats.json`
- Next step: inspect smoke safety/history and tune the feedback controller before any formal 50 epoch experiment.
<!-- ONLINE_AUG_SMOKE_END -->

<!-- ONLINE_DIAG_POLICY_001_50EP_START -->
## Online Diag Policy 001 50 Epoch

- Run ID: `20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep`
- Entrypoint: `scripts/train_yolo_online_aug.py`.
- Mechanism: custom policy is sampled online in the YOLO training dataloader; no fixed augmented dataset is built.
- Train image count: `2301`; no train image doubling.
- Fixed augmented dataset generated: `false`
- Validation custom augmentation: `false`; val uses original val tiles.
- YOLO built-in augmentation: disabled for `only_custom_online_aug`.
- Online copy-paste: pending object-bank implementation; copy_paste ops are skipped safely.
- Train success: `true`
- Val success: `true`
- Val P/R/mAP50/mAP50-95: `0.7297/0.6772/0.6814/0.4607`
- Online better than offline DiagAug by mAP50-95: `false`
- Online close to YOLO default by mAP50-95 within 0.03: `true`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/online_diag_policy_001_50ep_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/online_diag_policy_001_50ep_metrics.json`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/compare_online_offline_yolo_default_random.md`
- Stats JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/online_aug_stats.json`
<!-- ONLINE_DIAG_POLICY_001_50EP_END -->

<!-- ONLINE_RANDOM_LIKE_50EP_START -->
## Online Random-Like 50 Epoch

- Run ID: `20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep`
- Entrypoint: `scripts/train_yolo_online_aug.py`.
- Policy: online random-like mix of `sharpen_mild`, `brightness`, `cutout_safe`, and `horizontal_flip`.
- Mechanism: policy is sampled online in the YOLO training dataloader; no fixed augmented dataset is built.
- Train image count: `2301`; no train image doubling.
- Fixed augmented dataset generated: `false`
- Validation custom augmentation: `false`; val uses original val tiles.
- YOLO built-in augmentation: disabled for `only_custom_online_aug`.
- Online copy-paste: disabled.
- Train success: `true`
- Val success: `true`
- Val P/R/mAP50/mAP50-95: `0.7132/0.6641/0.6859/0.4661`
- Online random-like better than offline random by mAP50-95: `false`
- Online random-like better than online DiagAug by mAP50-95: `true`
- Random external advantage source: `operator_combo_helps_but_offline_doubling_or_training_variance_still_contributes`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/reports/online_random_like_50ep_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/reports/online_random_like_metrics.json`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/reports/compare_online_random_like_with_all.md`
- Stats JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_random_like_yolo11n_50ep/reports/online_aug_stats.json`
<!-- ONLINE_RANDOM_LIKE_50EP_END -->

<!-- FEEDBACK_ONLINE_AUG_SMOKE_START -->
## Feedback Online Augmentation Smoke

- Run ID: `feedback_online_policy_2stage_smoke`
- Entrypoint: `scripts/train_yolo_online_aug.py` with `--feedback-enabled`.
- Mechanism: custom YOLO-like/industrial online augmentation remains inside the training dataloader; no fixed augmented dataset is built.
- Stage count: `2`
- Policy history updates: `1`
- Train image count: `2301`; no train image doubling.
- Fixed augmented dataset generated: `false`
- Validation custom augmentation: `false`; val uses original val tiles.
- YOLO built-in augmentation: disabled for `only_custom_online_aug`.
- Online copy-paste: pending object-bank implementation; feedback may raise pending copy-paste probabilities but execution is skipped safely.
- Train success: `true`
- Val success: `true`
- Val P/R/mAP50/mAP50-95: `0.6540/0.3189/0.3338/0.2071`
- Report: `outputs/experiments/feedback_online_policy_2stage_smoke/reports/online_aug_smoke_report.md`
- Stats JSON: `outputs/experiments/feedback_online_policy_2stage_smoke/reports/online_aug_stats.json`
- Policy history JSON: `outputs/experiments/feedback_online_policy_2stage_smoke/reports/policy_history.json`
<!-- FEEDBACK_ONLINE_AUG_SMOKE_END -->

<!-- CUSTOM_YOLO_LIKE_BASE_50EP_START -->
## Custom YOLO-Like Base 50 Epoch

- Run ID: `20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep`
- Entrypoint: `scripts/train_yolo_online_aug.py`.
- Policy: `configs/online_policies/yolo_like_base_policy.json`.
- Mechanism: custom YOLO-like operators are sampled online in the YOLO training dataloader; no fixed augmented dataset is built.
- Train image count: `2301`; no train image doubling.
- Fixed augmented dataset generated: `false`
- Validation custom augmentation: `false`; val uses original val tiles.
- YOLO built-in augmentation: disabled for `only_custom_online_aug`.
- Feedback applied: `false`; policy history records no feedback applied.
- close_mosaic active: `true`
- Train success: `true`
- Val success: `true`
- Val P/R/mAP50/mAP50-95: `0.6661/0.7490/0.7129/0.4415`
- Close to YOLO default by mAP50-95 within 0.03: `false`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/custom_yolo_like_base_50ep_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/custom_yolo_like_base_50ep_metrics.json`
- Comparison: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/compare_custom_yolo_like_with_yolo_default.md`
- Stats JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/online_aug_stats.json`
- Policy history JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep/reports/policy_history.json`
<!-- CUSTOM_YOLO_LIKE_BASE_50EP_END -->

<!-- BEGIN YOLO_DEFAULT_DIAGNOSIS_CONSTRAINED_50EP -->
## YOLO Default Diagnosis-Constrained 50 Epoch

- Run ID: `20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep`
- Objective: keep Ultralytics YOLO default augmentation enabled and add only constrained diagnostic online augmentation.
- Groups: YOLO default, diagnosis_light, diagnosis_precision_safe, diagnosis_recall_safe, plus old custom_yolo_like_base control.
- All A-D groups use `yolo11n.pt`, tiled safe no-OK/no-position data, `epochs=50`, `imgsz=1024`, `batch=2`, `workers=0`, `device=0`, `seed=42`.
- YOLO built-in augmentation: enabled for A-D; custom diagnosis policies are added online in the dataloader.
- Fixed augmented dataset generated: `false`.
- YOLO default reference P/R/mAP50/mAP50-95: `0.7132/0.7600/0.7759/0.5241`
- Best under industrial constraints: `YOLO default`
- Recall improved while constraints hold: `false`
- Failure driver if no improvement: `diagnosis_light: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis; diagnosis_precision_safe: FP increased by 18; diagnosis_precision_safe: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis; diagnosis_recall_safe: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis`
- Report: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/diagnosis_constrained_experiment_report.md`
- Metrics JSON: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/diagnosis_constrained_metrics.json`
- Constraint scoring: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/constraint_scoring.json`
<!-- END YOLO_DEFAULT_DIAGNOSIS_CONSTRAINED_50EP -->

<!-- YOLO_DEFAULT_FEEDBACK_AUG_SMOKE_START -->
## YOLO Default Feedback Augmentation Smoke

- Entrypoint: `scripts/train_yolo_default_with_feedback.py`.
- Base: Ultralytics YOLO default augmentation remains enabled; custom YOLO-like `mosaic4` and `randaugment_like` are not used.
- Scope: 2-stage smoke when `epochs=2 feedback_interval=1`; no formal 50 epoch run in this step.
- Output: `outputs/experiments/yolo_default_feedback_aug_50ep/`
- Stage count: `2`
- Policy history updates: `1`
- Fixed augmented dataset generated: `false`
- Final P/R/mAP50/mAP50-95: `0.5259/0.3311/0.3109/0.1961`
- Constraint accepted: `false`
- Report: `outputs/experiments/yolo_default_feedback_aug_50ep/reports/yolo_default_feedback_smoke_report.md`
- Policy history: `outputs/experiments/yolo_default_feedback_aug_50ep/reports/policy_history.json`
<!-- YOLO_DEFAULT_FEEDBACK_AUG_SMOKE_END -->

<!-- YOLO_DEFAULT_FEEDBACK_AUG_50EP_FULL_START -->
## YOLO Default Feedback Augmentation

- Entrypoint: `scripts/train_yolo_default_with_feedback.py`.
- Base: Ultralytics YOLO default augmentation remains enabled; custom YOLO-like `mosaic4` and `randaugment_like` are not used.
- Scope: formal 50 epoch segmented feedback run.
- Output: `outputs/experiments/yolo_default_feedback_aug_50ep_full/`
- Stage count: `10`
- Policy history updates: `9`
- Fixed augmented dataset generated: `false`
- Final P/R/mAP50/mAP50-95: `0.7712/0.6689/0.7439/0.4993`
- Constraint accepted: `false`
- Report: `outputs/experiments/yolo_default_feedback_aug_50ep_full/reports/final_report.md`
- Policy history: `outputs/experiments/yolo_default_feedback_aug_50ep_full/reports/policy_history.json`
<!-- YOLO_DEFAULT_FEEDBACK_AUG_50EP_FULL_END -->
<!-- YOLO_DEFAULT_INLOOP_FEEDBACK_SMOKE_START -->
## YOLO Default In-Loop Feedback / Control

- Entrypoint: `scripts/train_yolo_default_with_inloop_feedback.py`.
- The previous `yolo_default_feedback_aug_50ep_full` run is a segmented fine-tune experiment, not strict continuous feedback.
- New direction: one `YOLO.train()` run with in-loop feedback callbacks; optimizer/scheduler/EMA/epoch/close_mosaic remain under one Ultralytics trainer.
- Feedback controller: `CATF` (Constraint-Aware Trust-region Feedback Controller).
- CATF uses the clean native YOLO default reference curve at matching feedback epochs, trust-region step limits, group budgets, delayed acceptance, rollback, cooldown, and epoch>=40 freeze.
- CATF-v1 is global feedback; CATF-v2 is class-aware, issue-aware, and sample-aware feedback with ROI-aware industrial augmentation.
- CATF-v2 current goal is to reduce CATF-v1 Precision instability by activating only diagnosed classes and freezing stable classes.
- Current CATF-v2 work is smoke-only; no formal 50 epoch CATF-v2 run should be inferred from it.
- No-feedback control disables both feedback and industrial augmentation, using Ultralytics YOLO default augmentation as the behavior check.
- The old YOLO default reference is not the final baseline after parity audit; feedback comparisons should use `clean_native_yolo_default_seed42_50ep`.
- Output: `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/`
- Epochs: `50`
- Feedback enabled: `true`
- Industrial augmentation enabled: `true`
- CATF version: `v2`
- Class-aware feedback: `true`
- ROI-aware augmentation: `true`
- Sample-aware routing: `true`
- Reference curve loaded: `true`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Stage restart count: `0`
- Epoch continuous: `true`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- Constraint baseline: `clean_native_yolo_default`
- Constraint failed: `True`
- Report: `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/reports/final_report.md`
- Policy history: `outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun/reports/policy_history.json`
<!-- YOLO_DEFAULT_INLOOP_FEEDBACK_SMOKE_END -->
<!-- YOLO_DEFAULT_INLOOP_PARITY_AUDIT_START -->
## YOLO Default In-Loop Parity Audit

- Scope: parity audit only; no new 50 epoch feedback or augmentation experiment was run.
- Reference run: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/runs/yolo_default_seed42/`.
- In-loop no-feedback control: `outputs/experiments/yolo_default_inloop_no_feedback_control_50ep/`.
- Finding: the selected reference run used `OnlineAugDetectionTrainer` with an empty passthrough policy, so it is not a pure native YOLO CLI/Python baseline.
- Finding: the completed in-loop control used default trainer by command, but the old script still attached a no-op in-loop callback, as shown by `epoch_records.json`.
- Args diff: data path absolute vs relative, plus project/save_dir; augmentation args and validation args matched.
- close_mosaic: both runs used `close_mosaic=10` and both logs triggered `Closing dataloader mosaic`.
- Repair: `feedback=false` and `industrial_aug=false` now enters a native passthrough branch with no custom trainer, dataset, transform, or feedback callback.
- 1 epoch parity smoke: passed; native Python API and repaired in-loop no-feedback had identical args except output paths, identical loss/metric/lr values, and zero in-loop callback records.
- Report: `outputs/audits/yolo_default_inloop_parity/parity_audit_report.md`
- JSON: `outputs/audits/yolo_default_inloop_parity/parity_audit.json`
<!-- YOLO_DEFAULT_INLOOP_PARITY_AUDIT_END -->
<!-- CLEAN_NATIVE_YOLO_DEFAULT_REFERENCE_START -->
## Clean Native YOLO Default Reference

- The old YOLO default reference is no longer treated as the final baseline because parity audit found it used `OnlineAugDetectionTrainer` with an empty passthrough policy.
- New baseline: `outputs/experiments/clean_native_yolo_default_seed42_50ep/`.
- Training mode: pure native Ultralytics `YOLO.train(**same_args)`.
- Custom trainer / callback / dataset / transform / industrial augmentation: `false`.
- Train image count: `2301`
- Precision/Recall/mAP50/mAP50-95: `0.7262/0.6844/0.7616/0.5250`
- close_mosaic official schedule expected: `true`
- close_mosaic expected start epoch: `41`
- Future feedback experiments should compare only against this clean native reference.
- Report: `outputs/experiments/clean_native_yolo_default_seed42_50ep/reports/clean_native_yolo_default_report.md`
- Metrics JSON: `outputs/experiments/clean_native_yolo_default_seed42_50ep/reports/clean_native_yolo_default_metrics.json`
<!-- CLEAN_NATIVE_YOLO_DEFAULT_REFERENCE_END -->

<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_INLOOP_FEEDBACK_START -->
## Multiseed Clean YOLO Default vs In-Loop Feedback

- Scope: seeds `0, 1, 2`; seed 42 is not included in the multiseed mean.
- Output: `outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback/`.
- Clean group uses pure native Ultralytics `YOLO.train`; feedback group uses single-run in-loop feedback with YOLO default augmentation still enabled.
- Train images: `2301`; fixed augmented dataset generated: `false`; copy_paste remains pending/not enabled.
- Feedback wins under industrial constraints: `1/3`.
- Constraint failed seeds: `2/3`.
- Mean delta P/R/mAP50/mAP50-95: `-0.0275/0.0254/0.0093/0.0211`.
- Verdict: not stable enough to claim as the paper main result yet; use as diagnostic/ablation unless a stricter controller passes multiseed constraints.
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback/reports/multiseed_summary.md`.
- JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback/reports/multiseed_summary.json`.
<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_INLOOP_FEEDBACK_END -->

<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_CATF_FEEDBACK_START -->
## Multiseed Clean YOLO Default vs CATF Feedback

- Scope: seeds `0, 1, 2`; seed 42 is retained as a positive single-seed case but is not included in this multiseed mean.
- Output: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_feedback/`.
- Clean native results were reused from `outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback/seed_*/clean_native_yolo_default/`; no duplicate clean native training was run.
- CATF group uses single-run in-loop feedback with YOLO default augmentation still enabled, per-seed clean native reference curves, industrial online augmentation enabled, and copy_paste pending/not enabled.
- Train images: `2301`; fixed augmented dataset generated: `false`; val uses original val tiles.
- Per-seed deltas P/R/mAP50/mAP50-95:
  - seed 0: `-0.0272/-0.0014/+0.0179/+0.0445`, constraint_failed=`true`.
  - seed 1: `-0.0349/+0.0621/+0.0008/+0.0386`, constraint_failed=`true`.
  - seed 2: `-0.0116/-0.0120/-0.0084/-0.0424`, constraint_failed=`true`.
- Mean delta P/R/mAP50/mAP50-95: `-0.0246/+0.0163/+0.0034/+0.0136`.
- CATF wins under industrial constraints: `0/3`; constraint failed seeds: `3/3`.
- Control statistics across seeds: rollback `17`, cooldown `3`, freeze `3`; frozen policy records `15`.
- Compared with old in-loop feedback, CATF is more conservative in logs but not more stable by the industrial constraint criterion: old feedback failed `2/3`, CATF failed `3/3`.
- Verdict: CATF is not recommended as the paper main method based on seeds `0/1/2`; keep seed42 as a positive case and report this as an ablation/controller attempt unless a later controller passes multiseed constraints.
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_feedback/reports/multiseed_catf_summary.md`.
- JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_feedback/reports/multiseed_catf_summary.json`.
<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_CATF_FEEDBACK_END -->

<!-- CATF_V2_ACTIVATION_AUDIT_START -->
## CATF-v2 Activation Audit

- Scope: `outputs/experiments/catf_v2_class_aware_10ep_smoke/reports/`.
- No training was run; this is a report-only audit of epoch 5 CATF-v2 activation.
- Audit outputs: `outputs/experiments/catf_v2_class_aware_10ep_smoke/reports/catf_v2_activation_audit.md` and `.json`.
- Active classes from smoke: class `1` OK3, class `6` 婕忚儗閿? class `8` 鑴忔薄.
- OK3 activation is judged not reasonable for a formal run: Recall is already high (`0.9891`), FN count is only `2`, FP count is `49`, and OK-like classes should default to stable/no_aug unless evidence is very strong.
- 婕忚儗閿?activation is judged reasonable: low Recall (`0.2826`), many FN (`32`), low-contrast evidence, and conservative ROI `sharpen_mild`/`local_contrast` ops.
- 鑴忔薄 activation is partially reasonable as low Recall, but should be guarded by stain/dirty high-FP domain priors and should not lower threshold or escalate photometric before FP behavior is known.
- Low-support classes `2` and `3` did not trigger strong photometric augmentation; they remain oversampling/copy-paste pending candidates.
- ROI augmentation applied `150` times, including OK3 (`122`), which is the main activation-rule concern.
- Recommendation: do not enter CATF-v2 seed42 50ep until activation rules are tightened with OK2/OK3 no_aug, stronger activation threshold, domain high-FP guards for stain/oil/dirty classes, and likely top_k reduced from `3` to `2`.
<!-- CATF_V2_ACTIVATION_AUDIT_END -->

<!-- CATF_V2_ACTIVATION_FIXED_START -->
## CATF-v2 Activation Rule Fix

- Scope: no 50 epoch training; only rule changes, tests, and a 10 epoch smoke run.
- Output: `outputs/experiments/catf_v2_activation_fixed_10ep_smoke/`.
- Report: `outputs/experiments/catf_v2_activation_fixed_10ep_smoke/reports/catf_v2_activation_fixed_report.md`.
- Rule changes: OK2/OK3 default no_aug, stricter activation thresholds, domain high-FP prior for OK2/OK3/oil/dirty classes, top_k_active_classes=`2`, ROI blocks no_aug/high-FP conflict classes.
- Smoke result: train_success=`true`, val_success=`true`, train_images=`2301`, fixed_augmented_dataset_generated=`false`.
- Active classes after fix: class `6` 婕忚儗閿?(texture_boundary_weak), class `8` 鑴忔薄 (low_recall).
- OK3 active=`false`; OK3 ROI applied=`0`.
- 婕忚儗閿?active=`true` with conservative ROI sharpen/local_contrast.
- 鑴忔薄 domain_high_fp_prior=`true`; photometric probs `{'clahe': 0.0, 'gamma': 0.0, 'brightness': 0.0, 'contrast': 0.0}`; threshold recommendation `0.25` with reason `domain_high_fp_prior_keep_threshold`.
- ROI stats: `{'roi_aug_applied': 23, 'roi_aug_skipped_small_roi': 0, 'roi_aug_skipped_conflict': 2, 'affected_classes': {'6': 20, '8': 3}}`.
- BBox/class valid: `true`.
- Recommendation: proceed to CATF-v2 seed42 50 epoch validation only after this fixed activation rule set; do not use the earlier CATF-v2 smoke as formal evidence.
<!-- CATF_V2_ACTIVATION_FIXED_END -->

<!-- CATF_V2_SEED42_50EP_START -->
## CATF-v2 Seed42 50 Epoch

- Output: `outputs/experiments/catf_v2_seed42_50ep/`.
- Entry: `scripts/train_yolo_default_with_inloop_feedback.py` with `--catf-version v2`, class-aware feedback, ROI-aware augmentation, sample-aware routing, threshold calibration report, top_k=`2`, top_m=`2`.
- Training mode: single-run continuous YOLO default training with official YOLO augmentation kept enabled; no stage restart; no self-implemented mosaic/randaugment replacement.
- Train images: `2301`; fixed augmented dataset generated: `false`; copy_paste remains `pending_object_bank_design`.
- Epoch continuity: `1..50` continuous.
- Final metrics P/R/mAP50/mAP50-95: `0.7498/0.7257/0.7679/0.5212`.
- Delta vs clean native seed42 P/R/mAP50/mAP50-95: `+0.0236/+0.0413/+0.0062/-0.0039`.
- constraint_failed: `false`.
- OK2/OK3 active epochs: `[]`; OK3 ROI applied: `0`.
- Active class counts: `{'6:婕忚儗閿?: 1, '12:閿¤啅': 1}`.
- ROI stats: `{'roi_aug_applied': 37, 'roi_aug_skipped_small_roi': 0, 'roi_aug_skipped_conflict': 0, 'affected_classes': {'6': 18, '12': 19}}`.
- Feedback actions: `{'observe': 2, 'accept': 1, 'shrink': 4, 'freeze': 2}`; class actions: `{'propose': 2, 'observe': 3}`; rollback/cooldown/freeze: `0/0/4`.
- Report: `outputs/experiments/catf_v2_seed42_50ep/reports/final_report.md`.
- Metrics JSON: `outputs/experiments/catf_v2_seed42_50ep/reports/final_metrics.json`.
- Policy history: `outputs/experiments/catf_v2_seed42_50ep/reports/policy_history.json`.
- Best checkpoint: `outputs/experiments/catf_v2_seed42_50ep/train/weights/best.pt`.
- Verdict: seed42 passes industrial constraints and is suitable for CATF-v2 multiseed validation; do not claim final method before multiseed passes.
<!-- CATF_V2_SEED42_50EP_END -->

<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_CATF_V2_START -->
## Multiseed Clean YOLO Default vs CATF-v2

- Scope: seeds `0, 1, 2`; seed 42 remains a positive single-seed validation and is not included in the multiseed mean.
- Output: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/`.
- Clean native results were reused from existing per-seed clean YOLO default runs; CATF-v2 was newly trained for each seed.
- CATF-v2 settings: official YOLO default augmentation kept enabled, class-aware feedback, ROI-aware augmentation, sample-aware routing, threshold calibration report, top_k=`2`, top_m=`2`, feedback interval=`5`.
- Train images: `2301`; fixed augmented dataset generated: `false`; copy_paste remains `pending_object_bank_design`.
- Per-seed deltas P/R/mAP50/mAP50-95:
  - seed 0: `-0.0417/+0.0218/+0.0223/+0.0264`, constraint_failed=`true` due Precision drop.
  - seed 1: `-0.0034/+0.0474/+0.0007/+0.0099`, constraint_failed=`false`.
  - seed 2: `+0.1395/-0.1247/-0.0269/-0.0285`, constraint_failed=`true` due mAP50 and mAP50-95 drops.
- Mean delta P/R/mAP50/mAP50-95: `+0.0315/-0.0185/-0.0013/+0.0026`.
- CATF-v2 wins under industrial constraints: `1/3`; constraint failed seeds: `2/3`.
- OK2/OK3 were never active; OK3 ROI applied total: `0`.
- Active class counts: `{'8:鑴忔薄': 1, '11:閿″皷': 1}`; ROI affected totals: `{'8:鑴忔薄': 3, '11:閿″皷': 20}`.
- Control statistics across seeds: rollback `0`, cooldown `0`, freeze events `12`; policy actions `{'shrink': 14, 'accept': 2, 'observe': 5, 'freeze': 6}`.
- Compared with CATF-v1, CATF-v2 improves activation discipline and reduces constraint failures from `3/3` to `2/3`, but is still not stable enough for the paper main method.
- Report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/multiseed_catf_v2_summary.md`.
- JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/multiseed_catf_v2_summary.json`.
<!-- MULTISEED_CLEAN_YOLO_DEFAULT_VS_CATF_V2_END -->

<!-- CATF_V2_FAILURE_MODE_ANALYSIS_START -->
## CATF-v2 Multiseed Failure-Mode Analysis

- Scope: analysis only; no training was run and no CATF-v2 rules were changed.
- Source experiment: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/`.
- Generated reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/curve_diagnosis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/active_class_effect_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/precision_drop_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/recall_drop_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/threshold_calibration_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/controller_behavior_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/catf_v2_failure_mode_summary.md`
- Main failure modes: `intervention too weak`, `intervention wrong target`, `precision threshold issue`, `over-conservative freeze`, `per-class diagnosis trajectory sensitive`, and `in-loop diagnosis/callback side effect not isolated`.
- Seed 0 failure: Recall/mAP improved, but Precision failed due approximate FP increases across multiple classes, led by oil/dirty-like and defect classes; threshold calibration is the most direct repair candidate.
- Seed 1 pass: no ROI/industrial augmentation was applied, so the pass is not causal evidence for ROI augmentation; it may include in-loop diagnosis/callback RNG effects.
- Seed 2 failure: Precision increased while Recall/mAP dropped, indicating conservative confidence/detection behavior on a clean seed that already had high Recall.
- ROI-aware augmentation was too sparse to prove benefit: only `23` ROI applications across seeds `0/1/2`.
- Controller behavior: shrink-dominant (`14` shrink vs `2` accept), zero rollback/cooldown, and freeze at epoch 40/45; negative-effect attribution is missing.
- Recommendation: do not claim CATF-v2 as paper main method yet; next step should be diagnosis-only in-loop control plus per-class threshold calibration analysis before more 50 epoch training.
<!-- CATF_V2_FAILURE_MODE_ANALYSIS_END -->

<!-- THRESHOLD_CALIBRATION_DIAGNOSIS_ONLY_START -->
## CATF-v2 Threshold Calibration and Diagnosis-Only Control

- Scope: no new 50 epoch training and no CATF-v2 rule changes.
- Post-hoc threshold report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/threshold_calibration_posthoc.md`.
- Post-hoc JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/reports/threshold_calibration_posthoc.json`.
- Prediction-only inputs: cached/generated validation predictions for clean native and CATF-v2 seeds `0/1/2`; no training was run.
- Threshold grid: per-class confidence threshold `0.10..0.70` step `0.05`; objectives `constrained_score`, `balanced_score`, `industrial_score`.
- Constrained post-hoc result: CATF-v2 passes industrial constraints for `2/3` seeds after calibration.
- Seed 0: Precision/constraint failure is repairable by threshold calibration in the post-hoc evaluator; precision-oriented objectives raise thresholds for OK/oil/dirty-like and stable classes while lowering difficult defect classes.
- Seed 2: Recall/mAP failure is not repaired by threshold lowering; this remains a training/trajectory degradation, not a pure confidence-threshold issue.
- Diagnosis-only smoke: `outputs/experiments/diagnosis_only_inloop_control_10ep_smoke/`.
- Diagnosis-only plan: `outputs/experiments/diagnosis_only_inloop_control_plan.md`.
- Smoke result: diagnosis callback executed at epoch `5`; industrial samples augmented=`0`, ROI applied=`0`, policy update applied=`0`, train images=`2301`, fixed augmented dataset generated=`false`, epoch sequence `1..10` continuous, bbox/class legal.
- Recommendation: run a later diagnosis-only 50ep control before claiming CATF-v2 gains are caused by industrial augmentation rather than callback/RNG/training-path effects.
<!-- THRESHOLD_CALIBRATION_DIAGNOSIS_ONLY_END -->

<!-- DIAGNOSIS_ONLY_CONTROL_50EP_START -->
## Diagnosis-Only In-Loop Control 50 Epoch

- Scope: seeds `0, 1, 2`; this is the control for in-loop diagnosis callback/RNG/training-path effects.
- Output: `outputs/experiments/diagnosis_only_inloop_control_50ep/`.
- Entry: `scripts/train_yolo_default_with_inloop_feedback.py`.
- Configuration: YOLO default augmentation enabled, feedback diagnosis enabled, `diagnosis_only=true`, `industrial_aug_enabled=false`, ROI-aware augmentation disabled, sample-aware routing disabled, threshold mutation disabled, copy_paste not enabled.
- Train images: `2301`; fixed augmented dataset generated: `false`; results.csv epoch `1..50` continuous for all seeds.
- Control counters for all seeds: industrial samples augmented=`0`, ROI applied=`0`, policy update applied=`0`, bbox/class legal=`true`.
- Final diagnosis-only metrics P/R/mAP50/mAP50-95:
  - seed 0: `0.7846/0.6765/0.7347/0.4759`.
  - seed 1: `0.7725/0.6477/0.7542/0.4799`.
  - seed 2: `0.6962/0.7286/0.7692/0.5224`.
- Delta vs clean native for all seeds and all four metrics: `0.0000`; diagnosis-only constraint pass count: `3/3`.
- Interpretation: the diagnosis callback alone did not change training results. Seed 1 CATF-v2 success is not explained by callback/RNG alone, though CATF-v2 industrial-enabled training-path differences still need caution because that seed reported zero actual industrial/ROI augmentation.
- CATF-v2 + post-hoc threshold calibration retains independent value as a deployment/post-processing companion: pass count improves from `1/3` to `2/3`, but seed 2 remains unrepaired.
- Summary report: `outputs/experiments/diagnosis_only_inloop_control_50ep/reports/diagnosis_only_multiseed_summary.md`.
- Summary JSON: `outputs/experiments/diagnosis_only_inloop_control_50ep/reports/diagnosis_only_multiseed_summary.json`.
<!-- DIAGNOSIS_ONLY_CONTROL_50EP_END -->

<!-- CATF_V2_NOOP_PARITY_AUDIT_START -->
## CATF-v2 No-op Parity Audit

- Scope: no new CATF-v2 strategy training; added explicit `--catf-noop` audit mode and ran parity controls.
- Output smoke: `outputs/experiments/catf_v2_noop_parity_smoke/`.
- Output 50ep control: `outputs/experiments/catf_v2_noop_control_50ep/seed_1/`.
- `--catf-noop` behavior: CATF-v2 custom trainer/dataset/router objects are built, but the online transform returns labels unchanged before bbox conversion, sample router is not called, router random draws=`0`, ROI applied=`0`, industrial samples augmented=`0`, and policy update applied=`0`.
- 1ep parity smoke: clean native vs CATF-v2 noop metrics/loss/lr/results.csv numeric fields are identical except wall-clock `time`; args.yaml differs only in `project` and `save_dir`.
- 1ep smoke report: `outputs/experiments/catf_v2_noop_parity_smoke/reports/noop_parity_report.md`.
- Random path audit: `outputs/experiments/catf_v2_noop_parity_smoke/reports/random_path_audit.md`.
- Seed1 50ep CATF-v2 noop metrics P/R/mAP50/mAP50-95: `0.7725/0.6477/0.7542/0.4799`.
- Delta seed1 noop vs clean native: `0.0000/0.0000/0.0000/0.0000`; constraint_failed=`false`.
- Delta seed1 noop vs CATF-v2: `+0.0034/-0.0474/-0.0007/-0.0099`.
- Interpretation: diagnosis-only already showed the callback itself is neutral; CATF-v2 noop now shows the custom CATF-v2 framework path is also neutral when augmentation/policy mutation are hard-disabled. Seed1 CATF-v2 gain is therefore not explained by no-op framework perturbation, but still cannot be attributed to ROI augmentation for that seed because actual ROI/industrial counters were zero.
- Seed1 noop report: `outputs/experiments/catf_v2_noop_control_50ep/reports/noop_control_seed1_report.md`.
<!-- CATF_V2_NOOP_PARITY_AUDIT_END -->

<!-- FIXED_CATF_V2_SEED2_THRESHOLD_ANALYSIS_START -->
## Fixed CATF-v2 Seed2 Failure and Threshold Calibration Analysis

- Scope: analysis only; no training was run and CATF-v2 rules were not changed.
- Source experiment: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/`.
- Analysis script: `scripts/analyze_fixed_catf_v2_seed2_and_thresholds.py`.
- Generated reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed2_failure_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_threshold_calibration_posthoc.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_vs_old_catf_v2_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_next_step_summary.md`
- Fixed multiseed status before post-hoc calibration: seed0 and seed1 pass constraints; seed2 fails with Precision up but Recall/mAP down.
- Seed2 fixed CATF-v2 vs clean native delta P/R/mAP50/mAP50-95: `+0.0674/-0.0423/-0.0110/-0.0257`.
- Seed2 main failure mode: CATF-v2 shifted an already high-recall clean seed toward fewer, more precise detections; active ROI targets did not fully match the final degraded classes.
- Seed2 Recall drop classes: `浅划伤`, `轮廓划伤`, `加强筋打伤`, `漏背锡`, `锡膏`.
- Seed2 AP/mAP drag classes include `锡丝残留`, `脏污`, `锡膏`, `轮廓划伤`, `开裂`, `加强筋打伤`.
- Seed2 active classes: class `9` 轮廓划伤, class `11` 锡尖, class `8` 脏污; ROI applied totals: class `9`=`25`, class `11`=`59`, class `8`=`6`.
- Post-hoc per-class threshold calibration: seed2 is repairable in the analysis evaluator (`constraint_failed=false`, Recall delta `+0.0667`, mAP50 delta `+0.0452`, mAP50-95 delta `+0.0112`).
- Calibrated fixed CATF-v2 constraint pass count remains `2/3` under the post-hoc evaluator because seed0 still violates the mAP50-95 tolerance.
- Recommendation: fixed CATF-v2 is a strong candidate, but not yet a finalized stable paper main method. Next implementation work should prioritize class-level rollback, negative-effect attribution, high-recall baseline protection, and a more conservative threshold-calibration objective.
<!-- FIXED_CATF_V2_SEED2_THRESHOLD_ANALYSIS_END -->

<!-- CATF_V2_RC_SEED0_CALIBRATION_ROLLBACK_START -->
## CATF-v2-RC Seed0 Calibration and Rollback Analysis

- Scope: analysis only; no training was run and CATF-v2 augmentation rules were not rewritten.
- Source experiment: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/`.
- Added formal deployment/post-processing helper: `AutoAugment/catf_v2/per_class_thresholds.py`.
- Analysis script: `scripts/refine_fixed_catf_v2_seed0_calibration_and_rollback.py`.
- New reports:
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed0_failure_analysis.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/seed0_threshold_calibration_posthoc.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_threshold_calibration_all_seeds.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_class_level_rollback_simulation.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_final_candidate_plan.md`
  - `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/fixed_catf_v2_ready_for_paper_summary.md`
- Seed0 status: official fixed CATF-v2 seed0 already passes constraints; the remaining seed0 issue was the earlier post-hoc evaluator/threshold-search mAP50-95 failure.
- Seed0 RC calibration result in post-hoc evaluator: P/R/mAP50/mAP50-95 `0.6636/0.7915/0.6737/0.4303`, deltas vs clean default evaluator `+0.0397/+0.0269/+0.0038/-0.0099`, constraint_failed=`false`.
- RC threshold calibration pass count: `3/3` seeds in the post-hoc evaluator.
- Recommended per-class thresholds: OK2 `0.25`, OK3 `0.10`, 加强筋打伤 `0.25`, 开裂 `0.25`, 油污 `0.10`, 浅划伤 `0.10`, 漏背锡 `0.10`, 碰伤 `0.10`, 脏污 `0.10`, 轮廓划伤 `0.10`, 锡丝残留 `0.25`, 锡尖 `0.25`, 锡膏 `0.10`.
- Rollback simulation buckets: keep fixed CATF-v2 for `油污`, `锡尖`; threshold calibration only for `脏污`; rollback candidates `加强筋打伤`, `锡丝残留`; high-risk negative-effect classes `轮廓划伤`, `锡膏`.
- CATF-v2-RC definition: fixed CATF-v2 + per-class constrained threshold calibration + class-level rollback/high-recall protection.
- Caveat: RC calibration metrics are post-hoc prediction-evaluator metrics and must be confirmed against the official validation/export path before final paper claims.
<!-- CATF_V2_RC_SEED0_CALIBRATION_ROLLBACK_END -->

<!-- CATF_V2_RC_OFFICIAL_PATH_VALIDATION_START -->
## CATF-v2-RC Official Val/Predict Path Validation

- Scope: validation/inference only; no training was run.
- Script: `scripts/validate_catf_v2_rc_official_path.py`.
- Threshold config validated: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_per_class_thresholds.json`.
- Output report: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_official_path_validation.md`.
- Output JSON: `outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/catf_v2_rc_official_path_validation.json`.
- Fixed CATF-v2 best checkpoints for seeds `0/1/2` were rerun through Ultralytics `YOLO.val` and `YOLO.predict(conf=0.10)`.
- Official Ultralytics val confirms fixed CATF-v2 training metrics:
  - seed0 fixed vs clean delta P/R/mAP50/mAP50-95: `-0.0060/-0.0068/+0.0090/+0.0136`.
  - seed1 fixed vs clean delta P/R/mAP50/mAP50-95: `+0.0127/+0.0528/+0.0284/+0.0390`.
  - seed2 fixed vs clean delta P/R/mAP50/mAP50-95: `+0.0674/-0.0423/-0.0110/-0.0257`.
- Official predict + per-class threshold post-processing with the saved unified RC threshold table passes only `1/3` seeds, not `3/3`.
- RC failures:
  - seed0 fails Precision after threshold lowering: delta P/R/mAP50/mAP50-95 `-0.0666/+0.0321/+0.0054/-0.0095`.
  - seed2 fails Precision after threshold lowering: delta P/R/mAP50/mAP50-95 `-0.0148/+0.0655/+0.0435/+0.0096`.
  - seed1 passes with delta `+0.0496/+0.0501/+0.0395/+0.0289`.
- Interpretation: the previous post-hoc per-seed/RC search remains useful, but the single robust threshold table is too recall-aggressive for seed0 and seed2. Do not claim CATF-v2-RC is 3/3 ready until the official path threshold table is re-optimized or made seed/model-specific with stronger Precision guard.
<!-- CATF_V2_RC_OFFICIAL_PATH_VALIDATION_END -->

<!-- CP_CATF_OFFLINE_CAUSAL_PROBE_START -->
## CP-CATF Offline Causal Probe

- Scope: strategy-selection implementation and offline analysis only; no training was run.
- Added module: `AutoAugment/catf_v2/causal_probe.py`.
- Added offline runner: `scripts/run_catf_v2_offline_causal_probe.py`.
- RiskGuard update: the fixed class-op registry is downgraded to an audit/debug prior. It is no longer a default training-time blacklist and cannot be the final accept/reject reason for CATF policy selection.
- Motivation: seed2 showed that a diagnosis trigger is not equivalent to augmentation benefit; class-local fixes such as RiskGuard can recover the audited class but still fail overall due non-active class regression.
- CP-CATF rule: candidate policies are evaluated by run-specific causal probe metrics before entering the sample router. Data-specific outcomes are probe results, not hard-coded method rules.
- Development-mode caveat: the offline probe uses existing validation diagnostics for mechanism checking (`development_probe_uses_existing_val_diagnostics=true`). Paper-mode CP-CATF must use a train/probe split or train hard examples for policy selection.
- Offline decisions:
  - seed0 selects `candidate_policy_1_roi_texture` / `accept`.
  - seed1 selects `candidate_policy_1_roi_texture` / `accept`.
  - seed2 rejects image-space candidates and selects `candidate_policy_3_sampler_only` with image modification disabled and sample weighting marked pending.
- Same generic candidate `candidate_policy_1_roi_texture` is accepted for seeds `0/1` and rejected for seed `2`, based on run metrics rather than seed id, class id, or dataset category name.
- Reports:
  - `outputs/experiments/catf_v2_causal_probe/reports/offline_causal_probe_summary.md`
  - `outputs/experiments/catf_v2_causal_probe/reports/offline_causal_probe_summary.json`
  - `outputs/experiments/catf_v2_causal_probe/reports/cp_catf_training_plan.md`
- Verification: `python -m py_compile AutoAugment/catf_v2/causal_probe.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py AutoAugment/catf_v2/high_risk_class_ops.py scripts/run_catf_v2_offline_causal_probe.py`; targeted pytest list passed `132 passed`.
- Next step: run CP-CATF training validation only when explicitly requested. CP-CATF can be a paper main-method candidate only after probe-gated training confirms seed0/seed1 retain gains while seed2 is protected.
<!-- CP_CATF_OFFLINE_CAUSAL_PROBE_END -->

<!-- CP_CATF_PAPER_MODE_VALIDATION_START -->
## CP-CATF Paper-Mode Probe Split Validation

- Scope: leakage-controlled CP-CATF validation using a train/probe split; final val is used only for final metrics.
- New split root: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/`.
- Split seed: `2026`; original train images `2301`; train_core `2071`; probe `230`; final val `677`; train_core/probe/final-val overlap count `0`.
- Split reports:
  - `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/reports/probe_split_report.md`
  - `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/reports/probe_split_report.json`
- Paper-mode implementation:
  - Added `--paper-probe-mode`, `--probe-data`, `--train-core-data`, `--probe-source`, and `--forbid-final-val-policy-selection`.
  - Policy selection source is recorded as `probe_split`.
  - Final val diagnostics are forbidden for candidate decisions in paper mode.
  - Leakage audit confirms final val was not used for policy selection and all split image sets are disjoint.
- Smoke run: `outputs/experiments/cp_catf_paper_mode_10ep_smoke/`; seed2 10ep passed, used train_core for train and probe split for causal probe, selected `candidate_policy_3_sampler_only`, strict image no-op with industrial samples `0`, ROI `0`, router random draws `0`.
- Full paper-mode multiseed run: `outputs/experiments/multiseed_cp_catf_paper_mode/`.
- Clean paper baseline metrics:
  - seed0: P/R/mAP50/mAP50-95 `0.7513/0.6763/0.7566/0.5114`.
  - seed1: `0.7220/0.7582/0.7777/0.5251`.
  - seed2: `0.6290/0.6385/0.6590/0.4381`.
- CP-CATF paper-mode metrics:
  - seed0: `0.7513/0.6763/0.7566/0.5114`, `constraint_failed=false`.
  - seed1: `0.7220/0.7582/0.7777/0.5251`, `constraint_failed=false`.
  - seed2: `0.6290/0.6385/0.6590/0.4381`, `constraint_failed=false`.
- Mean CP-CATF paper-mode delta vs clean paper baseline: dP/dR/dmAP50/dmAP50-95 `+0.0000/+0.0000/+0.0000/+0.0000`.
- Candidate behavior:
  - seed0 mostly selected `candidate_policy_3_sampler_only`; one probe point accepted `candidate_policy_1_roi_texture`.
  - seed1 mostly selected `candidate_policy_3_sampler_only`; two probe points accepted `candidate_policy_1_roi_texture`.
  - seed2 alternated between sampler-only and roi_texture accept after epoch 20.
  - Actual industrial samples augmented, ROI applied, and router random draw count were `0` for all CP-CATF paper-mode seeds, so results reproduce the clean paper baseline.
- Outcome: `3/3` constraint pass and no final-val leakage, but no retained gain over clean paper baseline.
- Interpretation: paper-mode CP-CATF is a valid leakage-control/safety validation, not a final main-result candidate yet. Development-mode CP-CATF remains a feasibility result; paper-mode needs stronger train/probe evidence, larger probe split, or train hard-example probe design before claiming CP-CATF as the paper main method.
- Reports:
  - `outputs/experiments/cp_catf_paper_mode_10ep_smoke/reports/paper_mode_smoke_report.md`
  - `outputs/experiments/cp_catf_paper_mode_10ep_smoke/reports/paper_mode_smoke_report.json`
  - `outputs/experiments/multiseed_cp_catf_paper_mode/reports/multiseed_cp_catf_paper_mode_summary.md`
  - `outputs/experiments/multiseed_cp_catf_paper_mode/reports/multiseed_cp_catf_paper_mode_summary.json`
- Verification: requested py_compile checks passed; requested pytest suite passed `142 passed`.
<!-- CP_CATF_PAPER_MODE_VALIDATION_END -->

<!-- CP_CATF_ACCEPT_TO_EXECUTION_AUDIT_START -->
## CP-CATF Paper-Mode Accept-to-Execution Audit

- Scope: audit and minimal implementation fix only; no 50ep rerun was performed.
- Audit reports:
  - `outputs/experiments/multiseed_cp_catf_paper_mode/reports/cp_catf_accept_to_execution_audit.md`
  - `outputs/experiments/multiseed_cp_catf_paper_mode/reports/cp_catf_accept_to_execution_audit.json`
- Existing paper-mode multiseed audit found `8` causal-probe accept events, but `0` accept events produced executable active class-op policy entries.
- The break was between accepted causal-probe candidate and policy materialization:
  - accepted `candidate_policy_1_roi_texture` was recorded,
  - candidate ops were only used as a whitelist,
  - no target class/op probability was injected into `policy_matrix_epoch_*_after_causal_probe.json`,
  - `class_policy_history.json` had no active classes,
  - sample router had no eligible active policy,
  - industrial samples, ROI applied, and router random draws stayed `0`.
- This was not caused by final-val leakage, RiskGuard blocking, sampler-only globally disabling augmentation, or `--industrial-aug-enabled` being dropped.
- Fix: accepted causal-probe image candidates now materialize into an active target class with executable op probabilities/strengths before sample routing; rejected and sampler-only candidates remain strict image no-op.
- Fixed 10ep smoke:
  - Run root: `outputs/experiments/cp_catf_paper_mode_execution_fixed_10ep_smoke/`.
  - Seed `1`, 10 epochs, paper-mode source `probe_split`.
  - Final-val leakage remained `false` and all split overlaps remained `0`.
  - The only probe event selected `candidate_policy_3_sampler_only`, so no image candidate accept occurred in this short smoke.
  - Industrial samples augmented `0`, ROI applied `0`, router random draw count `0`; this is expected for sampler-only/no-op and should not be reported as applied augmentation.
- Smoke reports:
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_10ep_smoke/reports/execution_fixed_smoke_report.md`
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_10ep_smoke/reports/execution_fixed_smoke_report.json`
- Verification: requested py_compile checks passed; targeted pytest suite passed `74 passed`.
- Current interpretation: the prior paper-mode multiseed is leakage-free no-op safety validation, not proof of effective paper-mode enhancement. The accept-to-execution bug is fixed in code/tests, but a new 50ep paper-mode multiseed rerun is needed before making paper-mode CP-CATF performance claims.
<!-- CP_CATF_ACCEPT_TO_EXECUTION_AUDIT_END -->

<!-- CP_CATF_SEED0_EXECUTION_VALIDATION_START -->
## CP-CATF Paper-Mode Seed0 Execution Validation

- Scope: seed0 only; no clean rerun, no seed1/seed2, and no multiseed summary.
- Run root: `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/`.
- Purpose: verify that paper-mode CP-CATF accepted image candidates actually execute after the accept-to-execution fix.
- Reused clean paper seed0 baseline: P/R/mAP50/mAP50-95 `0.751343/0.676301/0.756646/0.511423`.
- Seed0 CP-CATF paper-mode result: `0.739931/0.676311/0.759345/0.502791`.
- Delta vs reused clean seed0: dP/dR/dmAP50/dmAP50-95 `-0.011412/+0.000010/+0.002699/-0.008631`.
- Constraint result: `constraint_failed=true` due Precision drop greater than `0.01`.
- Execution flow:
  - epoch 25 accepted `candidate_policy_1_roi_texture` for class `9`.
  - executable policy was generated with `sharpen_mild(prob=0.20,strength=0.22)` and `local_contrast(prob=0.18,strength=0.20)`.
  - sample router executed; router random draw count `1330`.
  - industrial samples augmented `226`.
  - ROI applied `312`, all recorded for class `9`.
  - final-val leakage remained `false`; train_core/probe/final-val overlap stayed `0`.
- Reports:
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_execution_flow_report.md`
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_execution_flow_report.json`
- Interpretation: the paper-mode accept-to-execution chain is now functionally connected. This run is not evidence of final CP-CATF paper-mode performance because seed0 fails the industrial Precision constraint. Seed1/seed2 should not be launched as a performance validation until this seed0 precision risk is reviewed.
<!-- CP_CATF_SEED0_EXECUTION_VALIDATION_END -->

<!-- CP_CATF_SEED0_PRECISION_RISK_AUDIT_START -->
## CP-CATF Paper-Mode Seed0 Precision-Risk Audit

- Scope: analysis only; no training, no seed1/seed2, no multiseed, and no clean rerun.
- Script: `scripts/audit_seed0_cp_catf_precision_risk.py`.
- Reports:
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_precision_risk_audit.md`
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/seed0_precision_risk_audit.json`
- Prediction caches:
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/precision_risk_predictions/clean_final_val_predictions.json`
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/precision_risk_predictions/cp_final_val_predictions.json`
  - `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/reports/precision_risk_predictions/cp_probe_predictions.json`
- Main finding: seed0 paper-mode roi_texture execution caused a small global Precision constraint failure, but the main source is non-active class FP spillover, not class9 itself.
- At conf `0.25`, matched final-val FP changed from `344` to `354` (`+10`), TP changed `+3`, and FN changed `-3`.
- Precision-drop / FP-increase drivers:
  - class5: FP `3 -> 17` (`+14`), metric Precision `-0.1481`.
  - class6: FP `33 -> 45` (`+12`), metric Precision `-0.1056`.
  - class8: FP `16 -> 22` (`+6`), metric Precision `-0.0721`.
  - class2: FP `1 -> 4` (`+3`), metric Precision `-0.1217`.
- Class9 was active and ROI-affected, but class9 metric Precision/AP improved and matched FP decreased `50 -> 38`; it is not the primary Precision-drop class.
- Threshold calibration:
  - Final-val diagnostic calibration can restore Precision but is leakage-only.
  - Probe-based threshold selected on probe did not restore final-val Precision within clean-minus-0.01.
- Interpretation: paper-mode causal probe accepted class9 because local benefit was positive, but it underweighted non-active FP spillover and operating-point Precision risk.
- Recommendation: add a precision-aware accept gate before continuing paper-mode seed1/seed2 performance validation.
<!-- CP_CATF_SEED0_PRECISION_RISK_AUDIT_END -->

<!-- IMAGE_ONLY_WEAK_AUG_MULTISEED_SANITY_START -->
## Image-Only Weak Augmentation Seed0/Seed1 Sanity

- Date: `2026-06-14`.
- Scope: image-only weak CP-CATF sanity validation for seed0 and seed1 only; seed2 reused the completed image-only weak augmentation result.
- No clean rerun, no seed2 rerun, no sampler_only, no weighted index list, no sampling change, no gate change, no attenuation-ratio change, and no data-split change were performed.
- Method: `candidate_policy_1b_weak_roi_texture`, attenuation ratio `0.25`, single retained low-risk ROI texture op per accepted weak candidate, strict no-op otherwise.
- Sampler-only status: demoted to engineering exploration/ablation and not used in this mainline validation.
- Seed0 run: `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed0/`.
  - Weak image augmentation executed: `true`; industrial images augmented `16`; ROI applied `18`; router random draw count `341`.
  - `sampler_only_enabled=false`, `weighted_index_list_enabled=false`, `sampled_distribution_changed=false`.
  - Metrics: P/R/mAP50/mAP50-95 `0.679043/0.711821/0.722170/0.476005`.
  - Delta vs clean seed0: `-0.105511/+0.035364/-0.012525/+0.000111`.
  - Delta vs fixed CATF-v2 seed0: `-0.099463/+0.042167/-0.021487/-0.013536`.
  - Constraint: `constraint_failed=true`, reasons `precision_drop_gt_0.01` and `map50_drop_gt_0.01`.
- Seed1 run: `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed1/`.
  - Weak image augmentation executed: `true`; industrial images augmented `32`; ROI applied `38`; router random draw count `775`.
  - `sampler_only_enabled=false`, `weighted_index_list_enabled=false`, `sampled_distribution_changed=false`.
  - Metrics: P/R/mAP50/mAP50-95 `0.765605/0.712074/0.776844/0.506815`.
  - Delta vs clean seed1: `-0.006920/+0.064394/+0.022653/+0.026896`.
  - Delta vs fixed CATF-v2 seed1: `-0.019594/+0.011564/-0.005773/-0.012064`.
  - Constraint: `constraint_failed=false`; fixed CATF-v2 mAP50-95 gain is not fully retained within 0.01.
- Seed2 reused run: `outputs/experiments/catf_v2_image_only_weak_aug_seed2_50ep/`.
  - Metrics: P/R/mAP50/mAP50-95 `0.753254/0.694235/0.772718/0.515138`.
  - Delta vs clean seed2: `+0.057015/-0.034395/+0.003515/-0.007233`.
  - Constraint: `constraint_failed=false`; recall warning remains because Recall is below clean by more than 0.01.
- Three-seed summary:
  - Constraint pass count: `2/3`; `3/3 pass=false`.
  - Mean weak metrics: P/R/mAP50/mAP50-95 `0.732634/0.706043/0.757244/0.499319`.
  - Mean delta vs clean: `-0.018472/+0.021788/+0.004548/+0.006591`.
  - Mean delta vs fixed CATF-v2: `-0.043154/+0.020541/-0.004263/-0.002382`.
  - Total industrial images augmented `128`; total ROI applied `151`.
  - Sampler-only involved: `false`; weighted index list involved: `false`; sampled distribution changed: `false`; final val used for policy selection: `false`.
- Interpretation: image-only weak augmentation repaired seed2 and seed1 passes constraints, but seed0 fails hard constraints. This exact weak augmentation setting is not yet a 3-seed paper main-method result.
- Next image-only direction: add a recall/precision-aware image augmentation safety layer or refine weak augmentation acceptance so seed0 does not trade away Precision/mAP50; do not use sampler_only or sampling reweighting to repair the main result.
- Reports:
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed0/reports/seed0_weak_image_aug_report.md`
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed1/reports/seed1_weak_image_aug_report.md`
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/reports/weak_image_aug_multiseed_summary.md`
  - `outputs/experiments/catf_v2_image_only_weak_aug_multiseed/reports/weak_image_aug_multiseed_summary.json`
- Verification:
  - Requested py_compile checks passed for the training entry, causal probe, policy matrix, and sample router.
  - Requested targeted pytest suite passed `55 passed`.
  - New summary script `scripts/summarize_weak_image_aug_multiseed.py` passed py_compile.
<!-- IMAGE_ONLY_WEAK_AUG_MULTISEED_SANITY_END -->
