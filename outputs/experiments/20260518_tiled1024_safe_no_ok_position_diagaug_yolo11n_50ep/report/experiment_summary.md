# Experiment Summary

This project is positioned as a validation-error diagnostic-driven data augmentation framework for YOLO training.
It does not modify the YOLO Backbone, Neck, or Head.

## Selected Policy

- Policy: diag_policy_001
- Source issues: low_contrast_missed_defect

## Diagnosis
- TP: 668
- FP: 172
- FN: 215
- Precision: 0.7952380952380952
- Recall: 0.7381215469613259

## Method Comparison

| Method | mAP50 | mAP50-95 | Precision | Recall | small_object_recall | FP | FN | training_cost | policy_generation_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | skipped | None |
| Fixed Augment | TBD | TBD | TBD | TBD | TBD | TBD | TBD | not_run_in_pipeline | None |
| Random Augment | TBD | TBD | TBD | TBD | TBD | TBD | TBD | not_run_in_pipeline | None |
| Diagnosis-driven Augment | TBD | TBD | TBD | TBD | TBD | TBD | TBD | skipped | None |
