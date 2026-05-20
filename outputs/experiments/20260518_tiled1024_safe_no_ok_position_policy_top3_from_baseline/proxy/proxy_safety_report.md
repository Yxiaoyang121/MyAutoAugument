# Proxy Safety Report

## Filter Contract

- Severe hard reject only: class id out-of-range, invalid bboxes, or image save/read failure.
- Soft penalty only: bbox_safe_rate, original bbox retention, new bbox valid rate, total bbox valid rate, small object retention, exposure, class distribution drift, and strength.
- copy_paste policies are retained unless they trigger a severe structural error.

## Per-Policy Safety

| rank | policy_id | bbox_valid_rate | original_bbox_retention | new_bbox_valid_rate | total_bbox_valid_rate | small_object_retention | exposure_score | class_distribution_change | hard_filter_pass | hard_filter_reasons |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | diag_policy_001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | True |  |
| 2 | diag_policy_005 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0755 | True |  |
  - copy_paste_audit for `diag_policy_005`: new_bbox_count=`74`, new_bbox_valid_rate=`1.0000`, total_bbox_valid_rate=`1.0000`, class_out_of_range_count=`0`, hard_rejected=`false`.
| 3 | diag_policy_002 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | True |  |
| 4 | diag_policy_004 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | True |  |
| 5 | diag_policy_003 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0622 | True |  |
  - copy_paste_audit for `diag_policy_003`: new_bbox_count=`78`, new_bbox_valid_rate=`1.0000`, total_bbox_valid_rate=`1.0000`, class_out_of_range_count=`0`, hard_rejected=`false`.
