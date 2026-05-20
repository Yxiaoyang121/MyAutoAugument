# Proxy / Safety Ranking

- No YOLO training was run.
- Ranking uses proxy_score and safety_score only.
- Hard rejection is limited to severe structural errors: class id out-of-range, invalid bbox, or image save/read failure.
- copy_paste is not hard rejected for slight bbox safety or retention weakness; such risks are soft penalties.

| rank | policy_id | source_issue | contains_copy_paste | proxy_score | safety_score | combined_score | hard_filter_pass | soft_penalties |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | diag_policy_001 | low_contrast_missed_defect | False | 0.9386 | 1.0000 | 0.9570 | True |  |
| 2 | diag_policy_005 | combined | True | 0.9373 | 1.0000 | 0.9561 | True |  |
| 3 | diag_policy_002 | low_contrast_missed_defect | False | 0.9337 | 1.0000 | 0.9536 | True |  |
| 4 | diag_policy_004 | localization_bias | False | 0.9007 | 1.0000 | 0.9305 | True |  |
| 5 | diag_policy_003 | class_imbalance | True | 0.8976 | 1.0000 | 0.9283 | True |  |
