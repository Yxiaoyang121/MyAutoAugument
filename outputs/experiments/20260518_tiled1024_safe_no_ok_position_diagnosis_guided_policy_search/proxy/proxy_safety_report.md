# Proxy Safety Report

- Proxy role: safety filtering and coarse screening only.
- Final policy selection is based on 5 epoch short-training balanced score.
- Candidate count: `30`
- Proxy samples: `96`
- Hard filter pass count: `20`

| rank | policy_id | pass | proxy_score | safety_score | combined | bbox_valid | original_retention | new_bbox_valid | small_retention | cutout_occ | reasons |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | search_policy_019 | true | 0.9584 | 1.0000 | 0.9854 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 2 | search_policy_005 | true | 0.9142 | 1.0000 | 0.9700 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 3 | search_policy_030 | true | 0.8870 | 1.0000 | 0.9605 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 4 | search_policy_011 | true | 0.9001 | 0.9856 | 0.9557 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 5 | search_policy_029 | true | 0.8727 | 1.0000 | 0.9555 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 6 | search_policy_016 | true | 0.8832 | 0.9928 | 0.9544 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 7 | search_policy_009 | true | 0.8685 | 1.0000 | 0.9540 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 8 | search_policy_010 | true | 0.8809 | 0.9928 | 0.9536 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 9 | search_policy_017 | true | 0.8781 | 0.9928 | 0.9526 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 10 | search_policy_006 | true | 0.8741 | 0.9928 | 0.9512 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 11 | search_policy_007 | true | 0.8731 | 0.9928 | 0.9509 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 12 | search_policy_022 | true | 0.8844 | 0.9856 | 0.9502 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 13 | search_policy_021 | true | 0.8706 | 0.9928 | 0.9500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 14 | search_policy_023 | true | 0.8695 | 0.9856 | 0.9450 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 15 | search_policy_012 | true | 0.8659 | 0.9856 | 0.9437 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 16 | search_policy_015 | true | 0.8292 | 1.0000 | 0.9402 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 17 | search_policy_025 | true | 0.8744 | 0.9716 | 0.9376 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 18 | search_policy_004 | true | 0.8420 | 0.9856 | 0.9353 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 19 | search_policy_027 | true | 0.8396 | 0.9856 | 0.9345 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 20 | search_policy_001 | true | 0.8353 | 0.9856 | 0.9330 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  |
| 21 | search_policy_018 | false | 0.9088 | 0.9926 | 0.9633 | 0.9926 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | invalid bbox count=1 |
| 22 | search_policy_008 | false | 0.9054 | 0.9926 | 0.9621 | 0.9926 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | invalid bbox count=1 |
| 23 | search_policy_026 | false | 0.9032 | 0.9926 | 0.9613 | 0.9926 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | invalid bbox count=1 |
| 24 | search_policy_013 | false | 0.8780 | 0.9926 | 0.9525 | 0.9926 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | invalid bbox count=1 |
| 25 | search_policy_020 | false | 0.8637 | 0.9926 | 0.9475 | 0.9926 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | invalid bbox count=1 |
| 26 | search_policy_002 | false | 0.8725 | 0.9784 | 0.9414 | 0.9927 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | invalid bbox count=1 |
| 27 | search_policy_014 | false | 0.8651 | 0.9784 | 0.9388 | 0.9927 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | invalid bbox count=1 |
| 28 | search_policy_028 | false | 0.8476 | 0.9784 | 0.9326 | 0.9927 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | invalid bbox count=1 |
| 29 | search_policy_003 | false | 0.8677 | 0.9647 | 0.9307 | 0.9928 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | invalid bbox count=1 |
| 30 | search_policy_024 | false | 0.8341 | 0.9647 | 0.9190 | 0.9928 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | invalid bbox count=1 |
