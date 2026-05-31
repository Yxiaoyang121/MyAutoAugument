# CATF-v2 Seed42 vs Clean Native YOLO Default

| Run | Precision | Recall | mAP50 | mAP50-95 | Constraint |
|---|---:|---:|---:|---:|---|
| Clean native seed42 | 0.7262 | 0.6844 | 0.7616 | 0.5250 | reference |
| CATF-v2 seed42 | 0.7498 | 0.7257 | 0.7679 | 0.5212 | failed=`false` |
| Delta | +0.0236 | +0.0413 | +0.0062 | -0.0039 | [] |

CATF-v2 passes the seed42 industrial constraints. It should move to multiseed validation before being treated as the paper main method.
