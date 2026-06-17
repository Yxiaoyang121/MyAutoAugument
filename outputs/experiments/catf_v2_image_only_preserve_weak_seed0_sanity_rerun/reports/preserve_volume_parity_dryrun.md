# Preserve Original Volume Parity Dry-Run

This dry-run does not train. It loads the epoch-exact fixed CATF-v2 policy schedule, applies the preserve_original execution path, and checks policy_matrix/sample_router class visibility.

## Totals

- fixed total industrial expected: 41
- preserve total industrial expected after fix: 41
- fixed total ROI expected: 45
- preserve total ROI expected after fix: 45
- old preserve industrial actual: 155
- old preserve ROI actual: 187
- old industrial volume ratio: 3.780488
- old ROI volume ratio: 4.155556
- industrial volume ratio after fix: 1.0
- ROI volume ratio after fix: 1.0

## Execution Check

- epoch-exact preserve: True
- stale policy accumulation after fix: False
- expected class union: [4, 11, 12]
- runtime policy_matrix union: [4, 11, 12]
- sample_router eligible union: [4, 11, 12]
- final executable union: [4, 11, 12]
- sampler_only involved: False
- weighted_index_list involved: False

The corrected dry-run no longer shows the 3x/4x volume amplification seen in the previous seed0 rerun.
