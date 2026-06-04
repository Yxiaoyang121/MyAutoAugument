# Fixed vs Old CATF-v2 Analysis

## Key Change

- Old CATF-v2 had a formal no-augmentation path that could rewrite labels/Instances even when no op was applied.
- Fixed CATF-v2 bypasses label conversion, clipping, and Instances rebuild unless an actual augmentation is applied.

## Metrics And Constraints

- Old CATF-v2 constraint_failed: `2/3`.
- Fixed CATF-v2 constraint_failed: `1/3`.
- Fixed active classes: `{'11': 3, '4': 2, '12': 1, '9': 1, '8': 1}`.
- Fixed ROI affected classes: `{'4': 14, '11': 132, '12': 14, '8': 6, '9': 25}`.
- OK3 fixed ever active: `False`.
- OK3 fixed ROI total: `0`.

## Interpretation

- The repair improved seed0 from failed to pass and preserved the strong fixed seed1 result.
- Seed2 remains the limiting case, with mAP50 and mAP50-95 below constraint tolerance.
- Fixed CATF-v2 is more credible than old CATF-v2, but remaining stability risk is concentrated in seed2 rather than OK3/no-op safety.
