# CATF-v2 No-op Seed1 50ep Control

- Clean native: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_1\clean_native_yolo_default`
- CATF-v2: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_1\catf_v2`
- CATF-v2 noop: `outputs\experiments\catf_v2_noop_control_50ep\seed_1`
- Noop reproduces clean native: `true`
- Noop constraint failed: `false`
- Industrial samples augmented: `0`
- ROI applied: `0`
- Policy update applied count: `0`
- Router random draw count: `0`
- CATF-v2 seed1 gain explained by no-op framework: `false`

## Metrics

| metric | clean | noop | CATF-v2 | noop-clean | noop-CATF-v2 |
|---|---:|---:|---:|---:|---:|
| precision | 0.772525 | 0.772525 | 0.769106 | +0.000000 | +0.003419 |
| recall | 0.647680 | 0.647680 | 0.695033 | +0.000000 | -0.047353 |
| map50 | 0.754191 | 0.754191 | 0.754913 | +0.000000 | -0.000723 |
| map50_95 | 0.479919 | 0.479919 | 0.489814 | +0.000000 | -0.009896 |

## Interpretation

- CATF-v2 noop reproduces clean native exactly, so the CATF-v2 framework path by itself did not explain seed1's CATF-v2 result.
- Diagnosis-only had already shown the callback alone is not a source of drift; this no-op control isolates the custom trainer/router path.
