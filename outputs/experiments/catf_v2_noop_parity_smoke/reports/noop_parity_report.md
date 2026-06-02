# CATF-v2 No-op 1ep Parity Audit

- Clean run: `outputs\experiments\catf_v2_noop_parity_smoke\clean_native_1ep`
- CATF-v2 noop run: `outputs\experiments\catf_v2_noop_parity_smoke\catf_v2_noop_1ep`
- 1ep parity pass: `true`
- Metrics identical: `true`
- results.csv numeric max abs diff excluding time: `0`
- results.csv wall-clock time diff: `-25.652000`
- args.yaml diff count: `2`
- Industrial samples augmented: `0`
- ROI applied: `0`
- Router random draw count: `0`
- No-op transform calls: `2301`
- Policy update applied count: `0`

## Metric Delta

| metric | clean | noop | noop-clean |
|---|---:|---:|---:|
| precision | 0.827730 | 0.827730 | +0.000000 |
| recall | 0.197746 | 0.197746 | +0.000000 |
| map50 | 0.212652 | 0.212652 | +0.000000 |
| map50_95 | 0.146403 | 0.146403 | +0.000000 |

## Args Diff

| key | clean | noop |
|---|---|---|
| `project` | `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_noop_parity_smoke\clean_native_1ep` | `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_noop_parity_smoke\catf_v2_noop_1ep` |
| `save_dir` | `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_noop_parity_smoke\clean_native_1ep\train` | `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_noop_parity_smoke\catf_v2_noop_1ep\train` |
