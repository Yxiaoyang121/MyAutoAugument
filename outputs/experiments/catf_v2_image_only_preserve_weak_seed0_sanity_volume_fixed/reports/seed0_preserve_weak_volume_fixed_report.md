# Seed0 Preserve-Weak Volume-Fixed Sanity Report

## Scope

- Completed 50ep: `True`
- Seed: `0`
- Image-only mainline: `true`
- sampler_only_enabled: `false`
- weighted_index_list_enabled: `False`
- sampled_distribution_changed: `False`
- attenuation_ratio: `0.25`
- Offline decisions: `epoch_exact_fixed_*` schedule

## Replay / Execution

| item | value |
|---|---|
| preserve_original / weak_roi_texture / strict_noop | `9 / 0 / 0` |
| uses epoch_exact_fixed_* fields | `True` |
| epoch5 executable classes | `[4, 11]` |
| epoch15 executable classes | `[12]` |
| other feedback epochs executable classes | `[]` |
| stale ops cleared | `True` |
| seed-level union avoided | `True` |
| weak class9 replacement | `False` |
| image augmentation executed | `True` |

## Volume

| run | industrial augmented | ROI applied | ROI by class |
|---|---:|---:|---|
| fixed CATF-v2 seed0 expected | 41 | 45 | `{'4': 9, '11': 22, '12': 14}` |
| pre-fix preserve rerun | 155 | 187 | over-applied class4/class12 |
| volume-fixed preserve rerun | 41 | 45 | `{'4': 9, '11': 22, '12': 14}` |

Volume close to fixed: `True`. Industrial delta vs fixed: `0`. ROI delta vs fixed: `0`.

## Metrics

| run | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean seed0 | 0.784600 | 0.676500 | 0.734700 | 0.475900 |
| fixed CATF-v2 seed0 | 0.778500 | 0.669700 | 0.743700 | 0.489500 |
| pre-fix preserve rerun | 0.700514 | 0.623545 | 0.712481 | 0.456610 |
| volume-fixed preserve rerun | 0.778506 | 0.669654 | 0.743657 | 0.489541 |

Delta vs clean seed0: `dP=-0.006094`, `dR=-0.006846`, `dM50=+0.008957`, `dM95=+0.013641`.

Delta vs fixed CATF-v2 seed0: `dP=+0.000006`, `dR=-0.000046`, `dM50=-0.000043`, `dM95=+0.000041`.

Delta vs pre-fix preserve rerun: `dP=+0.077992`, `dR=+0.046109`, `dM50=+0.031176`, `dM95=+0.032931`.

Constraint failed vs requested clean seed0 rules: `False`.

## Epoch Decisions

| epoch | executable classes | stale cleared | empty epoch cleared |
|---:|---|---|---|
| 5 | `[4, 11]` | `False` | `False` |
| 10 | `[]` | `True` | `True` |
| 15 | `[12]` | `False` | `False` |
| 20 | `[]` | `True` | `True` |
| 25 | `[]` | `False` | `True` |
| 30 | `[]` | `False` | `True` |
| 35 | `[]` | `False` | `True` |
| 40 | `[]` | `False` | `True` |
| 45 | `[]` | `False` | `True` |

## Interpretation

The training-side preserve volume/lifetime parity fix succeeded. The run reproduces fixed CATF-v2 seed0 volume exactly (`41/45`) and reproduces fixed seed0 metrics within rounding. The previous precision/mAP collapse was caused by preserve lifetime over-activation, not by the three-stage decision rule itself.

Recommended next step: `continue with seed2 validation`.
