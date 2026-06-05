# CATF-v2-Gated Retrospective Simulation

Generated: `2026-06-05T19:01:52`
Code commit used: `ff77080a1046f3bdb05da10b5c1904c803380491`

## Gate Replay Results

| seed | fallback expected | fallback epoch | expected final choice | expected dP | expected dR | expected dM50 | expected dM95 | expected constraint_failed | gate reasons |
|---:|---|---:|---|---:|---:|---:|---:|---|---|
| 0 | false | None | fixed_catf_v2 | -0.0060 | -0.0068 | +0.0090 | +0.0136 | false | `{}` |
| 1 | false | None | fixed_catf_v2 | +0.0127 | +0.0528 | +0.0284 | +0.0390 | false | `{}` |
| 2 | true | 10 | clean_fallback | +0.0000 | +0.0000 | +0.0000 | +0.0000 | false | `{'epoch10_bad_pattern_A': 1}` |

## Required Answers

1. Seed0 fallback under new gate: `false`.
2. Seed1 fallback under new gate: `false`.
3. Seed2 fallback under new gate: `true` at epoch `10`.
4. Expected final choice: seed0 fixed CATF-v2, seed1 fixed CATF-v2, seed2 clean fallback.
5. Expected 3/3 constraint pass: `true`.
6. Overfit assessment: No obvious seed-id overfit: the controller uses metric-shape gates and active-class evidence rather than seed IDs or class IDs. The thresholds were derived from the current three-seed failure analysis, so this still needs validation beyond seeds 0/1/2 before being claimed as generally robust.

## Event Trace

### Seed 0

| epoch | phase | action | positive_gain | bad_patterns | reasons | dP | dR | dM50 | dM95 |
|---:|---|---|---|---|---|---:|---:|---:|---:|
| 5 | epoch5_observe | observe | false | `[]` | `[]` | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| 10 | first_formal_gate | gate_continue | true | `[]` | `[]` | -0.0165 | +0.0909 | +0.0465 | +0.0201 |
| 15 | monitor | gate_continue | true | `[]` | `[]` | +0.1834 | -0.1065 | +0.0165 | -0.0010 |
| 20 | monitor | gate_continue | true | `[]` | `[]` | -0.0354 | +0.0412 | +0.0149 | -0.0142 |
| 25 | monitor | gate_continue | false | `[]` | `[]` | -0.0050 | -0.0475 | -0.0222 | -0.0354 |
| 30 | monitor | gate_continue | true | `[]` | `[]` | -0.0087 | +0.0891 | +0.0236 | +0.0220 |
| 35 | monitor | gate_continue | true | `[]` | `[]` | +0.0643 | +0.0763 | +0.0692 | +0.0444 |
| 40 | monitor | gate_continue | true | `[]` | `[]` | +0.0129 | +0.0273 | +0.0329 | +0.0219 |
| 45 | monitor | gate_continue | true | `[]` | `[]` | +0.0566 | -0.0750 | +0.0268 | +0.0151 |

### Seed 1

| epoch | phase | action | positive_gain | bad_patterns | reasons | dP | dR | dM50 | dM95 |
|---:|---|---|---|---|---|---:|---:|---:|---:|
| 5 | epoch5_observe | observe | false | `[]` | `[]` | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| 10 | first_formal_gate | gate_continue | true | `[]` | `[]` | +0.0601 | -0.0080 | +0.0195 | +0.0119 |
| 15 | monitor | gate_continue | false | `[]` | `[]` | -0.0897 | +0.0468 | -0.0168 | -0.0034 |
| 20 | monitor | gate_continue | true | `[]` | `[]` | -0.1298 | +0.0022 | -0.0017 | +0.0119 |
| 25 | monitor | gate_continue | true | `[]` | `[]` | -0.1362 | +0.0457 | +0.0138 | +0.0055 |
| 30 | monitor | gate_continue | false | `[]` | `[]` | -0.0146 | -0.0529 | -0.0543 | -0.0431 |
| 35 | monitor | gate_continue | true | `[]` | `[]` | -0.0758 | +0.0487 | +0.0158 | +0.0187 |
| 40 | monitor | gate_continue | true | `[]` | `[]` | +0.0010 | +0.0153 | +0.0149 | +0.0043 |
| 45 | monitor | gate_continue | true | `[]` | `[]` | +0.0056 | -0.0066 | +0.0584 | +0.0465 |

### Seed 2

| epoch | phase | action | positive_gain | bad_patterns | reasons | dP | dR | dM50 | dM95 |
|---:|---|---|---|---|---|---:|---:|---:|---:|
| 5 | epoch5_observe | observe | false | `[]` | `[]` | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| 10 | first_formal_gate | no_op_freeze | false | `['bad_pattern_A', 'bad_pattern_B']` | `['epoch10_bad_pattern_A']` | +0.0368 | -0.0452 | -0.0208 | -0.0181 |

