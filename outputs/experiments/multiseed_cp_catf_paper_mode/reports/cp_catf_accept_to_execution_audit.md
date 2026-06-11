# CP-CATF Paper-Mode Accept-to-Execution Audit

## Conclusion

- Accept events found: `8`
- Accept events with executable policy matrix: `0`
- Industrial samples augmented across paper-mode runs: `0`
- ROI applied across paper-mode runs: `0`
- Router random draw count across paper-mode runs: `0`

The historical paper-mode runs recorded `candidate_policy_1_roi_texture` accept events, but the accepted candidate was not injected into the training policy matrix. The after-causal-probe policy files contain no active class-op entries, so the sample router had no executable target and bypassed before drawing any augmentation random numbers.

## Per-Seed Events

### Seed 0

| epoch | candidate | accepted | op list | policy matrix active ops | class history active classes | final status | no-op reason |
|---:|---|---|---|---|---|---|---|
| 25 | `candidate_policy_1_roi_texture` | `True` | `['sharpen_mild', 'local_contrast']` | `{}` | `[]` | `accepted_but_no_executable_policy` | candidate accept was recorded, but no target class/op was injected into after_causal_probe policy |

### Seed 1

| epoch | candidate | accepted | op list | policy matrix active ops | class history active classes | final status | no-op reason |
|---:|---|---|---|---|---|---|---|
| 25 | `candidate_policy_1_roi_texture` | `True` | `['sharpen_mild', 'local_contrast']` | `{}` | `[]` | `accepted_but_no_executable_policy` | candidate accept was recorded, but no target class/op was injected into after_causal_probe policy |
| 40 | `candidate_policy_1_roi_texture` | `True` | `['sharpen_mild', 'local_contrast']` | `{}` | `[]` | `accepted_but_no_executable_policy` | candidate accept was recorded, but no target class/op was injected into after_causal_probe policy |

### Seed 2

| epoch | candidate | accepted | op list | policy matrix active ops | class history active classes | final status | no-op reason |
|---:|---|---|---|---|---|---|---|
| 20 | `candidate_policy_1_roi_texture` | `True` | `['sharpen_mild', 'local_contrast']` | `{}` | `[]` | `accepted_but_no_executable_policy` | candidate accept was recorded, but no target class/op was injected into after_causal_probe policy |
| 25 | `candidate_policy_1_roi_texture` | `True` | `['sharpen_mild', 'local_contrast']` | `{}` | `[]` | `accepted_but_no_executable_policy` | candidate accept was recorded, but no target class/op was injected into after_causal_probe policy |
| 30 | `candidate_policy_1_roi_texture` | `True` | `['sharpen_mild', 'local_contrast']` | `{}` | `[]` | `accepted_but_no_executable_policy` | candidate accept was recorded, but no target class/op was injected into after_causal_probe policy |
| 35 | `candidate_policy_1_roi_texture` | `True` | `['sharpen_mild', 'local_contrast']` | `{}` | `[]` | `accepted_but_no_executable_policy` | candidate accept was recorded, but no target class/op was injected into after_causal_probe policy |
| 45 | `candidate_policy_1_roi_texture` | `True` | `['sharpen_mild', 'local_contrast']` | `{}` | `[]` | `accepted_but_no_executable_policy` | candidate accept was recorded, but no target class/op was injected into after_causal_probe policy |

## Root Cause

- `apply_offline_probe_decision_to_policy()` accepted the causal-probe candidate but only applied an op whitelist.
- The whitelist removed non-candidate ops, but it did not set a target class to active and did not assign nonzero prob/strength for the candidate ops.
- `policy_history.json` therefore shows `active_classes=[]` for accepted epochs.
- `SampleAwareAugmentationRouter.apply()` saw no `_has_active_ops(...)` target and bypassed before random draws.
- RiskGuard was not the blocker, and paper-mode did not globally disable industrial augmentation; `industrial_online_augmentation=true` but no executable policy reached the router.
