# CP-CATF Offline Causal Probe Summary

- Training run executed: `false`
- Probe mode: `development`
- Uses existing validation diagnostics: `true`
- Paper-mode note: policy selection must move to a train/probe split before final claims.
- RiskGuard fixed blacklist as final rule: `false`
- RiskGuard status: downgraded to audit/debug prior.

## Seed Decisions

| Seed | Selected candidate | Action | Image aug allowed | Image aug rejected | Main reason |
|---:|---|---|---:|---:|---|
| 0 | `candidate_policy_1_roi_texture` | `accept` | `true` | `false` | `accepted` |
| 1 | `candidate_policy_1_roi_texture` | `accept` | `true` | `false` | `accepted` |
| 2 | `candidate_policy_3_sampler_only` | `sampler_only` | `false` | `true` | `image candidates rejected; sampler_only pending` |

## Run-Specific Evidence

- Generic candidate `candidate_policy_1_roi_texture` accepted seeds: `[0, 1]`.
- Generic candidate `candidate_policy_1_roi_texture` rejected seeds: `[2]`.
- This supports CP-CATF as run-specific rather than dataset-specific: the rule never branches on seed id, class id, or category name.

## Seed2 Rejection

- Seed2 image-space candidates are rejected because fixed CATF-v2 shows negative Recall/mAP movement and the audit reports non-active class regression.
- RiskGuard recovered the audited class-local path but still failed overall constraints, so fixed class-op blacklist is insufficient as a final method.
- Selected fallback is sampler-only/no image modification, with sample weighting still marked pending when dataloader support is absent.

## Recommendation

- Enter CP-CATF training validation: `true`.
- Next validation should run seeds 0/1/2 with only probe-passed candidates entering the router and strict no-op when no image candidate passes.
