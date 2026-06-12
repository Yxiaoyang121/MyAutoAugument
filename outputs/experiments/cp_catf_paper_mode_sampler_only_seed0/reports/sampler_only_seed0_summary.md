# CP-CATF Paper-Mode Sampler-Only Seed0 Summary

## Scope

- Run: `outputs/experiments/cp_catf_paper_mode_sampler_only_seed0/`
- Seed: `0`
- Epochs: `50`
- Paper-mode: `true`
- Final val used for policy selection: `false`
- Image augmentation path: rejected by precision-aware gate and forced no-op.
- Sampler-only path: enabled with weighted index list.

## Dataloader Intervention

- Dataloader/dataset changed: `true`
- Strategy: `weighted_index_list`
- WeightedRandomSampler used: `false`
- Reason: Ultralytics `build_dataloader()` does not expose an external sampler injection point in this trainer path.
- Final sample weight map generated: `true`
- Final weighted train_core images count: `72`
- Weighted index list enabled: `true`
- Sampler-only effective: `true`
- Sampled distribution changed: `true`

## Feedback Epochs

| epoch | action | weighted train_core images | distribution changed |
|---:|---|---:|---|
| 5 | sampler_only_pending | 0 | false |
| 10 | sampler_only_pending | 0 | false |
| 15 | sampler_only_effective | 34 | true |
| 20 | sampler_only_effective | 55 | true |
| 25 | sampler_only_effective | 50 | true |
| 30 | sampler_only_effective | 50 | true |
| 35 | sampler_only_effective | 123 | true |
| 40 | sampler_only_effective | 13 | true |
| 45 | sampler_only_effective | 72 | true |

## Augmentation Safety

- Industrial image augmented: `0`
- ROI applied: `0`
- Router random draw count: `0`
- Invalid bbox count: `0`
- Class id out-of-bounds count: `0`

## Metrics vs Requested Clean Paper Seed0

| run | P | R | mAP50 | mAP50-95 | constraint_failed |
|---|---:|---:|---:|---:|---|
| clean paper seed0 | 0.7513 | 0.6763 | 0.7566 | 0.5114 | false |
| sampler-only seed0 | 0.7588 | 0.6878 | 0.7779 | 0.5204 | false |
| delta | +0.0075 | +0.0115 | +0.0213 | +0.0090 |  |

## Artifact Paths

- Final metrics: `outputs/experiments/cp_catf_paper_mode_sampler_only_seed0/reports/final_metrics.json`
- Online augmentation stats: `outputs/experiments/cp_catf_paper_mode_sampler_only_seed0/reports/online_aug_stats.json`
- Causal probe events: `outputs/experiments/cp_catf_paper_mode_sampler_only_seed0/reports/causal_probe_events.json`
- Final sampler map: `outputs/experiments/cp_catf_paper_mode_sampler_only_seed0/reports/catf_v2/sample_weight_map_epoch_45.json`
- Final weighted indices: `outputs/experiments/cp_catf_paper_mode_sampler_only_seed0/reports/catf_v2/weighted_train_indices_epoch_45.json`
- Final distribution audit: `outputs/experiments/cp_catf_paper_mode_sampler_only_seed0/reports/catf_v2/sampled_distribution_before_after_epoch_45.json`
