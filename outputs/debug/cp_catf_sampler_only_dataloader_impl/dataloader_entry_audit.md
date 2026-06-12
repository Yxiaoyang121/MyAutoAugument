# CP-CATF Sampler-Only Dataloader Entry Audit

## Entry Points

- Training script: `scripts/train_yolo_default_with_inloop_feedback.py`
- Custom trainer factory: `scripts/train_yolo_online_aug.py::make_online_trainer`
- Train dataset creation: `OnlineAugDetectionTrainer.build_dataset(..., mode="train")`
- Train dataset class: inner `OnlineYOLODataset`
- Train dataloader creation: Ultralytics `DetectionTrainer.get_dataloader(...)`, now observed by `OnlineAugDetectionTrainer.get_dataloader(...)`

## Findings

1. `scripts/train_yolo_default_with_inloop_feedback.py` does not create the YOLO train dataset directly. It passes `trainer=make_online_trainer(api, context)` into `YOLO.train(...)`.
2. `OnlineAugDetectionTrainer.build_dataset()` creates the train dataset with `OnlineYOLODataset(...)`.
3. Validation datasets still use `super().build_dataset(...)`; sampler-only never touches final val.
4. Ultralytics `build_dataloader(...)` does not expose an external PyTorch `sampler=` argument through this training path. It constructs its own sampler internally only for DDP.
5. Replacing the sampler directly would require overriding more of the Ultralytics dataloader construction surface.
6. The minimal viable hook is dataset index mapping:
   - `OnlineYOLODataset.__len__()` returns the weighted index-list length when enabled.
   - `OnlineYOLODataset.get_image_and_label(index)` maps the weighted index back to the original train_core image index.
   - Original image paths and label files are not copied or rewritten.
7. `OnlineAugDetectionTrainer.get_dataloader()` stores the current train loader in `OnlineTrainingContext`. When sampler-only is activated, the feedback callback calls `loader.reset()` so the current Ultralytics `InfiniteDataLoader` iterator observes the changed dataset length.

## Implementation Choice

- Chosen implementation: weighted index list.
- WeightedRandomSampler status: not used, because the active Ultralytics `build_dataloader()` call path does not accept a sampler injection point.
- Dataloader intervention point: `OnlineYOLODataset.set_weighted_indices(...)`.
- Effective condition: sample weight map generated, weighted train_core image count > 0, weighted index list installed, loader reset, and sampled class distribution changed.

## Audit Outputs

- `outputs/debug/cp_catf_sampler_only_dataloader_impl/sample_weight_map.json`
- `outputs/debug/cp_catf_sampler_only_dataloader_impl/weighted_train_indices.json`
- `outputs/debug/cp_catf_sampler_only_dataloader_impl/sampled_distribution_before_after.json`
