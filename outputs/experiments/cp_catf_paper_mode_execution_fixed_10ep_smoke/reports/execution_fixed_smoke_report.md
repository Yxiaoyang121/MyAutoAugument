# CP-CATF Paper-Mode Execution-Fixed 10ep Smoke

## Scope

- Run root: `outputs/experiments/cp_catf_paper_mode_execution_fixed_10ep_smoke/`.
- Purpose: verify the repaired paper-mode accept-to-execution implementation without running another 50ep experiment.
- Seed: `1`.
- Epochs: `10`.
- Mode: `paper_probe_mode=true`, `causal_probe_mode=true`.
- Policy selection source: `probe_split`.
- Final validation used for policy selection: `false`.

## Smoke Result

- The run completed successfully.
- Final-val leakage remained disabled:
  - `final_val_used_for_policy_selection=false`
  - train_core/probe overlap: `0`
  - probe/final-val overlap: `0`
  - train_core/final-val overlap: `0`
- The only causal-probe event occurred at epoch 5.
- Candidate decision: `candidate_policy_3_sampler_only`.
- Image modification allowed: `false`.
- Sample weighting status: `pending_dataloader_support`.
- No candidate image policy was accepted during this 10ep smoke.
- Industrial samples augmented: `0`.
- ROI applied: `0`.
- Router random draw count: `0`.
- BBox/class legality checks passed:
  - invalid bbox count: `0`
  - bbox out-of-bounds count: `0`
  - class id out-of-bounds count: `0`

## Interpretation

This smoke confirms that paper-mode no-leakage and strict image no-op behavior still hold after the accept-to-execution fix. It does not prove applied augmentation in a live 10ep training run, because this particular smoke selected `candidate_policy_3_sampler_only` at epoch 5 and had no `candidate_policy_1_roi_texture` accept within 10 epochs.

The executable accept path is covered by `tests/test_cp_catf_accept_to_execution.py`: accepted causal-probe policies now materialize into active class-op policy entries, enter the sample router, and can trigger ROI augmentation; rejected and sampler-only decisions remain strict image no-op.

## Follow-Up

The previous paper-mode multiseed result should remain classified as leakage-free no-op safety validation. A future paper-mode multiseed rerun is required to measure CP-CATF once accepted image policies actually execute during training.
