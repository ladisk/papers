# Dual Stage Base Pipeline

This folder contains an isolated two-stage thermoelastic SEMM workflow.

## Files
- `semm_thermoelastic_pipeline.py`: class-based processing module
- `SEMM_dual_stage_base_excitation.ipynb`: end-to-end notebook
- `configs/`: optional saved configuration artifacts (for example `stage2_crop.json`)
- `outputs/`: run outputs saved as `outputs/<timestamp>/stage1` and `outputs/<timestamp>/stage2`

## Run order
1. Open `SEMM_dual_stage_base_excitation.ipynb`
2. Run config cell and verify paths
3. Run Stage 1 cells (`build_hammer_stage1`, `run_stage1_numerical_and_semm`)
4. Run Stage 2 load/crop cell
- default: `crop_mode='inherit_stage1'`
- manual: set `crop_mode='manual'`, call `pick_crop_two_clicks`, then `save_stage2_crop`
5. Run Stage 2 build/identification/SEMM cells
- default parent mode: `parent_q_mode='virtual_base'` (stage-1 only, no stage-2 overlay fitting for `q`)
- optional old mode: `parent_q_mode='fit_overlay'`
- camera caching: set `stage2_camera_npy` and keep `cache_stage2_camera_npy=True` to auto-save/load `.npy`
6. Run diagnostics cell
7. Run save cell

## Output artifacts
Under `outputs/<timestamp>/`:
- `stage1/Y_SEMM_stage1.npy`
- `stage1/freq_axis.npy`
- `stage1/mapping_metadata.json`
- `stage2/Y_parent_acc.npy`
- `stage2/Y_overlay_base.npy`
- `stage2/Y_SEMM_stage2.npy`
- `stage2/q_equiv.npy`
- `stage2/fit_residual.npy`
- `stage2/stage2_crop_used.json`
- `run_metadata.json`

## Notes
- Existing notebooks are untouched.
- Stage-2 crop policy is explicit and persisted.
- Acceleration reference is converted from `g` to `m/s^2` when `Stage2Config.acc_in_g=True`.
