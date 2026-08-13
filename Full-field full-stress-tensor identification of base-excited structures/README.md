# Full-field full-stress-tensor identification of base-excited structures

Code and resources for the paper *Full-Field Full-Stress-Tensor Identification of Base-Excited Structures* by J. Šonc, K. Zaletelj and J. Slavič (Laboratory for Dynamics of Machines and Structures, University of Ljubljana), Mechanical Systems and Signal Processing (2026).

This directory contains the complete processing pipeline and the numerical-validation framework described in the article.

## Contents

- `dual_stage_base_pipeline/` — the two-step identification pipeline:
  - `stage1_hammer.ipynb` — Step 1: hybrid stress-mode shapes via SEMM (roving hammer + IR camera),
  - `transmissibility_expansion.ipynb` — Step 2: transmissibility-based expansion under random base excitation (primary method),
  - `direct_psd_expansion.ipynb` — accelerometer-free direct-PSD variant (Appendix),
  - `semm_thermoelastic_pipeline.py` — the pipeline implementation,
  - `method_*.md` — detailed method documentation.
- `synthetic_validation/` — the numerical-validation framework (Section 4 and Appendix B of the article): truth/parent FE models, forward model, noise calibration and injection, expansion, metrics, and the study/analysis scripts (`analysis/`). FE generation requires ANSYS MAPDL; the analytic path and the test suite run without ANSYS.
- `stage1_solver/` — the simplified flat-plate modal-stress solver used as the FE prior.
- `fig_*.py` — scripts that generate the article figures; `plot_style.py` holds the shared matplotlib style.
- `tests/` — test suite: `python -m pytest tests/` (no ANSYS required).
- `pyFBSmaster/` — vendored [pyFBS](https://gitlab.com/ladisk/pyFBS) providing the SEMM implementation.

## Measurement data

The measurement recordings (roving-hammer IR campaign and the base-excitation recording, ~3 GB) are too large to distribute here. A public deposit is in preparation; until it is available the recordings can be obtained from the authors.

The notebooks read the recordings' folder structure directly. `DATA_ROOT` is the only path that has to be adapted — set it in the configuration cell near the top of each notebook, or supply it through the `THERMO_DATA_ROOT` environment variable. Everything else is resolved relative to the repository.

Load the raw `.hcc` IR recordings with `fasthcc.read_hcc(path, calibrated=True)`; force and accelerometer records are LDAQ pickles.

## Reproducing the figures

The figure scripts run from any working directory and write to `synthetic_validation/figures/`:

```
python fig_tensor_maps.py
```

Thirteen of them work directly against the data included in `synthetic_validation/figures_data/`. The rest first need intermediates that are regenerated rather than shipped, because of their size: `figures_data/validation_fields.npz` (from `synthetic_validation/analysis/gen_validation_fields.py`) and `synthetic_validation/analysis/broad_analysis/` (from `gen_broad_analysis.py` and `gen_robustness_sweep_amp.py`). Regenerating those requires ANSYS MAPDL.

The published figures are typeset with LaTeX. Without a LaTeX installation the scripts fall back to matplotlib's mathtext, which is close but not identical; set `PLOT_STYLE_USETEX` to `0` or `1` to force either mode.

## Dependencies

Python 3.10+ with `numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm` (see `requirements.txt`); ANSYS MAPDL only for FE generation in the synthetic validation and the stage-1 prior.

## Development repository

Development history: https://github.com/jasasonc/thermoelastic-stress-expansion
