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
- `fig_*.py` — scripts that generate the article figures.
- `tests/` — test suite: `python -m pytest tests/` (no ANSYS required).
- `pyFBSmaster/` — vendored [pyFBS](https://gitlab.com/ladisk/pyFBS) providing the SEMM implementation.

## Measurement data

The measurement recordings (roving-hammer IR campaign and the base-excitation recording, ~3 GB) are published as a Zenodo dataset: **DOI to be added**. The notebooks read the dataset's folder structure directly — set `DATA_ROOT` in the first configuration cell of each notebook to the extracted dataset location.

Load the raw `.hcc` IR recordings with `fasthcc.read_hcc(path, calibrated=True)`; force and accelerometer records are LDAQ pickles.

## Dependencies

Python 3.10+ with `numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm` (see `requirements.txt`); ANSYS MAPDL only for FE generation in the synthetic validation and the stage-1 prior.

## Development repository

Development history: https://github.com/jasasonc/thermoelastic-stress-expansion
