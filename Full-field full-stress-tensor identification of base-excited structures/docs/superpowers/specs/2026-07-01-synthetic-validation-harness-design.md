# Synthetic Validation Harness — Design Spec

**Date:** 2026-07-01
**Project:** thermoelastic-stress-expansion (Clanek_2 / MSSP paper "Full-Field Full-Stress-Tensor Identification of Base-Excited Structures")
**Purpose:** Answer the MSSP major-revision reviewers' core asks — independent validation of the individual stress components, uncertainty quantification, FE-sensitivity, modal convergence, and conditioning — with a single simulation-based harness where the ground truth is known exactly.

---

## 1. Motivation & problem statement

The real experiment measures only the first stress invariant σx+σy (thermoelastic camera) and has **no independent sensor** for the individual components σxx, σyy, τxy (no strain gauges, no DIC). Both reviewers demand proof that the recovered individual components are accurate, plus uncertainty and FE-sensitivity analysis:

- **R1.2, R1.8, R2.2, R2.7** — quantitative validation of the individual components against a reference; error metrics (relative error, RMS, MAC); comparison against FE-only.
- **R2.1, R2.10** — how errors in the FE-derived component ratios propagate into the recovered components; separate what is *measured* from what is *model-inferred*.
- **R2.5, R1.9, R1.12** — uncertainty/noise quantification, confidence intervals, repeatability.
- **R1.10, R2.3** — justification of the two-mode truncation (convergence).
- **R2.4** — conditioning of the pseudoinverse; whether regularization is needed.

The harness answers all of these from one engine by running the method on synthetic data generated from a known "truth" model.

## 2. Core concept: truth vs parent

Two FE models, deliberately different, play **asymmetric** roles:

- **Truth model** — the stand-in for the "real structure." Generates all synthetic measurements (fake camera + accelerometer) and holds the exact answer key (its σxx, σyy, τxy are known analytically). The pipeline never sees the truth's individual components.
- **Parent model** — a deliberately imperfect/simplified FE, playing the role the simplified FE plays in the real SEMM pipeline. This is the only FE the method is allowed to use.

If truth == parent the test is circular. The gap between them **is** the FE-discrepancy experiment (Theme C).

### Two pipeline runs (differ only in the FE prior)

| Run | FE prior fed to method | Data source | Purpose |
|-----|------------------------|-------------|---------|
| **Realistic** | degraded parent | truth + calibrated noise | the actual test — what a real user gets |
| **Oracle ("cheat")** | truth itself | truth + calibrated noise | best case — isolates error from noise + modal truncation + conditioning |

- **Oracle error** = noise + truncation + conditioning floor (Themes B/D/E).
- **Realistic − Oracle gap** = error attributable specifically to FE-model discrepancy (Theme C, R2.1/R2.10).

The oracle run is built into the harness. Whether it appears in the paper is decided after seeing the numbers.

## 3. Non-goals / out of scope

- No new physical measurements (strain gauges / DIC handled separately; may be added as an experimental anchor later per mentor decision).
- Direct-PSD expansion method is **not** validated in this first harness (transmissibility, the main method, only). Structure the code so it can be added later.
- Clamping-*stiffness* (compliant boundary) discrepancy is not supported by the current fully-clamped solver; it is discussed as a limitation, not swept. Supported discrepancies: material modulus, density, thickness, plate dimensions, added point mass, force/clamp configuration.
- No fatigue-life calculation here (Theme G is a separate work item).

## 4. Architecture

New self-contained package inside the existing repo. **Only one change to existing code** (the solver extension in §5.1); everything else is new files.

```
thermoelastic-stress-expansion/
├── stage1_solver/
│   └── plate_stress_modal_superposition_*.py   # EXISTING — extended to save modal stress shapes (§5.1)
├── dual_stage_base_pipeline/
│   └── semm_thermoelastic_pipeline.py           # EXISTING — reused as the expansion engine
└── synthetic_validation/                        # NEW package
    ├── __init__.py
    ├── fe_models.py       # config → MAPDL run → (stress-tensor FRF, modal data); cache by config hash
    ├── forward_model.py   # truth modal data → Stage-1 (6 hammer FRFs) + Stage-2 (base-exc camera+accel)
    ├── noise.py           # calibrate camera & accel noise floors from real recordings; inject
    ├── expansion.py       # thin wrapper over semm_thermoelastic_pipeline (SEMM modes + transmissibility)
    ├── metrics.py         # NRMSE, per-point error, MAC, condition number
    ├── studies.py         # drivers for studies A/B/C/D/E
    ├── configs/
    │   ├── truth.json
    │   ├── parent.json
    │   └── sweep_*.json
    ├── cache/             # cached FE outputs keyed by config hash (git-ignored)
    └── notebooks/         # thin presentation notebooks → paper figures
tests/                     # known-answer unit tests
```

### End-to-end data flow

```
truth.json ─► fe_models ─► truth FRF + modal data ─┐
                                                    ├─► forward_model
parent.json ─► fe_models ─► parent FRF + modal data ┘        │
                                                             ├─ Stage-1: 6 synthetic hammer FRFs
real recordings ─► noise (calibrate floors) ───────────────┤    (truth→overlay σx+σy +noise, parent→full 4-comp)
                                                             └─ Stage-2: base-exc camera frames +noise, accel +noise
                                                                          │
                                  expansion (SEMM modes + transmissibility) ─► recovered σxx,σyy,τxy PSD
                                                                          │
truth modal data ─► forward_model ─► TRUE σxx,σyy,τxy PSD  ──► metrics ◄──┘  (NRMSE, MAC, per-point, ratios)
```

## 5. Components

### 5.1 `fe_models.py` + solver extension

- **Solver extension (existing file):** the solver already computes `modal_sx/sy/sxy` and `modal_ux/uy/uz` internally but discards them. Extend the saved `modal_data.npz` to also store:
  - `modal_sx, modal_sy, modal_sxy` — shape `(nmodes, nnodes)`, the per-mode stress shapes.
  - `modal_uz` and the lumped nodal mass vector `m` — needed for the base-excitation participation factor, OR precompute and save `gamma_base` directly (see §5.2).
  - This is additive; the existing force-FRF outputs are unchanged.
- **`fe_models.py`:** config-driven wrapper. Reads a JSON config (material E/ν/ρ, thickness, plate size, grid, point mass, clamp edge, damping ζ, mode count, frequency vector, force-node list for the 6 hammer locations). Launches MAPDL (user's choice), runs the solver, returns/caches:
  - `stress_tensor_frf` `(nfreq, nnodes, 4)` complex `[Pa/N]` (force-driven, per force location),
  - `modal_data` (freqs, ω, generalized mass, stress shapes, base participation factor).
- **Caching:** results keyed by a hash of the config dict; if the cache file exists, skip MAPDL. Makes sweeps/Monte-Carlo cheap and makes results reproducible without a license.

### 5.2 `forward_model.py`

Turns a model's modal data into synthetic measurements. Two synthesis paths:

**Base-excitation response (Stage-2 truth + reference).** For enforced base acceleration in Z, the stress-per-base-acceleration FRF for component c is
```
T_c(x, ω) = -Σ_r ψ_{r,c}(x) · Γ_r / (ω_r² − ω² + 2 i ζ_r ω_r ω)      [Pa/(m/s²)]
```
with base participation factor `Γ_r = (Σ_n m_n · φ_{r,n,z}) / M_r` (influence vector = unit rigid Z translation). The stress PSD is `S_σσ,c(x, ω) = |T_c(x, ω)|² · S_aa(ω)`.
- Evaluated from **truth** modal data for all 4 components → the **ground-truth reference** (no pipeline involved).
- Used to synthesize the fake camera signal (σx+σy component) and, with the chosen `S_aa`, the fake accelerometer.

**Synthesis mode (default: time-domain).** Generate a base-acceleration time realization consistent with `S_aa`, propagate through the truth modal model to stress time histories, take the σx+σy field at camera-pixel locations, convert to temperature `ΔT = −K_T (σx+σy)`, reshape to fake camera frames `(nframes, H, W)`, add calibrated per-pixel noise; produce the fake accelerometer time history + noise. The **existing pipeline** consumes these exactly as it consumes real data (Welch variance emerges naturally — important for study B). A faster frequency-domain path may be added for large sweeps.

**Stage-1 synthesis (6 hammer cases).** Uses the force-driven FRFs:
- **Parent overlay-free part:** parent model full 4-component stress FRFs at all nodes for the 6 force locations → SEMM *parent*.
- **Truth overlay (experimental analog):** truth model σx+σy FRF at the camera pixels for the 6 force locations + calibrated FRF-level noise → SEMM *overlay*.
- Noise on the Stage-1 overlay is applied at the FRF level, scaled to a comparable SNR (modeling choice, flagged for review). Stage-2 is where the honest time-domain noise + uncertainty live.

### 5.3 `noise.py`

- **Camera floor:** read the real Stage-2 camera `.npy` once; estimate the spatially-uncorrelated per-pixel sensor noise level (e.g. from high-frequency / signal-free content). Store as a reusable calibrated level.
- **Accelerometer floor:** estimate from the real accel `.pkl`.
- **Injection:** add Gaussian noise at the calibrated level(s). Study B multiplies the level to sweep SNR. Seeded RNG for reproducibility; the seed varies across Monte-Carlo repetitions.

### 5.4 `expansion.py`

Thin wrapper over `semm_thermoelastic_pipeline.py`. Runs Stage-1 SEMM (parent + overlay → hybrid stress mode shapes) and Stage-2 transmissibility expansion (H1 transmissibility → pseudoinverse modal decomposition → multi-component expansion → stress PSD). No re-implementation of SEMM or the expansion math; the harness only *feeds* synthetic inputs and *collects* outputs. Exposes hooks for study D (number of modes) and study E (condition number, optional regularization).

### 5.5 `metrics.py`

Score recovered vs truth, per component (σxx, σyy, τxy) and for σx+σy:
- **NRMSE** — normalized RMS error of the stress-PSD field, per frequency and aggregated (single-number "% off").
- **Per-point error** — at a few critical locations (feeds a reviewer table, R1.4/R1.12).
- **MAC** — spatial-pattern agreement between recovered and truth fields/mode shapes (0–1).
- **Component-ratio error** — error in σxx:σyy:τxy ratios specifically (directly targets R2.1, since the FE sets the ratios).
- **Condition number** of the modal basis matrix inverted in the decomposition (study E).

### 5.6 `studies.py`

Drivers; each is a loop over the harness. Each emits figures/tables for the response letter.

| Study | Loop | Output | Reviewer |
|-------|------|--------|----------|
| **A — Recovery** | single realistic run (+ oracle) at calibrated noise | error metrics + spatial maps of recovered vs truth σxx/σyy/τxy | R1.2, R1.8, R2.2, R2.7 |
| **C — FE-discrepancy** | sweep parent perturbations (E ±10%, thickness, density, added mass, …) | error vs discrepancy curves; realistic−oracle gap | R2.1, R2.10 |
| **B — Noise** | Monte-Carlo repetitions at calibrated level + swept SNR | confidence bands on recovered PSDs; repeatability stats | R2.5, R1.9, R1.12 |
| **D — Modal convergence** | expansion with 1,2,3,… modes | error vs #modes; justification of 2-mode truncation | R1.10, R2.3 |
| **E — Conditioning** | report condition number / singular-value spectrum; optional regularization on/off | conditioning table; effect of regularization | R2.4 |

Studies B, D, E reuse the A/C machinery — cheap once A and C work.

## 6. Testing (validating the validator)

Known-answer unit tests guard against a harness bug making the method look artificially good or bad:

1. **Zero-noise, parent==truth** → recovered components equal truth to near machine precision.
2. **Single-mode forward model** → closed-form check of `T_c(x,ω)` against the analytic single-DOF formula.
3. **Noise injection** → injected noise reproduces the target level/statistics; seeded RNG is deterministic.
4. **Metrics** → NRMSE = 0 for identical inputs; MAC = 1 for identical fields, ~0 for orthogonal ones; condition number matches a hand-computed small matrix.
5. **Caching** → same config hits cache; changed config misses.

## 7. Reviewer-comment coverage map

| Reviewer point | Covered by |
|----------------|-----------|
| R1.2, R1.8, R2.2 (quantitative component validation) | Study A + metrics |
| R2.7 (comparison vs FE-only) | Oracle vs realistic; parent-only baseline |
| R2.1, R2.10 (FE-error propagation; measured vs inferred) | Study C + realistic−oracle gap + component-ratio metric |
| R2.5, R1.9, R1.12 (uncertainty, repeatability) | Study B (calibrated noise Monte-Carlo) |
| R1.10, R2.3 (modal truncation) | Study D |
| R2.4 (conditioning / regularization) | Study E |

(Themes F cross-spectra, G fatigue, and the writing themes are handled outside this harness.)

## 8. Open items to confirm during implementation

- Exact form saved by the solver extension: raw `modal_uz` + lumped `m`, vs precomputed `Γ_r`.
- Stage-1 overlay noise model (FRF-level SNR) — revisit once real Stage-1 SNR is measured.
- Default `S_aa(ω)` for the synthetic base excitation (flat vs the real shaped profile).
- Which parent perturbations make the headline Study-C figure.
- Error thresholds that constitute "success" (a research judgment — set after seeing Study A).
```
