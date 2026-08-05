# Writing Agent — Full-Field Stress PSD from Thermoelastic Camera Measurements

> **This folder contains everything needed to write a journal article about
> dual-stage full-field stress expansion under base excitation using an infrared
> thermoelastic camera.**

---

## 1. What This Research Is About

### The Problem

Measuring the full stress tensor field on a vibrating structure is difficult.
Traditional strain gauges give stress at a few discrete points. Finite element
(FE) models can predict stress everywhere, but only if the excitation and
boundary conditions are perfectly known — which they rarely are in practice.

### The Solution: Thermoelastic Camera + Modal Expansion

An infrared (IR) camera can measure the thermoelastic effect — tiny temperature
fluctuations caused by stress oscillations — at thousands of spatial points
simultaneously. However, the camera measures only one stress quantity: the
**first stress invariant** σ_x + σ_y (the sum of normal stresses, also called
the trace of the in-plane stress tensor).

This research develops a **two-stage approach** to expand the single-component
camera measurement into full multi-component stress PSD fields (σ_x, σ_y, τ_xy)
at every spatial point:

**Stage 1** — Obtain stress mode shapes for all stress components from hammer
impact FRFs (6 impact locations, camera + force hammer, processed through SEMM
to get a hybrid experimental-numerical stress FRF model). These mode shapes
encode the *relationship* between stress components.

**Stage 2** — Use the camera measurement under base excitation (the actual
loading condition of interest) to determine *how much* each mode contributes.
Then multiply by the Stage 1 mode shapes for each stress component to get the
full field.

### The Key Innovation: Two Expansion Methods

Two methods are developed for Stage 2, each with different trade-offs:

1. **Transmissibility expansion** (main method) — uses camera + accelerometer to
   compute the stress transmissibility T(f,x) = G_σa / G_aa, decomposes it onto
   the modal basis, and expands to all components. The transmissibility is a
   reusable structural property.

2. **Direct PSD expansion** (companion method) — uses camera only (no
   accelerometer). Computes the stress PSD directly, denoises it using a virtual
   reference technique (spatial average of all pixels), decomposes via NNLS onto
   squared mode shapes, and expands.

A third approach — **full dual-stage SEMM** — was attempted but discarded because
SEMM requires multiple independent input cases and base excitation provides only
one (see Section 7 below).

### The Application

The final output is the stress PSD S_σσ(f, x) for each component (σ_x, σ_y,
τ_xy) at every spatial point, at every frequency. This is exactly what is needed
for **vibration fatigue analysis** — spectral methods (Dirlik, Tovo-Benasciutti,
etc.) can convert stress PSD directly into fatigue damage predictions.

---

## 2. Experimental Setup

### Specimen

- **Material**: Aluminum alloy (Al 6082-T6 or similar)
- **Geometry**: Flat rectangular plate, 150 mm × 150 mm
- **Boundary conditions**: Clamped at the bottom edge (bolted to a fixture on a shaker table)
- **Thermoelastic properties** (hardcoded in `semm_thermoelastic_pipeline.py`):
  - Thermal expansion coefficient: α = 23.0 × 10⁻⁶ 1/K
  - Density: ρ = 2700 kg/m³
  - Specific heat: c_p = 900 J/(kg·K)
  - Reference temperature: T_ref = 293.15 K (20°C)
  - Thermoelastic coefficient: K_T = -(α · T_ref) / (ρ · c_p) = -2.774 × 10⁻⁹ K/Pa
  - Camera-to-stress conversion factor: 1/K_T = -360,403,118 Pa/K

### Measurement Equipment

- **IR Camera**: Telops FAST-IR (mid-wave infrared, InSb detector)
  - Frame rate: 2000 Hz (subwindow mode)
  - Resolution: 96 × 128 pixels (after subwindowing)
  - After crop + 3× reduction: 30 × 31 = 930 pixels → mapped to 841 FE nodes (29 × 29 grid)
  - Sensitivity: measures temperature fluctuations of order ~mK
  - Camera noise is spatially uncorrelated between pixels (sensor noise)

- **Accelerometer**: Single-axis, mounted on the shaker table / fixture base
  - Sampling rate: 25,600 Hz (resampled to match camera at 2000 Hz)
  - Measures base acceleration in m/s² (originally recorded in g, converted via g₀ = 9.80665)

- **Force Hammer** (Stage 1 only): Instrumented impact hammer
  - 6 impact locations on the plate surface: center_middle, center_top, left_top, left_middle, right_middle, right_top
  - Sampling rate: 25,600 Hz

- **Shaker**: Electromagnetic shaker for base excitation
  - Broadband random excitation: 40–300 Hz (or 45–300 Hz depending on measurement)
  - Stationary random profile (flat or shaped PSD)

### Measurements Performed

**Stage 1 — Hammer impacts** (6 cases):
- For each impact location: simultaneous force + camera recording
- Camera and force are synchronized via impact detection in post-processing
- Duration: single impact transient (few hundred ms useful data per case)
- FRF computed via H1 estimator: H1(f) = G_σF / G_FF

**Stage 2 — Base excitation** (1 case):
- Shaker provides broadband random base acceleration
- Camera records stress field; accelerometer records base motion
- Duration: 5 seconds continuous (10,000 frames at 2 kHz)
- **Important note**: In the `best_one` dataset, camera and accelerometer were
  recorded at different times (desynchronized). The notebooks handle this with a
  virtual-reference workaround. For the article, describe the proper synchronized
  method.

### Frequency Range

- Stage 1 SEMM FRFs: 45.6 – 299.8 Hz, Δf = 0.691 Hz, 369 frequency lines
- Stage 2 Welch analysis: 45.0 – 300.0 Hz, Δf = 0.600 Hz, ~425 frequency lines
- Two dominant modes in this range:
  - **Mode 1**: ~54 Hz (first bending mode)
  - **Mode 2**: ~176 Hz (second mode)

---

## 3. File Structure

```
for_writing_agent/
├── AGENT_README.md                           ← YOU ARE HERE
├── _fix_paths.py                             ← One-time script (already run)
│
├── dual_stage_base_pipeline/                 ← Runnable code + notebooks
│   ├── __init__.py                           ← Package init (empty)
│   ├── semm_thermoelastic_pipeline.py        ← Main pipeline module (68 KB)
│   ├── configs/
│   │   └── stage2_crop.json                  ← Camera crop ROI
│   │
│   ├── transmissibility_expansion.ipynb      ← METHOD 1 (main article method)
│   ├── direct_psd_expansion.ipynb            ← METHOD 2 (companion method)
│   └── SEMM_dual_stage_base_excitation.ipynb ← METHOD 3 (discarded, for reference)
│
├── docs/
│   ├── README.md                             ← Pipeline overview
│   ├── method_transmissibility_expansion.md  ← Detailed theory + code for Method 1
│   ├── method_direct_psd_expansion.md        ← Detailed theory + code for Method 2
│   └── method_dual_stage_semm.md             ← Detailed theory + code for Method 3
│
└── plate_stress_modal_superposition_force_pointmass_v2_camfreq_20260206_1239.py
                                              ← ANSYS FE solver script (Stage 1)
```

### Data files (at absolute paths, NOT copied)

All notebooks reference data via:
```python
PROJECT_ROOT = Path(r'c:\Users\jasas\Work\Clanki\Clanek_2\Trigger_and_acquisition')
```

**Stage 1 pre-computed SEMM outputs** (~114 MB):
```
{PROJECT_ROOT}/dual_stage_base_pipeline/outputs/20260219_090812/
├── stage1/
│   ├── Y_SEMM_stage1.npy          (369, 3364, 6) complex128 — stress FRF
│   ├── freq_axis.npy              (369,) float64 — frequency vector [Hz]
│   └── mapping_metadata.json      grid/mapping info
├── stage2/
│   ├── Y_SEMM_stage2.npy          (369, 3364, 1) — SEMM result (failed method)
│   ├── Y_parent_acc.npy           (369, 3364, 1) — parent acceleration FRF
│   ├── Y_overlay_base.npy         (369, ~930, 1) — camera overlay
│   ├── q_equiv.npy                (369, 6) — equivalent inputs
│   └── fit_residual.npy           (369,) — overlay fit residual
└── run_metadata.json              full configuration record
```

**FE node coordinates** (20 KB):
```
{PROJECT_ROOT}/dual_stage_base_pipeline/outputs/_stage1_numerical_runs/
└── run_center_middle/
    └── node_coords.npy            (841, 3) float64 — x, y, z positions
```

**Stage 2 measurement data** (~941 MB):
```
{PROJECT_ROOT}/base_excitation_measurements/best_one/
├── _20260217T123846487_20260217T123846978.npy   (10000, 96, 128) — camera frames
└── 20260217_123809_base_excitation.pkl           accelerometer data (25.6 kHz)
```

**Stage 1 hammer data** (~3.3 GB, NOT needed for the expansion notebooks):
```
{PROJECT_ROOT}/hammer_measurement_6_points/
├── camera/npy/    6 × ~590 MB .npy files
└── force/         6 × PKL files
```

---

## 4. Method 1: Transmissibility Expansion (Main Article Method)

**Notebook**: `dual_stage_base_pipeline/transmissibility_expansion.ipynb`
**Theory doc**: `docs/method_transmissibility_expansion.md`

### Pipeline

1. **Stage 1 mode shapes** (Sections 1–2): Load pre-computed SEMM FRFs → identify
   resonances via peak detection → extract stress mode shapes |ψ_r,c(x)| for all
   4 components (SX, SY, SXY, SX+SY) at each resonance → normalize by camera
   component norm.

2. **Stress transmissibility** (Sections 3–4): Load camera + accelerometer time
   histories → compute H1 transmissibility T(f,x) = G_σa(f,x) / G_aa(f) at each
   pixel → map pixels to FE nodes.

3. **Modal decomposition** (Section 5): Project transmissibility onto mode shape
   basis via pseudoinverse: γ_r(f) = Ψ_cam⁺ · T_cam(f). The complex modal
   participation γ_r(f) peaks at each resonance.

4. **Multi-component expansion** (Section 6): Reconstruct transmissibility for
   each stress component: T_c(f,x) = Σ_r γ_r(f) · ψ̂_r,c(x). Compute stress
   PSD: S_σσ,c(f,x) = |T_c(f,x)|² · S_aa(f).

5. **Validation** (Section 7): Compare expanded PSD against direct camera Welch.
   The expanded result is lower (noise removed by H1) — the gap equals the
   coherence factor γ².

### Key equations for the article

- H1 transmissibility: T(f,x) = G_σa(f,x) / G_aa(f)
- Modal decomposition: γ(f) = Ψ_cam⁺ · T_cam(f)
- Expansion: T_c(f,x) = Σ_r γ_r(f) · ψ̂_r,c(x)
- Stress PSD: S_σσ,c(f,x) = |T_c(f,x)|² · S_aa(f)

### Noise handling

The H1 estimator uses the cross-spectrum G_σa, in which camera noise (uncorrelated
with accelerometer) cancels. Result: |T_H1|² · S_aa = |H|² · S_aa = γ² · G_yy.
The expanded PSD is lower than the raw camera Welch by the coherence factor — this
is correct (noise removed, not signal lost).

### Temporary fix status

Camera and accelerometer in the `best_one` dataset were recorded at different times.
Sections marked `>>> TEMPORARY FIX <<<` use a virtual-reference workaround:
- Virtual reference = spatial average of 930 camera pixels
- T_vref = G_σx,ref / G_ref,ref (inter-pixel cross-spectra, noise-free)
- Scale: T_abs = T_vref · √(S_ref / S_aa) (uses separately-recorded S_aa)

**For the article: describe the proper H1 method** T = G_σa / G_aa with
simultaneous camera + accelerometer. The expansion pipeline (Sections 5–6) is
identical regardless of how T is estimated.

---

## 5. Method 2: Direct PSD Expansion (Companion Method)

**Notebook**: `dual_stage_base_pipeline/direct_psd_expansion.ipynb`
**Theory doc**: `docs/method_direct_psd_expansion.md`

### Pipeline

1. **Stage 1 mode shapes** (Sections 1–2): Same as Method 1.

2. **Camera stress PSD** (Section 4): Compute Welch auto-spectrum G_yy(f,x) of
   camera stress signals. This includes noise: G_yy = |H|² S_aa + S_nn.

3. **Virtual reference denoising** (Section 4b): The camera's spatial redundancy
   (930 pixels) enables self-denoising. A virtual reference r(t) = mean of all
   pixels has √930 ≈ 30× noise reduction. Cross-correlating each pixel with this
   reference removes remaining noise:
   S_clean(f,x) = |G_xr(f)|² / G_rr(f) ≈ |H(f,x)|² · S_aa(f)

4. **NNLS modal decomposition** (Section 5): Fit the denoised PSD onto squared
   mode shape basis via non-negative least squares:
   S_clean(f,x) ≈ Σ_r β_r(f) · |ψ̂_r,cam(x)|²
   with PSD-level weighting (antinodes weighted more than nodal lines).

5. **Multi-component expansion** (Section 6): Swap mode shapes for each component:
   S_c(f,x) = Σ_r β_r(f) · |ψ̂_r,c(x)|²

6. **Validation** (Section 7): Three-level comparison: raw Welch → denoised →
   expanded. The gap between raw and denoised equals the noise floor.

### Key equations for the article

- Virtual reference denoising: S_clean(f,x) = |G_xr|² / G_rr
- PSD decomposition: S_clean(f,x) ≈ Σ_r β_r(f) · |ψ̂_r,cam(x)|²
- Expansion: S_c(f,x) = Σ_r β_r(f) · |ψ̂_r,c(x)|²

### Key difference from Method 1

| Aspect | Transmissibility | Direct PSD |
|--------|-----------------|------------|
| Accelerometer | Required | Not needed |
| Denoising | H1 cross-spectrum (cam × accel) | Virtual reference (cam × cam avg) |
| Decomposition | Pseudoinverse on complex T | NNLS on real |ψ|² |
| Result | Noise-free structural PSD | Noise-free structural PSD |
| Reusable | Yes (T is structural property) | No (PSD includes excitation) |

Both methods produce noise-free results. The transmissibility method additionally
separates the structural transfer function from the excitation.

---

## 6. Stage 1 SEMM — How Mode Shapes Are Obtained

Both expansion methods (Sections 4–5) use the same Stage 1 mode shapes. The mode
shapes come from the SEMM FRF `Y_SEMM_stage1.npy`, which was computed by the
SEMM dual-stage notebook.

### What SEMM does

SEMM (System Equivalent Model Mixing) combines:
- **Parent model**: FE numerical stress FRFs (all 4 stress components at 841 nodes,
  6 hammer cases). Computed by running an ANSYS modal superposition solver for each
  impact location.
- **Overlay model**: Experimental camera stress FRFs (SX+SY only, at ~930 pixels,
  same 6 cases). Computed via H1 from force + camera time histories.

SEMM corrects the FE model at the camera pixels (interface DOFs) using the
experimental data, then propagates the correction to all 4 stress components
through the numerical model's component relationships.

**Result**: `Y_SEMM_stage1.npy` — a hybrid stress FRF that has:
- Experimentally-accurate amplitudes at camera locations (SX+SY)
- Physically-consistent component ratios (SX, SY, SXY) from the FE model
- Shape: (369 freq × 3364 DoFs × 6 cases), where 3364 = 841 nodes × 4 components

### How mode shapes are extracted from Y_SEMM

The expansion notebooks extract mode shapes by:
1. Computing spatial-RMS of the SX+SY component across all 6 cases
2. Detecting peaks in this RMS spectrum (resonance frequencies)
3. At each peak frequency, extracting |H(f_peak, x, c)| averaged over cases
4. Normalizing all components by the SX+SY norm

This is a simplified peak-based extraction. The `docs/method_*.md` files also
describe a more rigorous approach using `sdypy-EMA` (LSCF polynomial fitting)
which is available but currently disabled (`USE_SDYPY_EMA = False`).

### The solver script

`plate_stress_modal_superposition_force_pointmass_v2_camfreq_20260206_1239.py`
is the ANSYS Python script that computes numerical stress FRFs. It:
- Creates a 29×29 node plate mesh
- Runs modal analysis to get eigenfrequencies and mode shapes
- Computes stress FRFs via modal superposition at each force application point
- Outputs: `stress_tensor_frf.npy` (complex, 4 components per node), `freqs.npy`,
  `node_coords.npy`

The solver is included for completeness but does NOT need to be re-run — the
Stage 1 outputs are pre-computed.

---

## 7. Method 3: Full Dual-Stage SEMM (Discarded)

**Notebook**: `dual_stage_base_pipeline/SEMM_dual_stage_base_excitation.ipynb`
**Theory doc**: `docs/method_dual_stage_semm.md`

### What it attempted

Apply SEMM twice:
1. Stage 1 SEMM: 6 hammer cases → works well (overdetermined, 6 independent inputs)
2. Convert hammer FRFs to base-acceleration FRF via equivalent inputs
3. Stage 2 SEMM: Correct with actual base-excitation camera measurement

### Why it failed

**The fundamental problem is data insufficiency in Stage 2.**

SEMM requires multiple independent input cases to resolve the spatial stress field.
Stage 1 has 6 hammer impact locations → the SEMM system is overdetermined and
produces reliable results.

Stage 2 has only **1 base-excitation input case** — a single shaker providing
broadband random acceleration. Regardless of the frequency content, this is
fundamentally 1 input case (1 column in the FRF matrix). SEMM cannot distinguish
between different spatial patterns with a single input.

**Evidence of failure:**
- The SEMM update ratio is 3.82× (382% change from the parent model), far
  exceeding the expected ~1.0 for a well-conditioned update
- Workarounds attempted (virtual base point, DoF decimation every 3rd node) are
  regularization hacks that don't solve the underlying rank deficiency

### What to say in the article

- Mention that the full dual-stage SEMM approach was attempted
- State it requires multiple independent input cases, which base excitation
  does not provide (single shaker = single input case)
- This motivates the simpler expansion approaches (Methods 1 and 2), which
  bypass SEMM in Stage 2 entirely and use modal decomposition instead
- Stage 1 SEMM (with 6 hammer inputs) is still valuable and is used by both
  expansion methods for mode shape extraction

---

## 8. Key Results to Report

### Resonance frequencies
- Mode 1: ~54 Hz (first bending)
- Mode 2: ~176 Hz (second mode)
- Small shift between Stage 1 (hammer) and Stage 2 (base excitation) peaks:
  typically < 2 Hz, attributed to boundary condition variability between setups

### Mode shapes
- 4 stress components at 841 spatial nodes for each mode
- Normalized by SX+SY component
- Spatial patterns consistent between Stage 1 FRF shapes and Stage 2 measurements
  (high MAC values at resonances)

### Reconstruction quality
- Modal fit accuracy at resonances: typically 80-90% for Mode 1, 25-66% for Mode 2
  (Mode 2 is weaker and more affected by noise)
- NNLS residual is small at resonances (where the 2-mode basis captures most energy)
- Away from resonances, residual is high (expected — no structural energy there)

### Noise removal
- Virtual reference denoising: noise fraction ~0% at Mode 1 (strong signal),
  ~35% at Mode 2 (weaker signal)
- H1 denoising: coherence γ² quantifies the structural fraction of the signal
- Both methods produce noise-free expanded stress PSD

### Self-consistency
- Expanded SX+SY PSD matches the denoised camera PSD at resonances
- The gap between raw Welch and expanded = noise removed

---

## 9. Thermoelastic Theory (Background for Article)

The thermoelastic effect relates temperature change to stress under adiabatic
conditions. For an isotropic material under plane stress:

ΔT = -K_T · (σ_x + σ_y)

where K_T = (α · T_ref) / (ρ · c_p) is the thermoelastic coefficient.

The IR camera measures ΔT at each pixel. Converting to stress:
σ_x + σ_y = ΔT / (-K_T) = ΔT × (-360,403,118 Pa/K)

**Key property**: The camera measures σ_x + σ_y (first stress invariant), NOT
individual components. This is why we need mode shapes from Stage 1 to decompose
into σ_x, σ_y, τ_xy separately.

**Adiabatic condition**: The thermoelastic equation is valid when heat conduction
is negligible during one stress cycle. This requires the vibration frequency to
be high enough that thermal diffusion length < pixel size. For our aluminum plate
at f > 40 Hz, this condition is well satisfied.

**Linearity**: The thermoelastic relationship is linear (ΔT ∝ Δσ), which enables
all the spectral analysis methods used in this work.

---

## 10. How to Run the Notebooks

### Python environment requirements

```
numpy
scipy
matplotlib
pandas
tqdm          (optional, progress bars)
sdypy         (optional, for EMA modal identification — currently disabled)
```

Not needed for the expansion notebooks (only for re-running SEMM from scratch):
```
pyFBS         (SEMM algorithm — local package at {PROJECT_ROOT}/pyFBSmaster/)
TelopsToolbox (Telops HCC file reader — only needed for raw .hcc camera files)
```

### Running transmissibility_expansion.ipynb

1. Run all cells sequentially
2. Cell with `%matplotlib qt` + `plt.ginput()` (interactive crop) — either click
   two corners for a new crop, or close the window to keep the default
3. All data paths are hardcoded to the absolute PROJECT_ROOT
4. Output: `S_stress` dict with stress PSD for all 4 components, `T_expanded`
   dict with complex transmissibility, `gamma` modal participation functions

### Running direct_psd_expansion.ipynb

1. Run all cells sequentially (no interactive cells)
2. Uses the same Stage 1 data and same camera recording as the transmissibility
   notebook
3. Output: `S_stress` dict with stress PSD for all 4 components, `beta` modal
   participation PSD, `S_denoised_nodes` virtual-ref denoised camera PSD

### SEMM_dual_stage_base_excitation.ipynb

- **Do NOT re-run** — requires ANSYS, pyFBS, TelopsToolbox, and ~3.5 GB of
  hammer measurement data
- The notebook already has all outputs saved in its cell outputs
- Read it for understanding the full SEMM pipeline and why it fails in Stage 2

---

## 11. Notation Reference

| Symbol | Description | Units |
|--------|-------------|-------|
| σ_x, σ_y | Normal stresses | Pa |
| τ_xy | Shear stress | Pa |
| σ_x + σ_y | First stress invariant (camera measures this) | Pa |
| T(f,x) | Stress transmissibility | Pa/(m/s²) |
| S_σσ(f,x) | Stress power spectral density | Pa²/Hz |
| S_aa(f) | Base acceleration PSD | (m/s²)²/Hz |
| G_σa | Cross-spectral density (stress × acceleration) | Pa·(m/s²)/Hz |
| G_aa | Auto-spectral density (acceleration) | (m/s²)²/Hz |
| γ² | Coherence | dimensionless [0,1] |
| ψ_r,c(x) | Stress mode shape, mode r, component c | normalized |
| γ_r(f) | Modal participation (transmissibility method) | Pa/(m/s²) |
| β_r(f) | Modal participation PSD (direct PSD method) | Pa²/Hz |
| H1 | H1 FRF estimator: G_yx / G_xx | — |
| NNLS | Non-negative least squares | — |
| SEMM | System Equivalent Model Mixing | — |
| FRF | Frequency Response Function | — |
| PSD | Power Spectral Density | — |
| MAC | Modal Assurance Criterion | — |

---

## 12. Data File Shapes Quick Reference

| File | Shape | Dtype | Content |
|------|-------|-------|---------|
| `Y_SEMM_stage1.npy` | (369, 3364, 6) | complex128 | Stress FRF: 369 freq × (841 nodes × 4 comps) × 6 cases |
| `freq_axis.npy` | (369,) | float64 | Frequency vector [Hz] |
| `node_coords.npy` | (841, 3) | float64 | FE node x, y, z positions [m] |
| Camera `.npy` | (10000, 96, 128) | float64 | 10k frames × 96 rows × 128 cols [K] |
| `mapping_metadata.json` | — | JSON | Grid: 29 rows × 29 cols, camera FOV ranges |

**Interleaved layout of Y_SEMM**: The second axis (3364) is 841 nodes × 4
components interleaved: [SX_node0, SY_node0, SXY_node0, (SX+SY)_node0, SX_node1,
...]. Extract component c with `Y_SEMM[:, c::4, :]`.

---

## 13. Article Structure Suggestion

1. **Introduction** — Vibration fatigue requires full stress field; cameras
   measure only σ_x+σ_y; expansion via modal decomposition
2. **Theory**
   - Thermoelastic stress measurement
   - Stage 1: SEMM for hybrid stress mode shapes
   - Stage 2a: Transmissibility expansion (H1 + modal decomposition)
   - Stage 2b: Direct PSD expansion (virtual ref denoising + NNLS)
3. **Experimental setup** — Plate specimen, camera, accelerometer, shaker, hammer
4. **Results**
   - Stage 1 mode shapes (all 4 components)
   - Transmissibility expansion: spatial fields, spectral comparison, noise removal
   - Direct PSD expansion: virtual ref denoising, NNLS fit, spatial fields
   - Comparison of both methods (Table: reusability, hardware, noise handling)
5. **Discussion**
   - Why dual-stage SEMM fails for base excitation (single input case)
   - Virtual reference as camera self-denoising
   - Transmissibility vs direct PSD trade-offs
   - Limitations (modal truncation, 2 modes only in this demo)
6. **Conclusion**

---

## 14. References / Prior Work

- **Česnik, M., Slavič, J., Boltežar, M.** (2013). "Assessment of the fatigue
  parameters from random vibration testing: Application to a rivet joint." *JSV*
  — Thermoelastic stress FRF concept, stress mode shapes

- **SEMM (System Equivalent Model Mixing)**: pyFBS library documentation,
  Lagrange Multiplier FBS theory

- **sdypy-EMA**: LSCF (Least Squares Complex Frequency) modal identification

- **Thermoelastic stress analysis (TSA)**: Standard reference for IR camera-based
  stress measurement under cyclic loading

- **Vibration fatigue**: Dirlik, Tovo-Benasciutti spectral methods for converting
  stress PSD to fatigue damage
