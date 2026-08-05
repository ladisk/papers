# Synthetic Validation Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a simulation-based validation harness that generates synthetic thermoelastic-camera data from a known "truth" FE model, runs the real expansion pipeline against a deliberately-imperfect "parent" FE model, and quantifies how accurately the individual stress components (σxx, σyy, τxy) are recovered — answering the MSSP revision reviewers on validation, uncertainty, FE-sensitivity, modal convergence, and conditioning.

**Architecture:** A new `synthetic_validation/` package inside the existing repo. Pure-numpy numeric units (forward model, modal decomposition/expansion, metrics, noise) are TDD-tested without ANSYS by operating on plain `modal_data` dicts. FE generation wraps the existing MAPDL solver with config-driven caching. The Stage-1 SEMM step reuses the existing pyFBS-based engine. The design keeps a "truth" data source separate from the "parent" FE prior so the FE-discrepancy gap is measurable (not circular).

**Tech Stack:** Python 3, numpy, scipy, pandas, pytest; existing `dual_stage_base_pipeline/semm_thermoelastic_pipeline.py` (SEMM engine) and `stage1_solver/plate_stress_modal_superposition_*.py` (MAPDL solver); pyFBS (local, for SEMM); ANSYS MAPDL via `ansys-mapdl-core` (FE generation only).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-01-synthetic-validation-harness-design.md` — authoritative; this plan implements it.
- Validate the **transmissibility** expansion method only (not direct-PSD).
- Two model roles: **truth** = data source + answer key; **parent** = imperfect FE prior fed to the method. Never feed the truth's individual components to the pipeline.
- Support two runs: **realistic** (parent prior) and **oracle** (truth prior).
- Reuse existing SEMM + expansion math; do not re-implement SEMM.
- Noise levels calibrated from the real recordings (not nominal).
- All FE outputs cached by config hash; harness numerics must be testable without ANSYS.
- Stress component index convention (matches `Y_SEMM`): `0=SX, 1=SY, 2=SXY, 3=SX+SY`; interleaved as `[SX_0,SY_0,SXY_0,(SX+SY)_0, SX_1,...]`, slice component `c` with `[:, c::4, ...]`.
- No Claude co-author trailer in commits (repo/user policy).
- Package path: `thermoelastic-stress-expansion/synthetic_validation/`; tests in `thermoelastic-stress-expansion/tests/synthetic_validation/`.

---

### Task 1: Package scaffold, config loader, and test modal fixture

**Files:**
- Create: `synthetic_validation/__init__.py`
- Create: `synthetic_validation/config.py`
- Create: `synthetic_validation/configs/truth.json`
- Create: `synthetic_validation/configs/parent.json`
- Create: `tests/synthetic_validation/__init__.py`
- Create: `tests/synthetic_validation/conftest.py`
- Create: `tests/synthetic_validation/test_config.py`
- Create: `.gitignore` entry for `synthetic_validation/cache/`

**Interfaces:**
- Produces:
  - `FEConfig` dataclass with fields: `E: float, nu: float, rho: float, thickness: float, plate_lx: float, plate_ly: float, grid_nx: int, grid_ny: int, point_mass: float, point_mass_xy: tuple[float,float], base_edge: str, zeta: float, nmodes: int, fmin: float, fmax: float, nfreq: int, force_points: dict[str, tuple[float,float]]`.
  - `load_config(path: str|Path) -> FEConfig`
  - `config_hash(cfg: FEConfig) -> str` (stable sha1 of sorted field dict)
  - Test helper `make_modal_data(freqs_hz, zetas, node_coords, psi_sx, psi_sy, psi_sxy, gamma_base, gen_mass) -> dict` in `conftest.py`, returning a dict with keys `{"modal_freqs","modal_omega","zeta","node_coords","modal_sx","modal_sy","modal_sxy","gamma_base","modal_mass"}`. This is the **contract** every `forward_model` function consumes.

- [ ] **Step 1: Write the failing test**

```python
# tests/synthetic_validation/test_config.py
from synthetic_validation.config import load_config, config_hash

def test_load_and_hash_truth_parent_differ(tmp_path):
    truth = load_config("synthetic_validation/configs/truth.json")
    parent = load_config("synthetic_validation/configs/parent.json")
    assert truth.grid_nx == 29 and truth.base_edge == "bottom"
    # parent is deliberately imperfect -> different modulus -> different hash
    assert parent.E != truth.E
    assert config_hash(truth) != config_hash(parent)
    # hashing is stable
    assert config_hash(truth) == config_hash(load_config("synthetic_validation/configs/truth.json"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic_validation/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'synthetic_validation.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# synthetic_validation/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json, hashlib

@dataclass(frozen=True)
class FEConfig:
    E: float; nu: float; rho: float; thickness: float
    plate_lx: float; plate_ly: float; grid_nx: int; grid_ny: int
    point_mass: float; point_mass_xy: tuple; base_edge: str
    zeta: float; nmodes: int; fmin: float; fmax: float; nfreq: int
    force_points: dict = field(default_factory=dict)

def load_config(path) -> FEConfig:
    d = json.loads(Path(path).read_text())
    d["point_mass_xy"] = tuple(d["point_mass_xy"])
    d["force_points"] = {k: tuple(v) for k, v in d.get("force_points", {}).items()}
    return FEConfig(**d)

def config_hash(cfg: FEConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, default=list)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]
```

Create `truth.json` (nominal: `E=63e9, nu=0.33, rho=2700, thickness=0.003, plate_lx=0.15, plate_ly=0.15, grid_nx=29, grid_ny=29, point_mass=0.1, point_mass_xy=[0.075,0.14], base_edge="bottom", zeta=0.005, nmodes=10, fmin=45.0, fmax=200.0, nfreq=200, force_points` = the 6 hammer points center_middle/center_top/left_top/left_middle/right_middle/right_top with plausible xy). `parent.json` identical **except** `E=58e9` (a deliberate ~8% modulus error) and `point_mass=0.0` (parent omits the added mass — the "missing steel block/feature" simplification).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic_validation/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Add the `make_modal_data` fixture helper to conftest**

```python
# tests/synthetic_validation/conftest.py
import numpy as np, pytest

def make_modal_data(freqs_hz, zetas, node_coords, psi_sx, psi_sy, psi_sxy, gamma_base, gen_mass):
    freqs_hz = np.asarray(freqs_hz, float)
    return {
        "modal_freqs": freqs_hz,
        "modal_omega": 2*np.pi*freqs_hz,
        "zeta": np.asarray(zetas, float),
        "node_coords": np.asarray(node_coords, float),
        "modal_sx": np.asarray(psi_sx, float),      # (nmodes, nnodes)
        "modal_sy": np.asarray(psi_sy, float),
        "modal_sxy": np.asarray(psi_sxy, float),
        "gamma_base": np.asarray(gamma_base, float), # (nmodes,)
        "modal_mass": np.asarray(gen_mass, float),   # (nmodes,)
    }

@pytest.fixture
def single_mode_modal():
    # one mode, 4 nodes, unit stress shapes, known participation
    nodes = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], float)
    return make_modal_data(
        freqs_hz=[54.0], zetas=[0.005], node_coords=nodes,
        psi_sx=[[1.0,0.5,0.5,0.2]], psi_sy=[[0.3,0.3,0.3,0.3]],
        psi_sxy=[[0.1,-0.1,0.1,-0.1]], gamma_base=[2.0], gen_mass=[1.0])
```

- [ ] **Step 6: Commit**

```bash
git add synthetic_validation/ tests/synthetic_validation/ .gitignore
git commit -m "feat(synval): package scaffold, config loader, test modal fixture"
```

---

### Task 2: `metrics.py` — NRMSE, MAC, per-point error, condition number

**Files:**
- Create: `synthetic_validation/metrics.py`
- Test: `tests/synthetic_validation/test_metrics.py`

**Interfaces:**
- Produces:
  - `nrmse(recovered, truth, axis=None) -> float|np.ndarray` — `sqrt(mean((r-t)**2)) / (max(t)-min(t))` over `axis`.
  - `mac(a, b) -> float` — modal assurance criterion for two complex/real vectors, in `[0,1]`.
  - `point_error(recovered, truth, node_indices) -> np.ndarray` — relative error at given nodes.
  - `condition_number(Psi) -> float` — `np.linalg.cond(Psi)`.
  - `component_ratio_error(recovered_by_comp, truth_by_comp) -> float` — error of σyy/σxx and τxy/σxx ratio fields.

- [ ] **Step 1: Write the failing test**

```python
# tests/synthetic_validation/test_metrics.py
import numpy as np
from synthetic_validation.metrics import nrmse, mac, condition_number

def test_nrmse_zero_for_identical():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert nrmse(x, x) == 0.0

def test_mac_bounds():
    a = np.array([1.0, 2.0, 3.0]); b = 2.5 * a
    assert abs(mac(a, b) - 1.0) < 1e-12          # colinear -> 1
    c = np.array([1.0, 0.0, 0.0]); d = np.array([0.0, 1.0, 0.0])
    assert abs(mac(c, d)) < 1e-12                # orthogonal -> 0

def test_condition_number_identity():
    assert abs(condition_number(np.eye(3)) - 1.0) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic_validation/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# synthetic_validation/metrics.py
import numpy as np

def nrmse(recovered, truth, axis=None):
    recovered = np.asarray(recovered); truth = np.asarray(truth)
    rms = np.sqrt(np.mean(np.abs(recovered - truth) ** 2, axis=axis))
    rng = np.ptp(np.abs(truth), axis=axis)
    return np.where(rng > 0, rms / rng, 0.0) if axis is not None else (rms / rng if rng > 0 else 0.0)

def mac(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    num = np.abs(np.vdot(a, b)) ** 2
    den = np.vdot(a, a).real * np.vdot(b, b).real
    return float(num / den) if den > 0 else 0.0

def point_error(recovered, truth, node_indices):
    r = np.asarray(recovered); t = np.asarray(truth)
    idx = np.asarray(node_indices, int)
    denom = np.where(np.abs(t[..., idx]) > 0, np.abs(t[..., idx]), np.nan)
    return np.abs(r[..., idx] - t[..., idx]) / denom

def condition_number(Psi):
    return float(np.linalg.cond(np.asarray(Psi)))

def component_ratio_error(recovered_by_comp, truth_by_comp):
    def ratios(d):
        base = np.abs(d["SX"]) + 1e-30
        return np.stack([np.abs(d["SY"]) / base, np.abs(d["SXY"]) / base])
    rr, rt = ratios(recovered_by_comp), ratios(truth_by_comp)
    return float(np.sqrt(np.mean((rr - rt) ** 2)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic_validation/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add synthetic_validation/metrics.py tests/synthetic_validation/test_metrics.py
git commit -m "feat(synval): metrics (nrmse, mac, point error, condition number, ratio error)"
```

---

### Task 3: `forward_model.py` — base-excitation FRF and truth component PSD

**Files:**
- Create: `synthetic_validation/forward_model.py`
- Test: `tests/synthetic_validation/test_forward_frf.py`

**Interfaces:**
- Consumes: `modal_data` dict (Task 1 contract), `freqs_hz`, `S_aa`.
- Produces:
  - `base_excitation_frf(modal_data, freqs_hz) -> dict[str, np.ndarray]` — keys `"SX","SY","SXY","SX+SY"`, each `(nfreq, nnodes)` complex, the stress-per-base-acceleration FRF `T_c(x,ω) = -Σ_r ψ_{r,c}(x)·Γ_r / (ω_r² - ω² + 2iζ_r ω_r ω)`; `SX+SY = SX + SY`.
  - `truth_component_psd(modal_data, freqs_hz, S_aa) -> dict[str, np.ndarray]` — `S_c = |T_c|² · S_aa`, real `(nfreq, nnodes)`.

- [ ] **Step 1: Write the failing test** (single-mode closed-form check)

```python
# tests/synthetic_validation/test_forward_frf.py
import numpy as np
from synthetic_validation.forward_model import base_excitation_frf, truth_component_psd

def test_single_mode_frf_matches_closed_form(single_mode_modal):
    md = single_mode_modal
    freqs = np.linspace(40, 70, 200)
    T = base_excitation_frf(md, freqs)
    w = 2*np.pi*freqs; wr = md["modal_omega"][0]; z = md["zeta"][0]
    D = wr**2 - w**2 + 2j*z*wr*w
    expect_sx = -md["gamma_base"][0] * md["modal_sx"][0][None, :] / D[:, None]
    assert np.allclose(T["SX"], expect_sx, rtol=1e-10)
    # invariant is the sum
    assert np.allclose(T["SX+SY"], T["SX"] + T["SY"], rtol=1e-12)

def test_psd_is_abs2_times_saa(single_mode_modal):
    md = single_mode_modal
    freqs = np.linspace(40, 70, 50); S_aa = np.ones_like(freqs) * 3.0
    T = base_excitation_frf(md, freqs); S = truth_component_psd(md, freqs, S_aa)
    assert np.allclose(S["SXY"], np.abs(T["SXY"])**2 * S_aa[:, None])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic_validation/test_forward_frf.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# synthetic_validation/forward_model.py
import numpy as np

def base_excitation_frf(modal_data, freqs_hz):
    w = 2*np.pi*np.asarray(freqs_hz, float)
    wr = modal_data["modal_omega"]; z = modal_data["zeta"]; g = modal_data["gamma_base"]
    D = wr[None, :]**2 - w[:, None]**2 + 2j*z[None, :]*wr[None, :]*w[:, None]  # (nfreq, nmodes)
    q = -g[None, :] / D                                                       # (nfreq, nmodes)
    out = {}
    for key, arr in (("SX", modal_data["modal_sx"]), ("SY", modal_data["modal_sy"]),
                     ("SXY", modal_data["modal_sxy"])):
        out[key] = q @ arr                                                    # (nfreq, nnodes)
    out["SX+SY"] = out["SX"] + out["SY"]
    return out

def truth_component_psd(modal_data, freqs_hz, S_aa):
    T = base_excitation_frf(modal_data, freqs_hz); S_aa = np.asarray(S_aa, float)
    return {k: (np.abs(v)**2) * S_aa[:, None] for k, v in T.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic_validation/test_forward_frf.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add synthetic_validation/forward_model.py tests/synthetic_validation/test_forward_frf.py
git commit -m "feat(synval): base-excitation FRF + truth component PSD from modal data"
```

---

### Task 4: `forward_model.py` — time-domain synthesis of camera frames + accelerometer

**Files:**
- Modify: `synthetic_validation/forward_model.py`
- Test: `tests/synthetic_validation/test_forward_synthesis.py`

**Interfaces:**
- Consumes: `modal_data`, `S_aa` spec, sampling params, camera grid mapping.
- Produces:
  - `synthesize_base_excitation(modal_data, fs, n_frames, saa_level, rng, camera_to_stress_factor, grid_shape) -> dict` with keys `"cam_frames"` `(n_frames, n_rows, n_cols)` float (temperature, **noise-free**), `"accel"` `(n_frames,)` float (noise-free), `"S_aa_target"` callable/array. Camera value = `(SX+SY at node) / camera_to_stress_factor`. Uses band-limited white base acceleration realization → modal ODE (frequency-domain multiply by `q(ω)`) → stress time histories at nodes → reshape to grid.
  - Helper `_realize_base_accel(fs, n_frames, saa_level, rng) -> np.ndarray`.

- [ ] **Step 1: Write the failing test** (synthesized accel PSD ≈ target level; camera field is invariant/factor)

```python
# tests/synthetic_validation/test_forward_synthesis.py
import numpy as np
from scipy.signal import welch
from synthetic_validation.forward_model import synthesize_base_excitation, base_excitation_frf

def test_synth_accel_psd_matches_level(single_mode_modal):
    rng = np.random.default_rng(0)
    out = synthesize_base_excitation(single_mode_modal, fs=2000, n_frames=20000,
                                     saa_level=2.0, rng=rng,
                                     camera_to_stress_factor=-3.6e8, grid_shape=(2,2))
    f, Saa = welch(out["accel"], fs=2000, nperseg=2048)
    band = (f > 45) & (f < 65)
    assert 1.0 < np.median(Saa[band]) < 4.0            # ~ target level 2.0

def test_cam_frames_shape(single_mode_modal):
    rng = np.random.default_rng(1)
    out = synthesize_base_excitation(single_mode_modal, fs=2000, n_frames=4096,
                                     saa_level=1.0, rng=rng,
                                     camera_to_stress_factor=-3.6e8, grid_shape=(2,2))
    assert out["cam_frames"].shape == (4096, 2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic_validation/test_forward_synthesis.py -v`
Expected: FAIL with `ImportError: cannot import name 'synthesize_base_excitation'`

- [ ] **Step 3: Write minimal implementation** (append to `forward_model.py`)

```python
def _realize_base_accel(fs, n_frames, saa_level, rng):
    # white acceleration with two-sided PSD ~ saa_level over the band
    # variance = saa_level * (fs/2); generate white noise scaled to that variance
    std = np.sqrt(saa_level * (fs / 2.0))
    return rng.standard_normal(n_frames) * std

def synthesize_base_excitation(modal_data, fs, n_frames, saa_level, rng,
                               camera_to_stress_factor, grid_shape):
    a = _realize_base_accel(fs, n_frames, saa_level, rng)
    freqs = np.fft.rfftfreq(n_frames, d=1.0/fs)
    A = np.fft.rfft(a)
    w = 2*np.pi*freqs
    wr = modal_data["modal_omega"]; z = modal_data["zeta"]; g = modal_data["gamma_base"]
    D = wr[None, :]**2 - w[:, None]**2 + 2j*z[None, :]*wr[None, :]*w[:, None]
    q = (-g[None, :] / D) * A[:, None]                       # modal coord spectra (nfreq, nmodes)
    inv = q @ modal_data["modal_sx"] + q @ modal_data["modal_sy"]   # SX+SY spectrum (nfreq, nnodes)
    sigma = np.fft.irfft(inv, n=n_frames, axis=0)           # (n_frames, nnodes) stress [Pa]
    cam = sigma / camera_to_stress_factor                   # temperature [K]
    nrows, ncols = grid_shape
    return {"cam_frames": cam.reshape(n_frames, nrows, ncols),
            "accel": a, "S_aa_target": saa_level}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic_validation/test_forward_synthesis.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add synthetic_validation/forward_model.py tests/synthetic_validation/test_forward_synthesis.py
git commit -m "feat(synval): time-domain synthesis of camera frames + accelerometer"
```

---

### Task 5: `noise.py` — calibrate floors from real recordings and inject

**Files:**
- Create: `synthetic_validation/noise.py`
- Test: `tests/synthetic_validation/test_noise.py`

**Interfaces:**
- Produces:
  - `estimate_camera_noise_floor(frames, method="hf_temporal") -> float` — per-pixel temperature noise std, estimated from high-frequency temporal content (median over pixels of the std of the temporal high-pass residual).
  - `estimate_accel_noise_floor(accel, fs, quiet_band=(700, 900)) -> float` — accel noise std from an out-of-excitation band.
  - `inject_camera_noise(frames, sigma_T, rng) -> np.ndarray` — additive Gaussian, spatially uncorrelated.
  - `inject_accel_noise(accel, sigma_a, rng) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
# tests/synthetic_validation/test_noise.py
import numpy as np
from synthetic_validation.noise import inject_camera_noise, estimate_camera_noise_floor

def test_injection_reproduces_level():
    rng = np.random.default_rng(0)
    frames = np.zeros((5000, 4, 4))
    noisy = inject_camera_noise(frames, sigma_T=0.002, rng=rng)
    assert abs(np.std(noisy) - 0.002) < 1e-4

def test_estimate_recovers_known_floor():
    rng = np.random.default_rng(2)
    frames = rng.standard_normal((5000, 6, 6)) * 0.003     # pure white noise floor
    est = estimate_camera_noise_floor(frames)
    assert 0.002 < est < 0.004
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic_validation/test_noise.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# synthetic_validation/noise.py
import numpy as np

def inject_camera_noise(frames, sigma_T, rng):
    return np.asarray(frames) + rng.standard_normal(np.shape(frames)) * sigma_T

def inject_accel_noise(accel, sigma_a, rng):
    return np.asarray(accel) + rng.standard_normal(np.shape(accel)) * sigma_a

def estimate_camera_noise_floor(frames, method="hf_temporal"):
    frames = np.asarray(frames)
    # temporal first difference isolates high-frequency (noise-dominated) content;
    # diff of white noise has variance 2*sigma^2 -> divide by sqrt(2)
    d = np.diff(frames, axis=0)
    per_pixel = np.std(d.reshape(d.shape[0], -1), axis=0) / np.sqrt(2.0)
    return float(np.median(per_pixel))

def estimate_accel_noise_floor(accel, fs, quiet_band=(700, 900)):
    from scipy.signal import welch
    f, S = welch(np.asarray(accel), fs=fs, nperseg=min(4096, len(accel)))
    band = (f >= quiet_band[0]) & (f <= quiet_band[1])
    df = f[1] - f[0]
    return float(np.sqrt(np.sum(S[band]) * df))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic_validation/test_noise.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add synthetic_validation/noise.py tests/synthetic_validation/test_noise.py
git commit -m "feat(synval): noise floor calibration + injection"
```

---

### Task 6: `expansion.py` — modal decomposition, multi-component expansion, stress PSD

**Files:**
- Create: `synthetic_validation/expansion.py`
- Test: `tests/synthetic_validation/test_expansion_core.py`

**Interfaces:**
- Produces (pure numpy, mirrors `method_transmissibility_expansion.md` steps 3–5):
  - `normalize_mode_shapes(psi_by_comp) -> dict` — scale all components by `||ψ_cam||` per mode (`psi_by_comp[c]` is `(nnodes, nmodes)`; `c` in `SX,SY,SXY,SX+SY`).
  - `modal_decompose(Psi_cam, T_cam_nodes) -> np.ndarray` — `gamma = (pinv(Psi_cam) @ T_cam_nodes.T).T`, shape `(nfreq, nmodes)`.
  - `expand_components(gamma, psi_by_comp) -> dict` — `T_c = gamma @ Psi_c.T`, keys per component, `(nfreq, nnodes)` complex.
  - `stress_psd(T_by_comp, S_aa) -> dict` — `|T_c|² · S_aa`, real.

- [ ] **Step 1: Write the failing test** (round-trip: known gamma × mode shapes recovers itself)

```python
# tests/synthetic_validation/test_expansion_core.py
import numpy as np
from synthetic_validation.expansion import modal_decompose, expand_components, stress_psd

def test_decompose_inverts_expansion():
    rng = np.random.default_rng(0)
    nnodes, nmodes, nfreq = 20, 2, 30
    Psi_cam = rng.standard_normal((nnodes, nmodes)) + 1j*rng.standard_normal((nnodes, nmodes))
    gamma_true = rng.standard_normal((nfreq, nmodes)) + 1j*rng.standard_normal((nfreq, nmodes))
    T_cam_nodes = (Psi_cam @ gamma_true.T).T                 # (nfreq, nnodes)
    gamma_est = modal_decompose(Psi_cam, T_cam_nodes)
    assert np.allclose(gamma_est, gamma_true, atol=1e-8)

def test_expand_and_psd():
    gamma = np.ones((3, 1), complex)
    psi = {"SX": np.array([[2.0]]), "SY": np.array([[1.0]]),
           "SXY": np.array([[0.5]]), "SX+SY": np.array([[3.0]])}
    T = expand_components(gamma, psi)
    assert np.allclose(T["SX"], 2.0)
    S = stress_psd(T, S_aa=np.array([4.0, 4.0, 4.0]))
    assert np.allclose(S["SX"], 4.0 * 4.0)                   # |2|^2 * 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic_validation/test_expansion_core.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# synthetic_validation/expansion.py
import numpy as np

COMPONENTS = ("SX", "SY", "SXY", "SX+SY")

def normalize_mode_shapes(psi_by_comp):
    cam = psi_by_comp["SX+SY"]                       # (nnodes, nmodes)
    scale = np.linalg.norm(cam, axis=0)              # per-mode ||psi_cam||
    scale = np.where(scale > 0, scale, 1.0)
    return {c: psi_by_comp[c] / scale[None, :] for c in psi_by_comp}

def modal_decompose(Psi_cam, T_cam_nodes):
    pinv = np.linalg.pinv(Psi_cam)                   # (nmodes, nnodes)
    return (pinv @ T_cam_nodes.T).T                  # (nfreq, nmodes)

def expand_components(gamma, psi_by_comp):
    return {c: gamma @ psi_by_comp[c].T for c in psi_by_comp}   # (nfreq, nnodes)

def stress_psd(T_by_comp, S_aa):
    S_aa = np.asarray(S_aa, float)
    return {c: (np.abs(T)**2) * S_aa[:, None] for c, T in T_by_comp.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic_validation/test_expansion_core.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add synthetic_validation/expansion.py tests/synthetic_validation/test_expansion_core.py
git commit -m "feat(synval): modal decomposition + multi-component expansion + stress PSD"
```

---

### Task 7: `expansion.py` — peak-based stress mode-shape extraction from a Y_SEMM array

**Files:**
- Modify: `synthetic_validation/expansion.py`
- Test: `tests/synthetic_validation/test_mode_extraction.py`

**Interfaces:**
- Consumes: `Y_SEMM` `(nfreq, nnodes*4, ncases)` complex, `freq_axis` `(nfreq,)`.
- Produces:
  - `extract_mode_shapes(Y_SEMM, freq_axis, n_modes=2, prominence=None) -> dict` returning `{"freqs": (nmodes,), "psi": {comp: (nnodes, nmodes)}}`. Detect peaks of the spatial-RMS of the `SX+SY` component averaged over cases; at each peak take `|H(f_peak)|` averaged over cases for each component (slice `Y_SEMM[:, c::4, :]`). Matches the notebook's simplified extraction.

- [ ] **Step 1: Write the failing test** (synthetic Y_SEMM with a single resonance)

```python
# tests/synthetic_validation/test_mode_extraction.py
import numpy as np
from synthetic_validation.expansion import extract_mode_shapes

def test_extract_single_peak():
    nfreq, nnodes, ncases = 200, 4, 1
    freq = np.linspace(45, 200, nfreq)
    fr = 54.0; w = 2*np.pi*freq; wr = 2*np.pi*fr
    D = wr**2 - w**2 + 2j*0.005*wr*w
    shape_cam = np.array([1.0, 0.8, 0.6, 0.4])
    Y = np.zeros((nfreq, nnodes*4, ncases), complex)
    # component 3 = SX+SY driven by the resonance
    Y[:, 3::4, 0] = shape_cam[None, :] / D[:, None]
    out = extract_mode_shapes(Y, freq, n_modes=1)
    assert abs(out["freqs"][0] - 54.0) < 1.0
    # extracted cam shape is proportional to the true shape
    est = out["psi"]["SX+SY"][:, 0]
    assert abs(abs(np.vdot(est, shape_cam))**2 /
               (np.vdot(est, est).real * shape_cam @ shape_cam) - 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic_validation/test_mode_extraction.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation** (append)

```python
from scipy.signal import find_peaks

def _spatial_rms_over_cases(Y_comp):        # Y_comp (nfreq, nnodes, ncases)
    return np.sqrt(np.mean(np.abs(Y_comp)**2, axis=(1, 2)))

def extract_mode_shapes(Y_SEMM, freq_axis, n_modes=2, prominence=None):
    cam = Y_SEMM[:, 3::4, :]                              # SX+SY (nfreq, nnodes, ncases)
    rms = _spatial_rms_over_cases(cam)
    peaks, props = find_peaks(rms, prominence=(prominence or np.ptp(rms) * 0.05))
    order = np.argsort(rms[peaks])[::-1][:n_modes]
    peak_idx = np.sort(peaks[order])
    comp_slices = {"SX": 0, "SY": 1, "SXY": 2, "SX+SY": 3}
    psi = {}
    for c, off in comp_slices.items():
        Hc = Y_SEMM[:, off::4, :]                         # (nfreq, nnodes, ncases)
        shapes = [np.mean(np.abs(Hc[fi, :, :]), axis=1) for fi in peak_idx]
        psi[c] = np.column_stack(shapes) if shapes else np.zeros((Hc.shape[1], 0))
    return {"freqs": freq_axis[peak_idx], "psi": psi}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic_validation/test_mode_extraction.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add synthetic_validation/expansion.py tests/synthetic_validation/test_mode_extraction.py
git commit -m "feat(synval): peak-based mode-shape extraction from Y_SEMM"
```

---

### Task 8: SEMM Stage-1 adapter — run the real SEMM on in-memory parent + overlay FRFs

**Files:**
- Create: `synthetic_validation/semm_adapter.py`
- Test: `tests/synthetic_validation/test_semm_adapter.py`

**Purpose & approach:** This is the one reuse boundary with real interface uncertainty. Rather than guess, the task begins with a bounded read of the existing SEMM path and pins the interface with a **characterization test** against the existing `Y_SEMM_stage1.npy`. The deliverable is a function that takes synthetic parent full-tensor FRFs + a synthetic camera-component overlay and returns a hybrid `Y_SEMM` of shape `(nfreq, nnodes*4, ncases)`.

**Interfaces:**
- Produces:
  - `run_semm_stage1(parent_frf, overlay_frf, node_coords, freq_axis, cfg) -> np.ndarray` where `parent_frf` is `(nfreq, nnodes*4, ncases)` complex (parent model, all components), `overlay_frf` is `(nfreq, npixels, ncases)` complex (truth SX+SY at camera pixels, noisy). Returns hybrid `Y_SEMM` `(nfreq, nnodes*4, ncases)`.

- [ ] **Step 1: Investigate the existing SEMM call.** Read `dual_stage_base_pipeline/semm_thermoelastic_pipeline.py` methods `_run_chunked_semm` (line ~853), `build_hammer_stage1` (line ~770), `_build_overlay_interface` (line ~722), and `run_stage1_numerical_and_semm` (line ~959). Identify the exact pyFBS SEMM entry point, the DoF/interface bookkeeping (which nodes/components are interface DoFs), and the array layout it expects. Write findings as a docstring in `semm_adapter.py`.

- [ ] **Step 2: Write the characterization test** (pins the interface against real data)

```python
# tests/synthetic_validation/test_semm_adapter.py
import numpy as np, pytest
from pathlib import Path
from synthetic_validation.semm_adapter import run_semm_stage1

REF = Path("dual_stage_base_pipeline/outputs/20260219_090812/stage1/Y_SEMM_stage1.npy")

@pytest.mark.skipif(not REF.exists(), reason="reference SEMM output not present")
def test_semm_shape_and_interface_consistency():
    Y_ref = np.load(REF)                                   # (369, 3364, 6)
    nfreq, ndof, ncases = Y_ref.shape
    nnodes = ndof // 4
    # feed the reference's own SX+SY as overlay and its full tensor as parent:
    parent = Y_ref.copy()
    overlay = Y_ref[:, 3::4, :]                            # SX+SY as pseudo-camera
    Y = run_semm_stage1(parent, overlay, node_coords=np.zeros((nnodes,3)),
                        freq_axis=np.linspace(45,300,nfreq), cfg=None)
    assert Y.shape == Y_ref.shape
    # SEMM with overlay == parent's own invariant should be close to the parent
    assert np.linalg.norm(Y - parent) / np.linalg.norm(parent) < 0.25
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/synthetic_validation/test_semm_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError` (or SKIP if reference absent — then create a small synthetic `Y_ref` fixture in the test instead).

- [ ] **Step 4: Implement `run_semm_stage1`** by extracting the SEMM-core call from `_run_chunked_semm` into a function that accepts in-memory arrays instead of loading files. Reuse pyFBS exactly as the engine does; do not re-implement SEMM. (Exact body determined by Step 1; must call the same pyFBS `SEMM` routine with parent = FE tensor FRF and overlay = camera-component FRF at interface DoFs.)

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/synthetic_validation/test_semm_adapter.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add synthetic_validation/semm_adapter.py tests/synthetic_validation/test_semm_adapter.py
git commit -m "feat(synval): SEMM stage-1 adapter for in-memory synthetic FRFs"
```

---

### Task 9: `fe_models.py` — config→MAPDL wrapper with caching + solver extension

**Files:**
- Create: `synthetic_validation/fe_models.py`
- Modify: `stage1_solver/plate_stress_modal_superposition_force_pointmass_v2_camfreq_20260206_1239.py`
- Test: `tests/synthetic_validation/test_fe_models.py`

**Interfaces:**
- Produces:
  - `generate_fe(cfg, force_label, runner=default_mapdl_runner, cache_dir="synthetic_validation/cache") -> dict` with keys `"stress_tensor_frf" (nfreq,nnodes,4) complex`, `"freqs"`, `"node_coords"`, `"modal_data"` (the Task-1 contract dict, incl. `modal_sx/sy/sxy`, `gamma_base`). Caches to `cache_dir/<config_hash>_<force_label>.npz`; if present, loads instead of running MAPDL.
  - `default_mapdl_runner(cfg, force_label, out_dir) -> dict` — sets env vars and invokes the solver script (subprocess), then loads its outputs.
  - `build_gamma_base(modal_uz, lumped_mass, gen_mass) -> np.ndarray` — `Γ_r = Σ_n m_n·φ_{r,n,z} / M_r`.

- [ ] **Step 1: Write the failing test** (caching + gamma, with a fake runner — no ANSYS)

```python
# tests/synthetic_validation/test_fe_models.py
import numpy as np
from synthetic_validation.fe_models import generate_fe, build_gamma_base
from synthetic_validation.config import load_config

def test_gamma_base_formula():
    modal_uz = np.array([[1.0, 1.0, 1.0, 1.0]])       # 1 mode, 4 nodes
    m = np.array([0.25, 0.25, 0.25, 0.25]); Mi = np.array([1.0])
    g = build_gamma_base(modal_uz, m, Mi)
    assert np.allclose(g, [1.0])                        # sum(m*uz)/Mi = 1.0

def test_generate_fe_uses_cache(tmp_path):
    calls = {"n": 0}
    def fake_runner(cfg, force_label, out_dir):
        calls["n"] += 1
        nn = 4
        return {"stress_tensor_frf": np.zeros((10, nn, 4), complex),
                "freqs": np.linspace(45, 200, 10), "node_coords": np.zeros((nn, 3)),
                "modal_data": {"modal_freqs": np.array([54.0]), "modal_omega": np.array([339.]),
                               "zeta": np.array([0.005]), "node_coords": np.zeros((nn,3)),
                               "modal_sx": np.zeros((1,nn)), "modal_sy": np.zeros((1,nn)),
                               "modal_sxy": np.zeros((1,nn)), "gamma_base": np.array([1.0]),
                               "modal_mass": np.array([1.0])}}
    cfg = load_config("synthetic_validation/configs/parent.json")
    a = generate_fe(cfg, "center_middle", runner=fake_runner, cache_dir=tmp_path)
    b = generate_fe(cfg, "center_middle", runner=fake_runner, cache_dir=tmp_path)
    assert calls["n"] == 1                              # second call hits cache
    assert a["stress_tensor_frf"].shape == b["stress_tensor_frf"].shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic_validation/test_fe_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Extend the solver** to save modal stress shapes + base-participation inputs. In the solver's final `np.savez(modal_data_outfile, ...)` call, add `modal_sx=modal_sx, modal_sy=modal_sy, modal_sxy=modal_sxy, modal_uz=modal_uz, lumped_mass=m`. (These arrays already exist in the solver; they are currently not saved.)

- [ ] **Step 4: Write `fe_models.py`** with `build_gamma_base`, a cache layer keyed by `config_hash(cfg)+force_label` using `np.savez`/`np.load`, and `default_mapdl_runner` that sets env (`FORCE_X/FORCE_Y`, `GRID_NX/NY`, `PLATE_LX/LY`, `POINT_MASS*`, `FREQS_FILE`) + runs the solver via subprocess and assembles the return dict (computing `gamma_base` from saved `modal_uz`+`lumped_mass`+`modal_mass`). The cache/`build_gamma_base` code paths are exercised by the fake runner in tests; the MAPDL path is verified manually in Step 6.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/synthetic_validation/test_fe_models.py -v`
Expected: PASS

- [ ] **Step 6: Manual MAPDL verification** (requires ANSYS; not in CI). Run `generate_fe(load_config('.../truth.json'), 'center_middle')` once. Confirm: MAPDL runs, `modal_data` contains non-empty `modal_sx/sy/sxy` and a finite `gamma_base`, and a cache `.npz` is written. Record the two modal frequencies (~54, ~176 Hz) as a sanity check against the paper.

- [ ] **Step 7: Commit**

```bash
git add synthetic_validation/fe_models.py tests/synthetic_validation/test_fe_models.py stage1_solver/
git commit -m "feat(synval): FE generation wrapper + caching; solver saves modal stress shapes"
```

---

### Task 10: End-to-end integration — oracle recovery equals truth (correctness gate)

**Files:**
- Create: `synthetic_validation/harness.py`
- Test: `tests/synthetic_validation/test_integration_oracle.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `run_case(truth_cfg, parent_cfg, *, noise=True, n_modes=2, saa_level, fs, n_frames, seed, use_truth_as_prior=False) -> dict` with keys `"recovered": {comp: (nfreq,nnodes)}`, `"truth": {comp: (nfreq,nnodes)}`, `"metrics": {...}`, `"freqs"`, `"condition_number"`. When `use_truth_as_prior=True` the SEMM parent is the truth model (oracle/cheat run).

- [ ] **Step 1: Write the failing integration test** (analytic path, no ANSYS: inject a modal_data directly; parent==truth; zero noise → recovery matches truth)

```python
# tests/synthetic_validation/test_integration_oracle.py
import numpy as np
from synthetic_validation.harness import run_case_from_modal   # analytic entry, bypasses MAPDL+SEMM

def test_oracle_zero_noise_recovers_truth(single_mode_modal):
    # parent == truth modal data, no noise, mode shapes taken as exact:
    res = run_case_from_modal(single_mode_modal, single_mode_modal,
                              saa_level=1.0, fs=2000, n_frames=40000, seed=0,
                              noise_sigma_T=0.0, n_modes=1)
    # recovered SX PSD should match truth SX PSD at the resonance band within a few %
    m = res["metrics"]
    assert m["nrmse"]["SX"] < 0.05
    assert m["nrmse"]["SXY"] < 0.10
    assert m["mac"]["SX"] > 0.98
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic_validation/test_integration_oracle.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `harness.py`.** Provide `run_case_from_modal(truth_md, prior_md, ...)` that: (a) synthesizes camera frames + accel from `truth_md` (Task 4), injects noise (Task 5), (b) computes `T_cam` via H1 on the synthetic signals using the existing engine's `frf_ir_h1`/`csd`/`welch` path (or scipy directly, matching `method_transmissibility_expansion.md` §2.3), (c) builds mode shapes from `prior_md` analytically as `psi_by_comp[c] = prior_md["modal_s*"]` transposed to `(nnodes, nmodes)` with `SX+SY = SX+SY`, normalizes (Task 6), (d) `modal_decompose` → `expand_components` → `stress_psd`, (e) computes truth PSD via `truth_component_psd(truth_md,...)`, (f) scores with `metrics`. Also implement the full `run_case(truth_cfg, parent_cfg, ...)` that routes through `fe_models` + `semm_adapter` + `extract_mode_shapes` for the real (MAPDL+SEMM) path, reusing the same steps (b)–(f).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic_validation/test_integration_oracle.py -v`
Expected: PASS (this is the key correctness gate: with a perfect prior and no noise the method reproduces the truth components).

- [ ] **Step 5: Commit**

```bash
git add synthetic_validation/harness.py tests/synthetic_validation/test_integration_oracle.py
git commit -m "feat(synval): end-to-end harness + oracle correctness gate"
```

---

### Task 11: `studies.py` — the five reviewer studies + presentation notebook

**Files:**
- Create: `synthetic_validation/studies.py`
- Create: `synthetic_validation/notebooks/synthetic_validation_results.ipynb`
- Test: `tests/synthetic_validation/test_studies.py`

**Interfaces:**
- Produces (each returns a results dict ready to plot/tabulate):
  - `study_a_recovery(truth_cfg, parent_cfg, **kw) -> dict` — realistic + oracle run; per-component NRMSE/MAC/per-point; spatial maps.
  - `study_c_fe_discrepancy(truth_cfg, parent_variants, **kw) -> dict` — error vs each perturbation; realistic−oracle gap.
  - `study_b_noise(truth_cfg, parent_cfg, n_reps, snr_levels, **kw) -> dict` — Monte-Carlo confidence bands + repeatability stats.
  - `study_d_modal_convergence(truth_cfg, parent_cfg, mode_counts, **kw) -> dict` — error vs #modes.
  - `study_e_conditioning(truth_cfg, parent_cfg, regularize, **kw) -> dict` — condition number, singular values, reg on/off.

- [ ] **Step 1: Write the failing test** (drivers run on the analytic path and return the documented keys)

```python
# tests/synthetic_validation/test_studies.py
from synthetic_validation.studies import study_b_noise
from tests.synthetic_validation.conftest import make_modal_data
import numpy as np

def test_study_b_returns_bands(single_mode_modal):
    res = study_b_noise(single_mode_modal, single_mode_modal, n_reps=5,
                        snr_levels=[1.0], analytic=True, fs=2000, n_frames=8000,
                        saa_level=1.0, n_modes=1)
    assert "confidence_bands" in res and "SX" in res["confidence_bands"]
    assert res["n_reps"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic_validation/test_studies.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `studies.py`.** Each driver loops `run_case`/`run_case_from_modal` (analytic path when `analytic=True` for tests; MAPDL+SEMM path otherwise). Study B collects `n_reps` recoveries per SNR (varying `seed`), returns per-frequency mean ± std as `confidence_bands`. Study C builds parent variants by mutating one `FEConfig` field at a time (E ±10%, thickness ±X, point_mass on/off) and records NRMSE vs the perturbation plus the realistic−oracle gap. Study D varies `n_modes`. Study E reports `condition_number` from the decomposition and, if `regularize`, swaps `pinv` for a truncated-SVD/Tikhonov variant and reports the effect.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic_validation/test_studies.py -v`
Expected: PASS

- [ ] **Step 5: Create the presentation notebook** that imports the studies, runs A–E with the real configs, and produces the figures/tables for the response letter (spatial recovered-vs-truth maps, error-vs-discrepancy curves, noise bands, convergence curve, conditioning table). Thin — logic lives in `studies.py`.

- [ ] **Step 6: Commit**

```bash
git add synthetic_validation/studies.py synthetic_validation/notebooks/ tests/synthetic_validation/test_studies.py
git commit -m "feat(synval): studies A-E drivers + results notebook"
```

---

## Self-Review

**Spec coverage:**
- §2 truth/parent + realistic/oracle → Tasks 8, 9, 10 (`use_truth_as_prior`). ✓
- §5.1 fe_models + solver extension → Task 9. ✓
- §5.2 forward model (base-exc FRF + synthesis) → Tasks 3, 4. ✓
- §5.3 noise calibration/injection → Task 5. ✓
- §5.4 expansion reuse → Tasks 6, 7, 8. ✓
- §5.5 metrics → Task 2. ✓
- §5.6 studies A–E → Task 11. ✓
- §6 testing (zero-noise==truth, single-mode closed form, noise reproduction, metrics identities, caching) → Tasks 10, 3, 5, 2, 9. ✓
- §7 reviewer map → covered by Task 11 studies. ✓

**Placeholder scan:** Task 8 Step 4 and Task 9 Step 4 intentionally defer *exact body* to a documented investigation because they wrap external code (pyFBS SEMM, MAPDL) whose interface must be read first; both are pinned by a concrete test (characterization / cache+gamma) rather than left vague. All pure-numpy tasks contain complete code. No TODO/TBD in requirement text.

**Type consistency:** `modal_data` dict keys are identical across Tasks 1, 3, 4, 9, 10. Component keys `SX/SY/SXY/SX+SY` are consistent across Tasks 3, 6, 7, 10, 11. `Y_SEMM` shape `(nfreq, nnodes*4, ncases)` and the `[:, c::4]` slicing convention are consistent across Tasks 7, 8. `run_case`/`run_case_from_modal` return dict keys match between Tasks 10 and 11.

**Known limitations to confirm during execution (from spec §8):** exact `gamma_base` saved form (Task 9 Step 3), Stage-1 overlay noise model, default `S_aa`, headline Study-C perturbation, and Study-A success thresholds — these are flagged in-task and do not block the build.
