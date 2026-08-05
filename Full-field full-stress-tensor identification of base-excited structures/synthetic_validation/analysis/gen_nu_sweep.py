"""Poisson-ratio sensitivity SWEEP (reviewer R2.1), for a line plot instead of two bars.

The main sweep (gen_robustness_sweep_amp.py) only had nu = 0.33 (truth) and 0.25. To draw
Poisson's sensitivity as a line (matching the Young's-modulus panel), we sweep the prior's
nu over a range while holding everything else at the E63 flat-plate baseline. The truth keeps
nu = 0.33, so the prior nu is a pure modelling error. sigma_xx should dip to its block-only
floor at nu = 0.33 (the true value) and rise as nu deviates in either direction.

Metrics in the AMPLITUDE domain (band-RMS stress field, 45-250 Hz). Writes a separate file
broad_analysis/robustness_sweep_nu.json.
"""
import os, sys, json, time
from dataclasses import replace
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["POINT_MASS_PATCH_NX"] = "5"
os.environ["POINT_MASS_PATCH_NY"] = "5"

from synthetic_validation.config import load_config
from synthetic_validation.fe_models import generate_fe
from synthetic_validation.semm_adapter import run_semm_stage1
from synthetic_validation.expansion import extract_mode_shapes
from synthetic_validation.harness import run_case_from_modal, _mode_data_to_modal_dict
from synthetic_validation.forward_model import base_excitation_frf

def _vm_psd(T, Saa):    # plane-stress von Mises equivalent-stress PSD from complex T
    return Saa[:, None] * (np.abs(T["SX"])**2 + np.abs(T["SY"])**2
                           - np.real(T["SX"] * np.conj(T["SY"])) + 3 * np.abs(T["SXY"])**2)

HERE = os.path.dirname(__file__)
CFG = os.path.join(HERE, "..", "configs")
OUT = os.path.join(HERE, "broad_analysis", "robustness_sweep_nu.json")
COMPS = ["SX", "SY", "SXY", "SX+SY"]
ACQ = dict(saa_level=1.0, fs=2000.0, n_frames=10000, seed=0)

t0 = load_config(os.path.join(CFG, "truth.json"))
p0 = load_config(os.path.join(CFG, "parent.json"))
dx = t0.plate_lx / (t0.grid_nx - 1)
truth = replace(t0, E=69e9, nu=0.33, point_mass=0.17, point_mass_xy=(2 * dx, 0.14))
base = replace(p0, E=63e9, nu=0.33, rho=2700, thickness=0.003, point_mass=0.0)

NU_VALUES = [0.25, 0.28, 0.31, 0.33, 0.36, 0.39]     # includes truth (0.33) and cited (0.25)

def stack(fes):
    parts = []
    for fe in fes:
        frf = fe["stress_tensor_frf"]
        if frf.ndim == 3:
            nf, nn, nc = frf.shape
            frf = frf.reshape(nf, nn * nc)
        parts.append(frf[:, :, None])
    return np.concatenate(parts, axis=2)

def solve_retry(cfg, fl, tries=4):
    """generate_fe with retries -- MAPDL license grabs can transiently time out."""
    for t in range(tries):
        try:
            return generate_fe(cfg, fl)
        except Exception as e:
            if t == tries - 1:
                raise
            print(f"    solve retry {t+1}/{tries-1} after: {str(e)[:70]}", flush=True)
            time.sleep(15)

labels = list(truth.force_points.keys())
print("solving truth (cached)...", flush=True)
truth_fes = [solve_retry(truth, fl) for fl in labels]
truth_modal = truth_fes[0]["modal_data"]
node_coords = truth_fes[0]["node_coords"]
freq_axis = truth_fes[0]["freqs"]
overlay = stack(truth_fes)[:, 3::4, :]

results = json.load(open(OUT)) if os.path.exists(OUT) else []          # resume
done = {round(r["nu"], 3) for r in results}
for nu in NU_VALUES:
    if round(nu, 3) in done:
        print(f"\n=== nu = {nu} (cached, skip) ===", flush=True); continue
    pcfg = replace(base, nu=nu)
    print(f"\n=== nu = {nu} ===", flush=True)
    p_fes = [solve_retry(pcfg, fl) for fl in labels]
    Y = run_semm_stage1(stack(p_fes), overlay, node_coords, freq_axis)
    prior = _mode_data_to_modal_dict(extract_mode_shapes(Y, freq_axis, n_modes=2),
                                     node_coords, truth_modal)
    r = run_case_from_modal(truth_modal, prior, n_modes=2, **ACQ)

    fr = r["freqs"]; df = float(fr[1] - fr[0]); band = (fr >= 45) & (fr <= 250)
    brms = lambda P: np.sqrt(np.sum(np.abs(P)[band], axis=0) * df)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))

    amp = {c: rel(brms(r["recovered"][c]), brms(r["truth"][c])) for c in COMPS}
    Tt = base_excitation_frf(truth_modal, fr)                             # von Mises (fatigue driver)
    amp["VM"] = rel(brms(_vm_psd(r["T_by_comp"], r["S_aa"])), brms(_vm_psd(Tt, r["S_aa"])))
    results.append(dict(nu=nu, amp=amp))
    with open(OUT, "w") as f:                                          # incremental save
        json.dump(results, f, indent=2)
    print("  AMP  " + "  ".join(f"{c} {amp[c]*100:5.1f}%" for c in COMPS), flush=True)

print("\n========== POISSON-RATIO SENSITIVITY (amplitude domain) ==========")
print(f"{'nu':>6s}" + "".join(f"{c:>9s}" for c in COMPS))
for r in results:
    a = r["amp"]
    print(f"{r['nu']:6.2f}" + "".join(f"{a[c]*100:8.1f}%" for c in COMPS))
print("\nSaved", OUT)
