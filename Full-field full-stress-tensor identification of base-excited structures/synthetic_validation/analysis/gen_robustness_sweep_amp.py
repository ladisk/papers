"""Prior-robustness sweep in the AMPLITUDE domain (band-RMS stress field, 45-250 Hz).

Same pipeline as gen_robustness_sweep.py (6 FE solves -> SEMM stage-1 -> signed mode
extraction -> transmissibility expansion), but the metrics are computed on the
band-RMS STRESS field (linear in stress) rather than on the PSD (quadratic).
This is the domain the method actually works in, and the domain the response-letter
tables now report. PSD values are kept alongside for cross-reference only.
"""
import os, sys, json
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
OUT = os.path.join(HERE, "broad_analysis", "robustness_sweep_amp.json")
COMPS = ["SX", "SY", "SXY", "SX+SY"]
ACQ = dict(saa_level=1.0, fs=2000.0, n_frames=10000, seed=0)

t0 = load_config(os.path.join(CFG, "truth.json"))
p0 = load_config(os.path.join(CFG, "parent.json"))
dx = t0.plate_lx / (t0.grid_nx - 1)
truth = replace(t0, E=69e9, point_mass=0.17, point_mass_xy=(2 * dx, 0.14))

PARENTS = [
    ("P0  E69 t3.0",          replace(p0, E=69e9, nu=0.33, rho=2700, thickness=0.003,  point_mass=0.0)),
    ("P1  E63 t3.0",          replace(p0, E=63e9, nu=0.33, rho=2700, thickness=0.003,  point_mass=0.0)),
    ("P2  E55 t2.8",          replace(p0, E=55e9, nu=0.33, rho=2700, thickness=0.0028, point_mass=0.0)),
    ("P3  E45 t2.6 rho2550",  replace(p0, E=45e9, nu=0.33, rho=2550, thickness=0.0026, point_mass=0.0)),
    ("P4  E38 t2.4 rho2400",  replace(p0, E=38e9, nu=0.33, rho=2400, thickness=0.0024, point_mass=0.0)),
    ("P5  nu=0.25 (ratio!)",  replace(p0, E=63e9, nu=0.25, rho=2700, thickness=0.003,  point_mass=0.0)),
]

def stack(fes):
    parts = []
    for fe in fes:
        frf = fe["stress_tensor_frf"]
        if frf.ndim == 3:
            nf, nn, nc = frf.shape
            frf = frf.reshape(nf, nn * nc)
        parts.append(frf[:, :, None])
    return np.concatenate(parts, axis=2)

labels = list(truth.force_points.keys())
print("solving truth (cached)...", flush=True)
truth_fes = [generate_fe(truth, fl) for fl in labels]
truth_modal = truth_fes[0]["modal_data"]
node_coords = truth_fes[0]["node_coords"]
freq_axis = truth_fes[0]["freqs"]
overlay = stack(truth_fes)[:, 3::4, :]

results = []
for name, pcfg in PARENTS:
    print(f"\n=== {name} ===", flush=True)
    p_fes = [generate_fe(pcfg, fl) for fl in labels]
    Y = run_semm_stage1(stack(p_fes), overlay, node_coords, freq_axis)
    prior = _mode_data_to_modal_dict(extract_mode_shapes(Y, freq_axis, n_modes=2),
                                     node_coords, truth_modal)
    r = run_case_from_modal(truth_modal, prior, n_modes=2, **ACQ)

    fr = r["freqs"]; df = float(fr[1] - fr[0]); band = (fr >= 45) & (fr <= 250)
    def brms(P):                       # band-RMS STRESS field (Pa) -> amplitude
        return np.sqrt(np.sum(np.abs(P)[band], axis=0) * df)
    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    amp, psd = {}, {}
    for c in COMPS:
        t_b, r_b = brms(r["truth"][c]), brms(r["recovered"][c])
        amp[c] = rel(r_b, t_b)                                             # amplitude
        psd[c] = rel(np.abs(r["recovered"][c]), np.abs(r["truth"][c]))     # old PSD metric
    Tt = base_excitation_frf(truth_modal, fr)                             # von Mises (fatigue driver)
    amp["VM"] = rel(brms(_vm_psd(r["T_by_comp"], r["S_aa"])), brms(_vm_psd(Tt, r["S_aa"])))
    results.append(dict(name=name, E=pcfg.E, nu=pcfg.nu, thickness=pcfg.thickness,
                        rho=pcfg.rho, amp=amp, psd=psd))
    print("  AMP  " + "  ".join(f"{c} {amp[c]*100:5.1f}%" for c in COMPS), flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

print("\n=============== AMPLITUDE-DOMAIN SEMM-CORRECTED SWEEP ===============")
print(f"{'parent':24s} {'sig_xx':>8s} {'sig_yy':>8s} {'tau_xy':>8s} {'invar':>8s}")
for r in results:
    a = r["amp"]
    print(f"{r['name']:24s} {a['SX']*100:7.1f}% {a['SY']*100:7.1f}% {a['SXY']*100:7.1f}% {a['SX+SY']*100:7.1f}%")
base = results[1]["amp"]                      # P1 = the baseline prior used in the paper
print("\nmax |change| vs the E63 baseline, over the ratio-NEUTRAL priors (P0,P2,P3,P4):")
for c in COMPS:
    d = max(abs(results[i]["amp"][c] - base[c]) for i in (0, 2, 3, 4)) * 100
    print(f"  {c:6s} {d:5.2f} pp")
print("\nratio-CHANGING prior (P5, nu=0.25) vs baseline:")
for c in COMPS:
    print(f"  {c:6s} {base[c]*100:5.1f}%  ->  {results[5]['amp'][c]*100:5.1f}%")
print("\nSaved", OUT)
