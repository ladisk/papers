"""Clamping-stiffness / boundary-condition sensitivity (reviewers R1.3 / R2.1).

The prior-degradation sweep (gen_robustness_sweep_amp.py) varies E, thickness, density
and nu. Both reviewers ALSO asked about boundary conditions and clamping stiffness,
which that sweep never touched -- the letters covered them with a physical argument.
This script measures them.

The PRIOR's rigid clamp is replaced by a COMPLIANT one: the bending rotation about the
clamp line is restrained by a torsional spring of k_rot [N*m/rad] per base node instead
of being rigid. The truth keeps its rigid clamp, so k_rot is a pure prior-modelling error.

  k_rot = 1e3 /node  ->  prior's f1 low by  1.1%   (a mild clamp-modelling error)
  k_rot = 1e2 /node  ->  prior's f1 low by  9.5%   (a severe one -- larger than the
                                                    7% offset our truth model itself has)

NOTE: a fully PINNED edge is NOT a valid degraded clamp. A cantilever whose only support
carries no moment is a hinge: the model returns a 0 Hz rigid-body mode and the resulting
"sensitivity" is a mechanism artefact, not a boundary-condition effect. Hence springs.

Metrics are in the AMPLITUDE domain (band-RMS stress field, 45-250 Hz), matching the
letters' tables. Appends to broad_analysis/robustness_sweep_amp.json.
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

HERE = os.path.dirname(__file__)
CFG = os.path.join(HERE, "..", "configs")
OUT = os.path.join(HERE, "broad_analysis", "robustness_sweep_amp.json")
COMPS = ["SX", "SY", "SXY", "SX+SY"]
ACQ = dict(saa_level=1.0, fs=2000.0, n_frames=10000, seed=0)

t0 = load_config(os.path.join(CFG, "truth.json"))
p0 = load_config(os.path.join(CFG, "parent.json"))
dx = t0.plate_lx / (t0.grid_nx - 1)
truth = replace(t0, E=69e9, point_mass=0.17, point_mass_xy=(2 * dx, 0.14))   # rigid clamp
base = replace(p0, E=63e9, nu=0.33, rho=2700, thickness=0.003, point_mass=0.0)

PARENTS = [
    ("C1  clamp k=1e3 (BC)", replace(base, base_bc="spring:1e3")),
    ("C2  clamp k=1e2 (BC)", replace(base, base_bc="spring:1e2")),
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

results = json.load(open(OUT)) if os.path.exists(OUT) else []
results = [r for r in results if not r["name"].startswith(("P6", "C1", "C2"))]   # idempotent

for name, pcfg in PARENTS:
    print(f"\n=== {name}  (base_bc={pcfg.base_bc}) ===", flush=True)
    p_fes = [generate_fe(pcfg, fl) for fl in labels]
    pf = np.asarray(p_fes[0]["modal_data"]["modal_freqs"]).ravel()
    print(f"  prior modal freqs: {np.round(pf[:2], 1)}  (rigid-clamp prior: [108.4 261.4])", flush=True)
    assert pf[0] > 1.0, "rigid-body mode -> invalid clamp model"

    Y = run_semm_stage1(stack(p_fes), overlay, node_coords, freq_axis)
    prior = _mode_data_to_modal_dict(extract_mode_shapes(Y, freq_axis, n_modes=2),
                                     node_coords, truth_modal)
    r = run_case_from_modal(truth_modal, prior, n_modes=2, **ACQ)

    fr = r["freqs"]; df = float(fr[1] - fr[0]); band = (fr >= 45) & (fr <= 250)
    brms = lambda P: np.sqrt(np.sum(np.abs(P)[band], axis=0) * df)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))

    amp, psd = {}, {}
    for c in COMPS:
        amp[c] = rel(brms(r["recovered"][c]), brms(r["truth"][c]))
        psd[c] = rel(np.abs(r["recovered"][c]), np.abs(r["truth"][c]))
    results.append(dict(name=name, E=pcfg.E, nu=pcfg.nu, thickness=pcfg.thickness,
                        rho=pcfg.rho, base_bc=pcfg.base_bc, amp=amp, psd=psd))
    print("  AMP  " + "  ".join(f"{c} {amp[c]*100:5.1f}%" for c in COMPS), flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

ref = [r for r in results if r["name"].startswith("P1")][0]["amp"]
print("\n========== CLAMPING-STIFFNESS SENSITIVITY (amplitude domain) ==========")
print(f"{'prior clamp':24s}" + "".join(f"{c:>9s}" for c in COMPS))
print(f"{'rigid (nominal)':24s}" + "".join(f"{ref[c]*100:8.1f}%" for c in COMPS))
for r in results:
    if r["name"].startswith("C"):
        a = r["amp"]
        print(f"{r['name']:24s}" + "".join(f"{a[c]*100:8.1f}%" for c in COMPS))
print("\nchange vs the rigid-clamp prior:")
for r in results:
    if r["name"].startswith("C"):
        a = r["amp"]
        print(f"  {r['name']:22s}" + "".join(f"{(a[c]-ref[c])*100:+7.2f} pp " for c in COMPS))
print("\nSaved", OUT)
