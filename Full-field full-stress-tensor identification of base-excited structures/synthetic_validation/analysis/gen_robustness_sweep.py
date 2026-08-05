"""
Prior-robustness sweep: does SEMM converge to the same ~21% tau_xy floor regardless
of how bad the parent FE prior is?

For each parent we run the FULL pipeline (6 force-case FE solves -> SEMM stage-1 ->
signed mode extraction -> transmissibility expansion) and record BOTH:
  - parent-alone error  = run_case_from_modal(truth, parent_modal)   (no SEMM)
  - SEMM-corrected error = run_case_from_modal(truth, prior_semm)     (the method)

Ratio-NEUTRAL parents (E / thickness / density wrong, no block) should converge to
the same SEMM floor; a ratio-CHANGING parent (wrong Poisson nu) should NOT.
Truth is fixed (real-Al 69 GPa + 0.17 kg top-left corner block; solves cached).
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
OUT = os.path.join(HERE, "broad_analysis", "robustness_sweep.json")
COMPS = ["SX", "SY", "SXY", "SX+SY"]
ACQ = dict(saa_level=1.0, fs=2000.0, n_frames=10000, seed=0)

t0 = load_config(os.path.join(CFG, "truth.json"))
p0 = load_config(os.path.join(CFG, "parent.json"))
dx = t0.plate_lx / (t0.grid_nx - 1)
truth = replace(t0, E=69e9, point_mass=0.17, point_mass_xy=(2 * dx, 0.14))

# Parent variants (all WITHOUT the block). First three are ratio-neutral and
# increasingly wrong; the last changes the component ratios (Poisson).
PARENTS = [
    ("P1  E63 t3.0",          replace(p0, E=63e9, nu=0.33, rho=2700, thickness=0.003, point_mass=0.0)),
    ("P2  E55 t2.8",          replace(p0, E=55e9, nu=0.33, rho=2700, thickness=0.0028, point_mass=0.0)),
    ("P3  E45 t2.6 rho2550",  replace(p0, E=45e9, nu=0.33, rho=2550, thickness=0.0026, point_mass=0.0)),
    ("P4  E38 t2.4 rho2400",  replace(p0, E=38e9, nu=0.33, rho=2400, thickness=0.0024, point_mass=0.0)),
    ("P5  nu=0.25 (ratio!)",  replace(p0, E=63e9, nu=0.25, rho=2700, thickness=0.003, point_mass=0.0)),
]

def relrms(a, b):
    a, b = np.abs(a), np.abs(b)
    return float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))

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
print("solving truth (cached if available)...")
truth_fes = [generate_fe(truth, fl) for fl in labels]
truth_modal = truth_fes[0]["modal_data"]
node_coords = truth_fes[0]["node_coords"]
freq_axis = truth_fes[0]["freqs"]
truth_frf = stack(truth_fes)
overlay = truth_frf[:, 3::4, :]

results = []
for name, pcfg in PARENTS:
    print(f"\n=== {name} ===")
    p_fes = [generate_fe(pcfg, fl) for fl in labels]
    p_modal = p_fes[0]["modal_data"]
    # parent-alone (no SEMM)
    r_alone = run_case_from_modal(truth_modal, p_modal, n_modes=2, **ACQ)
    # SEMM-corrected
    Y = run_semm_stage1(stack(p_fes), overlay, node_coords, freq_axis)
    prior = _mode_data_to_modal_dict(extract_mode_shapes(Y, freq_axis, n_modes=2), node_coords, truth_modal)
    r_semm = run_case_from_modal(truth_modal, prior, n_modes=2, **ACQ)

    alone = {c: relrms(r_alone["recovered"][c], r_alone["truth"][c]) for c in COMPS}
    semm  = {c: relrms(r_semm["recovered"][c],  r_semm["truth"][c])  for c in COMPS}
    macs  = {c: float(r_semm["metrics"]["mac"][c]) for c in COMPS}
    results.append(dict(name=name, E=pcfg.E, nu=pcfg.nu, thickness=pcfg.thickness,
                        rho=pcfg.rho, alone=alone, semm=semm, semm_mac=macs))
    print(f"  parent-alone : SX {alone['SX']*100:5.1f}  SXY {alone['SXY']*100:5.1f}  inv {alone['SX+SY']*100:5.1f}")
    print(f"  SEMM-correct : SX {semm['SX']*100:5.1f}  SXY {semm['SXY']*100:5.1f}  inv {semm['SX+SY']*100:5.1f}  (SXY MAC {macs['SXY']:.3f})")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

print("\n================ SUMMARY (tau_xy) ================")
print(f"{'parent':22s} {'alone SXY':>10s} {'SEMM SXY':>10s} {'SEMM inv':>9s}")
for r in results:
    print(f"{r['name']:22s} {r['alone']['SXY']*100:9.1f}% {r['semm']['SXY']*100:9.1f}% {r['semm']['SX+SY']*100:8.1f}%")
print("\nSaved", OUT)
