"""
Phase-0 generation for the BROAD tau_xy analysis.

Produces the faithful corner-mass dataset (truth = real-Al 69 GPa + 0.17 kg
top-left-corner mass; parent = paper's generic 63 GPa, NO mass) and saves all
modal-data variants the downstream analytic analyses need (run_case_from_modal
needs no MAPDL, so agents can sweep n_modes / noise / priors freely).

Outputs (in synthetic_validation/analysis/broad_analysis/):
  validation_fields.npz            <- ALSO copied to figures_data/ for the figures
  modal_truth.npz                  truth signed modal data (answer key)
  modal_prior_semm.npz             SEMM-extracted signed prior (the real method's prior)
  modal_parent_paper.npz           parent E63 no mass (paper's FE)
  modal_parent_E69.npz             parent E69 no mass (isolate the E effect)
  modal_parent_symmass.npz         parent E63 + symmetric 0.17 kg top-centre mass
  summary.json                     realistic / oracle / parent-oracle metrics
"""
import os, sys, json, shutil
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
OUTD = os.path.join(HERE, "broad_analysis")
os.makedirs(OUTD, exist_ok=True)
CFG = os.path.join(HERE, "..", "configs")
COMPS = ["SX", "SY", "SXY", "SX+SY"]
ACQ = dict(saa_level=1.0, fs=2000.0, n_frames=10000, seed=0)

t0 = load_config(os.path.join(CFG, "truth.json"))
p0 = load_config(os.path.join(CFG, "parent.json"))
dx = t0.plate_lx / (t0.grid_nx - 1)
corner = (2 * dx, 0.14)          # top-left, 5x5 patch anchors on the corner
centre = (0.075, 0.14)           # top-centre (symmetric)

# Faithful materials: truth = real 6061 Al 69 GPa; parent = paper's generic 63 GPa.
truth        = replace(t0, E=69e9, point_mass=0.17, point_mass_xy=corner)
parent_paper = replace(p0, E=63e9, point_mass=0.0,  point_mass_xy=corner)
parent_E69   = replace(p0, E=69e9, point_mass=0.0,  point_mass_xy=corner)
parent_sym   = replace(p0, E=63e9, point_mass=0.17, point_mass_xy=centre)

def save_modal(md, name):
    out = {k: np.asarray(v) for k, v in md.items()}
    np.savez(os.path.join(OUTD, name), **out)

def stack_frfs(fes):
    parts = []
    for fe in fes:
        frf = fe["stress_tensor_frf"]
        if frf.ndim == 3:
            nfreq, nn, nc = frf.shape
            frf = frf.reshape(nfreq, nn * nc)
        parts.append(frf[:, :, None])
    return np.concatenate(parts, axis=2)

print("=== solving truth (E69, corner mass) x6 forces ===")
labels = list(truth.force_points.keys())
truth_fes = [generate_fe(truth, fl) for fl in labels]
truth_modal = truth_fes[0]["modal_data"]
node_coords = truth_fes[0]["node_coords"]
freq_axis = truth_fes[0]["freqs"]

print("=== solving parent_paper (E63, no mass) x6 forces ===")
parent_fes = [generate_fe(parent_paper, fl) for fl in labels]

# SEMM stage-1 -> signed mode extraction -> prior modal dict (the real method's prior)
truth_frf = stack_frfs(truth_fes)
parent_frf = stack_frfs(parent_fes)
overlay = truth_frf[:, 3::4, :]
print("=== SEMM stage-1 + signed extraction ===")
Y_SEMM = run_semm_stage1(parent_frf, overlay, node_coords, freq_axis)
mode_data = extract_mode_shapes(Y_SEMM, freq_axis, n_modes=2)
prior_semm = _mode_data_to_modal_dict(mode_data, node_coords, truth_modal)

# Extra parent variants (modal only, 1 solve each) for the FE-discrepancy decomposition
print("=== solving parent variants (modal only) ===")
parent_E69_modal = generate_fe(parent_E69, labels[0])["modal_data"]
parent_sym_modal = generate_fe(parent_sym, labels[0])["modal_data"]
parent_paper_modal = parent_fes[0]["modal_data"]

save_modal(truth_modal, "modal_truth.npz")
save_modal(prior_semm, "modal_prior_semm.npz")
save_modal(parent_paper_modal, "modal_parent_paper.npz")
save_modal(parent_E69_modal, "modal_parent_E69.npz")
save_modal(parent_sym_modal, "modal_parent_symmass.npz")

def metrics(res):
    out = {}
    for c in COMPS:
        r = np.abs(res["recovered"][c]); t = np.abs(res["truth"][c])
        out[c] = dict(rel_rms=float(np.sqrt(np.sum((r-t)**2)/np.sum(t**2))),
                      mac=float(res["metrics"]["mac"][c]),
                      peak_ratio=float(res["metrics"]["peak_ratio"][c]))
    return out

print("=== analytic runs: realistic / oracle / parent-oracle ===")
realistic = run_case_from_modal(truth_modal, prior_semm, n_modes=2, **ACQ)
oracle    = run_case_from_modal(truth_modal, truth_modal, n_modes=2, **ACQ)
poracle   = run_case_from_modal(truth_modal, parent_paper_modal, n_modes=2, **ACQ)

# Save the field arrays from the realistic run for the figures
freqs = realistic["freqs"]
se = np.sum(np.abs(realistic["truth"]["SX+SY"]), axis=1)
fk = int(np.argmax(se)); cp = int(np.argmax(np.abs(realistic["truth"]["SX+SY"][fk, :])))
save = dict(x=node_coords[:, 0], y=node_coords[:, 1], freqs=freqs, fk=fk, cp=cp,
            comps=np.array(COMPS),
            metrics_relrms=np.array([np.sqrt(np.sum((np.abs(realistic["recovered"][c])-np.abs(realistic["truth"][c]))**2)/np.sum(np.abs(realistic["truth"][c])**2)) for c in COMPS]),
            metrics_mac=np.array([realistic["metrics"]["mac"][c] for c in COMPS]),
            metrics_nrmse=np.array([realistic["metrics"]["nrmse"][c] for c in COMPS]))
for c in COMPS:
    save[f"rec_{c}"] = realistic["recovered"][c]
    save[f"tru_{c}"] = realistic["truth"][c]
np.savez(os.path.join(OUTD, "validation_fields.npz"), **save)
shutil.copy(os.path.join(OUTD, "validation_fields.npz"),
            os.path.join(HERE, "..", "figures_data", "validation_fields.npz"))

summary = dict(
    truth_modal_freqs=[float(f) for f in truth_modal["modal_freqs"][:6]],
    truth_gamma_base=[float(abs(g)) for g in np.atleast_1d(truth_modal["gamma_base"])[:6]],
    resonance_hz=float(freqs[fk]), critical_node=cp,
    realistic=metrics(realistic), oracle=metrics(oracle), parent_oracle=metrics(poracle),
    acquisition=ACQ, truth_E=69e9, parent_E=63e9,
    corner_xy_mm=[corner[0]*1000, corner[1]*1000], patch="5x5 (~2cm)",
)
with open(os.path.join(OUTD, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== DONE ===")
print(json.dumps({k: summary[k] for k in ["truth_modal_freqs", "resonance_hz"]}, indent=2))
print("realistic rel-RMS %:", {c: round(realistic["metrics"]["mac"][c],3) for c in COMPS})
for c in COMPS:
    print(f"  {c:6s} realistic rel_rms={summary['realistic'][c]['rel_rms']*100:5.1f}%  "
          f"oracle={summary['oracle'][c]['rel_rms']*100:4.1f}%  parent-oracle={summary['parent_oracle'][c]['rel_rms']*100:5.1f}%")
