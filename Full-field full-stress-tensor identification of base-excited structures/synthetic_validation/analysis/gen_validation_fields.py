"""
Regenerate validation_fields.npz for the recovery figures (V1-V4), using the
CORRECTED truth model: a 170 g steel block smeared over a 2x2 cm TOP-LEFT-CORNER
patch (matching the real experiment's asymmetric tip mass), which makes a second
mode base-active (~49 & ~179 Hz, cf. measured 54 & 175 Hz).

Truth  = parametric plate + 0.17 kg corner mass  (data source + answer key)
Parent = parametric plate, NO mass               (deliberately rough SEMM prior,
                                                   matches the paper's FE model)

Full realistic pipeline: generate_fe (MAPDL) x 6 force cases x {truth,parent}
-> SEMM stage 1 -> mode-shape extraction -> transmissibility expansion.
Noise-free, so the residual is the pure FE-discrepancy + mode-extraction error.
"""
import os, sys
from dataclasses import replace
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Localise the 170 g smear to the top-left 2 cm corner (5x5 nodes ~= 2.14 cm).
os.environ["POINT_MASS_PATCH_NX"] = "5"
os.environ["POINT_MASS_PATCH_NY"] = "5"

from synthetic_validation.config import load_config
from synthetic_validation.harness import run_case

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
OUT     = os.path.join(os.path.dirname(__file__), "..", "figures_data", "validation_fields.npz")

truth0 = load_config(os.path.join(CFG_DIR, "truth.json"))
parent = load_config(os.path.join(CFG_DIR, "parent.json"))

dx = truth0.plate_lx / (truth0.grid_nx - 1)
truth = replace(truth0, point_mass=0.17, point_mass_xy=(2 * dx, 0.14))
print(f"Truth: 0.17 kg at top-left corner, patch centre "
      f"({truth.point_mass_xy[0]*1000:.1f}, {truth.point_mass_xy[1]*1000:.1f}) mm")

# Same acquisition settings as the original figure data:
# fs=2000 Hz, single 10000-sample segment -> 5001 lines, df=0.2 Hz, 0..1000 Hz.
res = run_case(
    truth, parent,
    noise=False, n_modes=2,
    saa_level=1.0, fs=2000.0, n_frames=10000, seed=0,
)

rec   = res["recovered"]     # {comp: (nfreq, nnodes)}
tru   = res["truth"]
freqs = np.asarray(res["freqs"])
coords = truth  # placeholder to avoid confusion; real coords loaded below

# Node coordinates from a truth FE solve (all cases share the mesh)
from synthetic_validation.fe_models import generate_fe
node_coords = generate_fe(truth, "center_middle")["node_coords"]  # cached
x = node_coords[:, 0]
y = node_coords[:, 1]

COMPS = ["SX", "SY", "SXY", "SX+SY"]

# Resonance bin = peak spatial energy of the measured invariant
spatial_energy = np.sum(np.abs(tru["SX+SY"]), axis=1)
fk = int(np.argmax(spatial_energy))
# Critical node = peak invariant at resonance
cp = int(np.argmax(np.abs(tru["SX+SY"][fk, :])))

def rel_rms(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2)))

met_relrms = np.array([rel_rms(rec[c], tru[c]) for c in COMPS])
met_mac    = np.array([res["metrics"]["mac"][c]   for c in COMPS])
met_nrmse  = np.array([res["metrics"]["nrmse"][c] for c in COMPS])

save = dict(
    x=x, y=y, freqs=freqs, fk=fk, cp=cp,
    metrics_nrmse=met_nrmse, metrics_relrms=met_relrms, metrics_mac=met_mac,
    comps=np.array(COMPS),
)
for c in COMPS:
    save[f"rec_{c}"] = rec[c]
    save[f"tru_{c}"] = tru[c]

np.savez(OUT, **save)
print(f"\nSaved {OUT}")
print(f"resonance bin fk={fk} -> {freqs[fk]:.2f} Hz | critical node cp={cp} "
      f"at ({x[cp]*1000:.1f}, {y[cp]*1000:.1f}) mm")
print("rel-RMS (SX, SY, SXY, SX+SY):", np.round(met_relrms * 100, 2), "%")
print("MAC     (SX, SY, SXY, SX+SY):", np.round(met_mac, 4))
print("Done.")
