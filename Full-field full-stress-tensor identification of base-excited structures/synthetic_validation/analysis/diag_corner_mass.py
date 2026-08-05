"""
Diagnostic: does a 170 g mass smeared over a 2x2 cm TOP-LEFT-CORNER patch make a
SECOND mode base-active (breaking the symmetry that the old top-centre mass kept)?

Single truth FE solve, then inspect modal_freqs, gamma_base (base participation),
and whether tau_xy is non-zero at the peak-stress node.
"""
import os, sys
from dataclasses import replace
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Localise the smear to the top-left 2 cm corner (5x5 nodes ~= 2.14 cm on the
# 29x29 / 0.15 m grid, dx = 5.36 mm). The solver reads these from the env.
os.environ["POINT_MASS_PATCH_NX"] = "5"
os.environ["POINT_MASS_PATCH_NY"] = "5"

from synthetic_validation.config import load_config
from synthetic_validation.fe_models import generate_fe

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
truth0 = load_config(os.path.join(CFG_DIR, "truth.json"))

# 170 g steel block, top-left corner. Patch centre at (2*dx, y=0.14) so the 5-node
# window anchors on the left edge (x in [0, 21] mm) and the top (y in [129,150] mm).
dx = truth0.plate_lx / (truth0.grid_nx - 1)
truth = replace(truth0, point_mass=0.17, point_mass_xy=(2 * dx, 0.14))
print(f"dx={dx*1000:.2f} mm | patch centre = ({truth.point_mass_xy[0]*1000:.1f}, "
      f"{truth.point_mass_xy[1]*1000:.1f}) mm | mass={truth.point_mass} kg")

fe = generate_fe(truth, "center_middle")
md = fe["modal_data"]
f  = np.asarray(md["modal_freqs"])
g  = np.abs(np.asarray(md["gamma_base"]))
gn = g / g.max()

print("\nmode |  freq [Hz] | |gamma_base| | normalised | base-active?")
print("-" * 62)
for i, (fi, gi, gni) in enumerate(zip(f, g, gn)):
    tag = "  <== BASE-ACTIVE" if gni > 0.05 else ""
    print(f"{i:4d} | {fi:9.2f} | {gi:11.3e} | {gni:9.3f} |{tag}")

# In-band base-active modes (45-200 Hz, meaningful participation)
inband = [(i, f[i]) for i in range(len(f)) if 45 <= f[i] <= 200 and gn[i] > 0.05]
print(f"\nBase-active modes in 45-200 Hz band: {len(inband)}")
for i, fi in inband:
    print(f"   mode {i}: {fi:.1f} Hz")

# tau_xy activity at the peak sigma_yy node (was ~0 with the centred mass)
sy  = np.abs(md["modal_sy"])   # (nmodes, nnodes)
sxy = np.abs(md["modal_sxy"])
m0 = int(np.argmax(g[:len(f)])) if len(f) else 0
peak_node = int(np.argmax(sy[m0]))
print(f"\nAt the mode-{m0} peak-sigma_yy node {peak_node}: "
      f"|sxy|/|sy| = {sxy[m0, peak_node]/sy[m0, peak_node]:.3f} "
      f"(was ~0 on the old symmetry axis)")
print("Done.")
