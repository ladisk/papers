"""Probe: what does a compliant clamp do to the PRIOR's modal frequencies?

One FE solve per stiffness, to choose meaningful k_rot values before paying for the
full 6-impact + SEMM sweep. A rigid clamp is the baseline; k_rot is the torsional
stiffness per base node [N*m/rad] restraining the bending rotation about the clamp line.

For scale: the plate's own bending stiffness is D*b/L ~ 160 N*m/rad in total, spread
over 29 base nodes -> ~5.5 N*m/rad per node. So k_rot >> 5.5 is a stiff clamp.
"""
import os, sys
from dataclasses import replace
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["POINT_MASS_PATCH_NX"] = "5"
os.environ["POINT_MASS_PATCH_NY"] = "5"

from synthetic_validation.config import load_config
from synthetic_validation.fe_models import generate_fe

CFG = os.path.join(os.path.dirname(__file__), "..", "configs")
p0 = load_config(os.path.join(CFG, "parent.json"))
base = replace(p0, E=63e9, nu=0.33, rho=2700, thickness=0.003, point_mass=0.0)
label = list(base.force_points.keys())[0]

print(f"{'clamp':>22s}  {'f1 [Hz]':>9s} {'f2 [Hz]':>9s}   {'shift in f1':>12s}")
ref = None
for tag, bc in [("rigid (nominal)", "clamped"),
                ("k=1e4 /node", "spring:1e4"),
                ("k=1e3 /node", "spring:1e3"),
                ("k=1e2 /node", "spring:1e2"),
                ("k=1e1 /node", "spring:1e1")]:
    cfg = replace(base, base_bc=bc)
    fe = generate_fe(cfg, label)
    f = np.asarray(fe["modal_data"]["modal_freqs"]).ravel()
    if ref is None:
        ref = f[0]
    shift = (f[0] - ref) / ref * 100.0
    flag = "  <-- RIGID-BODY MODE, invalid" if f[0] < 1.0 else ""
    print(f"{tag:>22s}  {f[0]:9.1f} {f[1]:9.1f}   {shift:+11.1f}%{flag}")
