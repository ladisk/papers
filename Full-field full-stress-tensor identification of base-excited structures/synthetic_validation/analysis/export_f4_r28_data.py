"""Export data for the F4 (partial-FoV / resolution robustness, R2.11) and
R2.8 (equivalent-stress PSD) figures, from the real base_excitation FE model.
Reproduces synthetic_validation/figures_data/{f4_partial_fov,equivalent_stress}.npz.
"""
import sys, numpy as np
sys.path.insert(0, ".")
from ansys.mapdl import reader as pymr
from synthetic_validation.forward_model import base_excitation_frf
from synthetic_validation.expansion import normalize_mode_shapes, modal_decompose, expand_components
from synthetic_validation.metrics import rel_rms_error

OUT = "synthetic_validation/figures_data"
RST = "../plosca_base_excitation_files/dp0/SYS/MECH/file.rst"
r = pymr.read_binary(RST)
f = np.array([float(v) for v in r.time_values]); nm = len(f); nn = r.mesh.nodes.shape[0]
SX = np.zeros((nm, nn)); SY = np.zeros((nm, nn)); SXY = np.zeros((nm, nn)); UZ = np.zeros((nm, nn)); d2 = np.zeros(nm)
for i in range(nm):
    _, s = r.nodal_stress(i); _, d = r.nodal_displacement(i)
    SX[i] = np.nan_to_num(s[:, 0]); SY[i] = np.nan_to_num(s[:, 2]); SXY[i] = np.nan_to_num(s[:, 5])
    UZ[i] = d[:, 1]; d2[i] = (d ** 2).sum()
g = np.array([UZ[i].sum() / (d2[i] + 1e-30) for i in range(nm)])
nc = r.mesh.nodes; xh, yh = nc[:, 0], nc[:, 2]
md = {"modal_freqs": f, "modal_omega": 2 * np.pi * f, "zeta": np.full(nm, 0.005),
      "node_coords": np.column_stack([xh, yh, 0 * xh]),
      "modal_sx": SX, "modal_sy": SY, "modal_sxy": SXY, "gamma_base": g, "modal_mass": d2}
freqs = np.linspace(45, 205, 400); Saa = np.ones_like(freqs)
Tt = base_excitation_frf(md, freqs); Tcam = Tt["SX+SY"]
psi = normalize_mode_shapes({k: v.T[:, :2] for k, v in {"SX": SX, "SY": SY, "SXY": SXY, "SX+SY": SX + SY}.items()})
ix = np.round((xh - xh.min()) / (xh.max() - xh.min()) * 30).astype(int)
iy = np.round((yh - yh.min()) / (yh.max() - yh.min()) * 30).astype(int)

# ---- F4: rel_rms vs number of camera points, noise-free vs 25% per-point noise ----
rng = np.random.default_rng(0); noise_rel = 0.25; scale = np.median(np.abs(Tcam[np.abs(Tcam) > 0]))
def recover(S, noisy):
    T = Tcam[:, S].copy()
    if noisy:
        T = T + noise_rel * scale * (rng.standard_normal(T.shape) + 1j * rng.standard_normal(T.shape))
    return expand_components(modal_decompose(psi["SX+SY"][S, :], T), psi)
def rr(Tc):
    return float(np.mean([rel_rms_error(np.abs(Tc[c]) ** 2 * Saa[:, None], np.abs(Tt[c]) ** 2 * Saa[:, None]) for c in ["SX", "SY", "SXY"]]))
npts = []; clean = []; noisy = []
for st in [1, 2, 3, 4, 5, 6, 10, 15]:
    S = np.where((ix % st == 0) & (iy % st == 0))[0]; npts.append(len(S))
    clean.append(rr(recover(S, False))); noisy.append(float(np.mean([rr(recover(S, True)) for _ in range(4)])))
np.savez(f"{OUT}/f4_partial_fov.npz", npoints=np.array(npts),
         relrms_noisefree=np.array(clean), relrms_noisy=np.array(noisy), noise_rel=noise_rel)

# ---- R2.8: von-Mises-equivalent stress PSD (plane stress) from the recovered 3x3 matrix ----
Tc = recover(np.arange(nn), False)
def svm(T):  # S_vM = Saa[|Txx|^2 + |Tyy|^2 - Re(Txx conj Tyy) + 3|Txy|^2]  (uses cross-term)
    return Saa[:, None] * (np.abs(T["SX"]) ** 2 + np.abs(T["SY"]) ** 2
                           - np.real(T["SX"] * np.conj(T["SY"])) + 3 * np.abs(T["SXY"]) ** 2)
Svm_rec, Svm_true = svm(Tc), svm(Tt)
fk = int(np.argmax(Svm_true.sum(axis=1))); cp = int(np.argmax(Svm_true[fk]))
np.savez(f"{OUT}/equivalent_stress.npz", x=xh, y=yh, freq=freqs[fk],
         svm_field_true=Svm_true[fk], svm_field_rec=Svm_rec[fk],
         freqs=freqs, svm_spec_true=Svm_true[:, cp], svm_spec_rec=Svm_rec[:, cp])
print("F4 field recovered vs truth rel_rms=%.4f; R2.8 field rel_rms=%.4f" % (
      rr(Tc), rel_rms_error(Svm_rec[fk], Svm_true[fk])))
