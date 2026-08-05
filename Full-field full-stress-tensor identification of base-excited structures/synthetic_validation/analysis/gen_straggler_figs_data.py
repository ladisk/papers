"""
Regenerate data for the three straggler figures on the CORNER-MASS truth, with the
REALISTIC SEMM prior (not the oracle), so they are consistent with V1-V7:
  - cross_spectra   (R2.9): 3x3 stress-PSD matrix, coherence, eigenvalues; recovered vs truth
  - equivalent_stress (R2.8): von Mises equiv-stress field + spectrum, recovered vs truth
  - partial_fov     (R2.11): recovery error vs number of camera points (noise-free & noisy)
Analytic (no MAPDL): uses the modal dicts in broad_analysis/.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from synthetic_validation.harness import run_case_from_modal
from synthetic_validation.forward_model import base_excitation_frf
from synthetic_validation.expansion import normalize_mode_shapes, modal_decompose, expand_components
from synthetic_validation.metrics import rel_rms_error

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "broad_analysis")
OUT  = os.path.join(HERE, "..", "figures_data", "straggler_data.npz")
ACQ  = dict(saa_level=1.0, fs=2000.0, n_frames=10000, seed=0)

def load_md(n):
    return {k: v for k, v in np.load(os.path.join(DATA, f"{n}.npz"), allow_pickle=True).items()}
truth = load_md("modal_truth"); prior = load_md("modal_prior_semm")

res = run_case_from_modal(truth, prior, n_modes=2, **ACQ)
Tr = res["T_by_comp"]; Saa = res["S_aa"]; freqs = res["freqs"]
Tt = base_excitation_frf(truth, freqs)          # analytic truth transmissibilities
x = truth["node_coords"][:, 0] * 1000.0; y = truth["node_coords"][:, 1] * 1000.0
nn = len(x); COMPS = ["SX", "SY", "SXY"]

# resonance bins (mode 1 ~51 Hz, mode 2 ~188 Hz) + one off-resonance between them
def nearest(fhz): return int(np.argmin(np.abs(freqs - fhz)))
b1, b2 = nearest(51.4), nearest(187.6); boff = nearest(120.0)

# ---- CROSS-SPECTRA: BAND-INTEGRATED covariance (single-freq S_ij is rank-1 by
# construction -> coherence==1; non-proportionality only appears once integrated) ----
def covar(T):                                        # (nnodes,3,3) complex
    Sig = np.zeros((nn, 3, 3), complex)
    for a in range(3):
        for bb in range(3):
            Sig[:, a, bb] = np.sum(T[COMPS[a]] * np.conj(T[COMPS[bb]]) * Saa[:, None], axis=0)
    return Sig
Sig_t = covar(Tt); Sig_r = covar(Tr)
# field-wide per-cell recovery error of the band-integrated |Sig_ij| (rel-RMS over nodes)
xspec_field_err = np.array([[np.linalg.norm(np.abs(Sig_r[:, a, b]) - np.abs(Sig_t[:, a, b]))
                             / np.linalg.norm(np.abs(Sig_t[:, a, b]))
                             for b in range(3)] for a in range(3)])
# per-node eigen-decomposition of the truth covariance -> non-proportionality lam2/lam1
lam = np.sort(np.abs(np.linalg.eigvalsh(Sig_t)), axis=1)[:, ::-1]   # (nnodes,3) desc
np_field = lam[:, 1] / (lam[:, 0] + 1e-300)
energy = np.real(np.trace(Sig_t, axis1=1, axis2=2))
# representative node: most BALANCED (multiaxial) among the high-energy nodes
mind = np.min(np.real(np.einsum('nii->ni', Sig_t)), axis=1)         # min diagonal per node
cand = np.where(energy > np.percentile(energy, 60))[0]
node = int(cand[np.argmax(mind[cand])])
Sig_tru = Sig_t[node]; Sig_rec = Sig_r[node]
def coherence(M):
    d = np.real(np.diag(M))
    return np.array([[abs(M[i, j])**2 / (d[i]*d[j] + 1e-300) for j in range(3)] for i in range(3)])
coh_tru = coherence(Sig_tru); coh_rec = coherence(Sig_rec)
# energy-weighted non-proportionality of the shear pair over the field
wsx_sxy = np.array([1 - coherence(Sig_t[m])[0, 2] for m in range(nn)])
nonprop_ew = float(np.sum(energy * wsx_sxy) / np.sum(energy))
# cross-term recovery error (all freqs/nodes, Frobenius)
xterm_err = float(np.linalg.norm(Tr["SX"]*np.conj(Tr["SY"]) - Tt["SX"]*np.conj(Tt["SY"]))
                  / np.linalg.norm(Tt["SX"]*np.conj(Tt["SY"])))
print(f"cross-spectra node {node} at ({x[node]:.0f},{y[node]:.0f}) mm; energy-wtd non-prop={nonprop_ew:.3f}")
print("cross-term SX*conj(SY) rel err = %.3f" % xterm_err)
print("coherence truth  (SX-SY, SX-SXY, SY-SXY) = %.3f %.3f %.3f" % (coh_tru[0,1], coh_tru[0,2], coh_tru[1,2]))
print("coherence recov  (SX-SY, SX-SXY, SY-SXY) = %.3f %.3f %.3f" % (coh_rec[0,1], coh_rec[0,2], coh_rec[1,2]))

# ---- EQUIVALENT STRESS (von Mises PSD) ----
def svm(T):
    return Saa[:, None] * (np.abs(T["SX"])**2 + np.abs(T["SY"])**2
                           - np.real(T["SX"]*np.conj(T["SY"])) + 3*np.abs(T["SXY"])**2)
vm_rec = svm(Tr); vm_tru = svm(Tt)
fk_vm = int(np.argmax(vm_tru.sum(axis=1))); cp_vm = int(np.argmax(vm_tru[fk_vm]))
vm_field_err = rel_rms_error(vm_rec[fk_vm], vm_tru[fk_vm])   # PSD field at resonance
print("vM field rel_rms at resonance = %.3f ; critical node (%.0f,%.0f) mm"
      % (vm_field_err, x[cp_vm], y[cp_vm]))
# amplitude-domain vM (band-RMS field) -- the domain the letters' metric tables use
_band = (freqs >= 45) & (freqs <= 250); _df = float(freqs[1] - freqs[0])
_brms = lambda P: np.sqrt(np.sum(np.abs(P)[_band], axis=0) * _df)
vm_amp_tru, vm_amp_rec = _brms(vm_tru), _brms(vm_rec)
vm_amp_err = rel_rms_error(vm_amp_rec, vm_amp_tru)
_hot = np.argsort(vm_amp_tru)[int(0.9 * nn):]                # top-10% most-stressed nodes
vm_amp_top10 = float(np.median(np.abs(vm_amp_rec[_hot] - vm_amp_tru[_hot]) / vm_amp_tru[_hot]))
print("vM amplitude (band-RMS): field %.3f  top-10%% %.3f" % (vm_amp_err, vm_amp_top10))

# cross-spectral structure and per-cell recovery error at the top-10% fatigue hotspots
Sig_hot = np.abs(Sig_t[_hot]).mean(axis=0)                   # mean |Sig_ij| over hotspots [Pa^2/Hz]
xspec_hot_err = np.array([[np.linalg.norm(np.abs(Sig_r[_hot, a, b]) - np.abs(Sig_t[_hot, a, b]))
                           / np.linalg.norm(np.abs(Sig_t[_hot, a, b]))
                           for b in range(3)] for a in range(3)])
print("hotspot |Sig| [MPa^2/Hz]:", np.round(Sig_hot / 1e12, 1).tolist())
print("hotspot per-cell err [%]:", np.round(xspec_hot_err * 100, 1).tolist())

# ---- PARTIAL FoV: recovery vs number of camera points (SEMM prior) ----
psi = normalize_mode_shapes({
    "SX":    prior["modal_sx"].T[:, :2],  "SY":  prior["modal_sy"].T[:, :2],
    "SXY":   prior["modal_sxy"].T[:, :2],
    "SX+SY": (prior["modal_sx"] + prior["modal_sy"]).T[:, :2],
})
Tcam = Tt["SX+SY"]                                   # truth invariant (what the camera sees)
rng = np.random.default_rng(0); noise_rel = 0.25
gx = np.round((x - x.min()) / (x.max() - x.min()) * 28).astype(int)
gy = np.round((y - y.min()) / (y.max() - y.min()) * 28).astype(int)

def recover(S, noisy):
    T = Tcam[:, S].copy()
    if noisy:
        # per-point noise at 25% of the LOCAL magnitude, so fewer points -> weaker
        # spatial averaging -> larger propagated error (the effect R2.11 asks about)
        T = T + noise_rel * np.abs(T) * (rng.standard_normal(T.shape) + 1j*rng.standard_normal(T.shape))
    return expand_components(modal_decompose(psi["SX+SY"][S, :], T), psi)
band = (freqs >= 45) & (freqs <= 250); df = float(freqs[1] - freqs[0])
def brms(P):                                    # PSD -> band-RMS stress field (Pa), per node
    return np.sqrt(np.sum(np.abs(P)[band], axis=0) * df)
def rr(Tc):
    # AMPLITUDE domain (band-RMS stress field), matching the letters' metric tables.
    # The method operates on transmissibilities; the PSD is quadratic and would
    # roughly double every relative error reported here.
    return float(np.mean([rel_rms_error(brms(np.abs(Tc[c])**2 * Saa[:, None]),
                                        brms(np.abs(Tt[c])**2 * Saa[:, None])) for c in COMPS]))
npts, clean, noisy = [], [], []
for st in [1, 2, 3, 4, 5, 6, 10, 15]:
    S = np.where((gx % st == 0) & (gy % st == 0))[0]
    npts.append(len(S)); clean.append(rr(recover(S, False)))
    noisy.append(float(np.mean([rr(recover(S, True)) for _ in range(4)])))
print("partial-FoV npoints:", npts)
print("  noise-free rel_rms %:", [round(c*100, 1) for c in clean])
print("  noisy(25%) rel_rms %:", [round(c*100, 1) for c in noisy])

# ---- save ----
np.savez(OUT,
         x=x, y=y, freqs=freqs, comps=np.array(COMPS),
         # cross-spectra (band-integrated covariance)
         cs_node=node, cs_fk1=b1, cs_fk2=b2, Sig_tru=Sig_tru, Sig_rec=Sig_rec,
         coh_tru=coh_tru, coh_rec=coh_rec, np_field=np_field,
         nonprop_ew=nonprop_ew, xterm_err=xterm_err, xspec_field_err=xspec_field_err,
         Sig_hot=Sig_hot, xspec_hot_err=xspec_hot_err,
         # equivalent stress
         vm_fk=fk_vm, vm_cp=cp_vm, vm_field_err=vm_field_err,
         vm_amp_err=vm_amp_err, vm_amp_top10=vm_amp_top10,
         vm_field_tru=vm_tru[fk_vm], vm_field_rec=vm_rec[fk_vm],
         vm_spec_tru=vm_tru[:, cp_vm], vm_spec_rec=vm_rec[:, cp_vm],
         # partial FoV
         npoints=np.array(npts), relrms_noisefree=np.array(clean),
         relrms_noisy=np.array(noisy), noise_rel=noise_rel)
print("\nSaved", OUT)
