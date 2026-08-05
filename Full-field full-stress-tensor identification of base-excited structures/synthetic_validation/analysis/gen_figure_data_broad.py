"""
Compute ALL data for the 6 broad-analysis figures into one npz + json, so the
plot scripts are thin and consistent. Analytic (no MAPDL): reuses the modal dicts
in broad_analysis/.  band = 45-250 Hz (paper's analysis band).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from synthetic_validation.harness import run_case_from_modal, run_direct_psd_from_modal

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "broad_analysis")
OUTN = os.path.join(HERE, "..", "figures_data", "fig_data_broad.npz")
OUTJ = os.path.join(HERE, "..", "figures_data", "fig_data_broad.json")
ACQ = dict(saa_level=1.0, fs=2000.0, n_frames=10000, seed=0)
COMPS = ["SX", "SY", "SXY", "SX+SY"]

def load_md(name):
    return {k: v for k, v in np.load(os.path.join(DATA, f"{name}.npz"), allow_pickle=True).items()}

truth   = load_md("modal_truth")
prior   = load_md("modal_prior_semm")
p_paper = load_md("modal_parent_paper")
p_E69   = load_md("modal_parent_E69")
p_sym   = load_md("modal_parent_symmass")

def relrms(a, b):
    a = np.abs(a); b = np.abs(b)
    return float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))

def macv(a, b):
    a = np.abs(a).ravel(); b = np.abs(b).ravel()
    return float(np.abs(a @ b) ** 2 / ((a @ a) * (b @ b)))

print("== realistic + oracle (n_modes=2), and full-mode oracle for truth vM ==")
res_r = run_case_from_modal(truth, prior, n_modes=2, **ACQ)      # recovered (method)
res_oT = run_case_from_modal(truth, truth, n_modes=None, **ACQ)  # exact truth (all modes)

freqs = res_r["freqs"]; df = float(freqs[1] - freqs[0])
band = (freqs >= 45) & (freqs <= 250)
node = np.load(os.path.join(DATA, "modal_truth.npz"))
x = truth["node_coords"][:, 0] * 1000.0
y = truth["node_coords"][:, 1] * 1000.0
nn = len(x)

se = np.sum(np.abs(res_r["truth"]["SX+SY"]), axis=1); fk = int(np.argmax(se))
cp = int(np.argmax(np.abs(res_r["truth"]["SX+SY"][fk, :])))

# ---- von Mises PSD (plane stress) from complex transmissibilities ----
def vm_psd(T, Saa):
    Txx, Tyy, Txy = T["SX"], T["SY"], T["SXY"]
    return (np.abs(Txx) ** 2 + np.abs(Tyy) ** 2 - np.real(Txx * np.conj(Tyy)) + 3 * np.abs(Txy) ** 2) * Saa[:, None]
vm_rec = vm_psd(res_r["T_by_comp"], res_r["S_aa"])       # (nfreq, nnodes)
vm_tru = vm_psd(res_oT["T_by_comp"], res_oT["S_aa"])

# ---- band-RMS per node for every field (truth & recovered) ----
def brms(P):  # (nfreq,nnodes) -> (nnodes,)
    return np.sqrt(np.sum(np.abs(P)[band], axis=0) * df)
fields = {}
for c in COMPS:
    fields[f"tru_{c}"] = brms(res_r["truth"][c])
    fields[f"rec_{c}"] = brms(res_r["recovered"][c])
fields["tru_VM"] = brms(vm_tru)
fields["rec_VM"] = brms(vm_rec)

# ---- resonance-bin PSD (for the SXY scatter) ----
res_fields = {f"tru_{c}_res": np.abs(res_r["truth"][c][fk]) for c in COMPS}
res_fields.update({f"rec_{c}_res": np.abs(res_r["recovered"][c][fk]) for c in COMPS})

# ---- per-component through-origin slope k, bias, scatter (band-RMS) ----
def k_split(tru, rec):
    k = float(np.sum(tru * rec) / np.sum(tru ** 2))
    rr = float(np.sqrt(np.sum((rec - tru) ** 2) / np.sum(tru ** 2)))
    bias = abs(k - 1.0)
    scatter = float(np.sqrt(max(rr ** 2 - bias ** 2, 0.0)))
    return k, bias, scatter, rr
kstats = {c: k_split(fields[f"tru_{c}"], fields[f"rec_{c}"]) for c in COMPS}
kstats["VM"] = k_split(fields["tru_VM"], fields["rec_VM"])
# also resonance-bin k for SXY (for the scatter fig annotation)
k_sxy_res = float(np.sum(res_fields["tru_SXY_res"] * res_fields["rec_SXY_res"]) / np.sum(res_fields["tru_SXY_res"] ** 2))

# ---- MAC of the band-RMS STRESS field (amplitude domain) ----
# NOTE: this must be computed on the SAME field the maps plot (band-RMS stress,
# linear in stress), not on the PSD -- squaring into the PSD depresses the MAC and
# would mislabel the figure.
mac = {c: macv(fields[f"rec_{c}"], fields[f"tru_{c}"]) for c in COMPS}
mac["VM"] = macv(fields["rec_VM"], fields["tru_VM"])
peak_ratio = {c: float(res_r["metrics"]["peak_ratio"][c]) for c in COMPS}

# ---- FE-discrepancy decomposition: 5 priors x 4 comps ----
priors = {"oracle": truth, "method": prior, "parent (E63,no mass)": p_paper,
          "parent (E69,no mass)": p_E69, "parent + sym mass": p_sym}
decomp = {}
for name, pm in priors.items():
    r = run_case_from_modal(truth, pm, n_modes=2, **ACQ)
    decomp[name] = {c: dict(rel_rms=relrms(r["recovered"][c], r["truth"][c]),
                            mac=float(r["metrics"]["mac"][c]),
                            peak_ratio=float(r["metrics"]["peak_ratio"][c])) for c in COMPS}
    print(f"  {name:24s} SXY rel_rms={decomp[name]['SXY']['rel_rms']*100:5.1f}%  MAC={decomp[name]['SXY']['mac']:.3f}")

# ---- noise sweep: transmissibility vs direct-PSD ----
# Reported on the band-RMS stress field (AMPLITUDE), the same domain as every other
# accuracy metric in the letters, so the zero-noise floor coincides with Table 1
# (tau_xy 11.3%) instead of its PSD-squared counterpart (21.5%).
sig = [0.0, 0.005, 0.01, 0.05, 0.1]
noise = {"sigma": sig}
for tag, fn in [("trans", run_case_from_modal), ("direct", run_direct_psd_from_modal)]:
    arr = np.zeros((len(sig), len(COMPS)))
    for i, s in enumerate(sig):
        r = fn(truth, prior, n_modes=2, noise_sigma_T=s, **ACQ)
        arr[i] = [relrms(brms(r["recovered"][c]), brms(r["truth"][c])) for c in COMPS]
    noise[tag] = arr
    print(f"  noise {tag}: SXY {arr[0,2]*100:.1f}% -> {arr[-1,2]*100:.1f}%")

# ---- quantity collapse + damage-weighted ----
def dmg_w(tru, rec, b):
    w = tru ** b
    return float(np.sum(w * np.abs(rec - tru) / tru) / np.sum(w))
def top_err(tru, rec, frac=0.10):
    o = np.argsort(tru); k = int((1 - frac) * len(tru))
    idx = o[k:]
    return float(np.median(np.abs(rec[idx] - tru[idx]) / tru[idx]))
collapse = dict(
    sxy_raw=relrms(res_r["recovered"]["SXY"], res_r["truth"]["SXY"]),
    sxy_bandmap=relrms(fields["rec_SXY"], fields["tru_SXY"]),
    vm_field=relrms(fields["rec_VM"], fields["tru_VM"]),
    vm_top10=top_err(fields["tru_VM"], fields["rec_VM"], 0.10),
    vm_dmg_b5=dmg_w(fields["tru_VM"], fields["rec_VM"], 5),
    sxy_dmg_b5=dmg_w(fields["tru_SXY"], fields["rec_SXY"], 5),
    sxy_top10=top_err(fields["tru_SXY"], fields["rec_SXY"], 0.10),
)
# damage-weighted per component across Basquin b
dmg_table = {c: {b: dmg_w(fields[f"tru_{c}"], fields[f"rec_{c}"], b) for b in [3, 5, 8, 10]} for c in COMPS}
dmg_table["VM"] = {b: dmg_w(fields["tru_VM"], fields["rec_VM"], b) for b in [3, 5, 8, 10]}

# ---- spatial: per-node SXY relative error (band-RMS), mass box, top10 invariant nodes ----
sxy_relerr = np.abs(fields["rec_SXY"] - fields["tru_SXY"]) / (fields["tru_SXY"] + 1e-30)
inv_brms = fields["tru_SX+SY"]
top10_inv = np.argsort(inv_brms)[::-1][:max(1, nn // 10)]
mass_box = [0.0, 21.4, 128.6, 150.0]  # x0,x1,y0,y1 mm (top-left 2x2cm patch, 5x5 nodes)

# ---- save ----
save = dict(x=x, y=y, fk=fk, cp=cp, freqs=freqs, band=band,
            k_sxy_res=k_sxy_res, top10_inv=top10_inv, mass_box=np.array(mass_box),
            sxy_relerr=sxy_relerr, sig=np.array(sig),
            noise_trans=noise["trans"], noise_direct=noise["direct"],
            comps=np.array(COMPS))
save.update(fields)
save.update(res_fields)
np.savez(OUTN, **save)

scal = dict(
    resonance_hz=float(freqs[fk]), critical_node=cp,
    mac={**mac}, peak_ratio=peak_ratio,
    kstats={c: dict(k=kstats[c][0], bias=kstats[c][1], scatter=kstats[c][2], rel_rms=kstats[c][3]) for c in kstats},
    k_sxy_res=k_sxy_res,
    decomp=decomp, collapse=collapse, dmg_table=dmg_table,
    noise_sigma=sig,
    noise_trans={c: [float(noise["trans"][i][j]) for i in range(len(sig))] for j, c in enumerate(COMPS)},
    noise_direct={c: [float(noise["direct"][i][j]) for i in range(len(sig))] for j, c in enumerate(COMPS)},
)
with open(OUTJ, "w") as f:
    json.dump(scal, f, indent=2)

print("\nSaved", OUTN)
print("MAC:", {k: round(v, 3) for k, v in mac.items()})
print("k (band-RMS):", {c: round(kstats[c][0], 3) for c in kstats})
print("collapse:", {k: round(v * 100, 1) for k, v in collapse.items()})
print("Done.")
