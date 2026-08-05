import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

d = np.load(os.path.join(os.path.dirname(__file__), "..", "figures_data", "validation_fields.npz"),
            allow_pickle=True)
freqs = d["freqs"]; fk = int(d["fk"]); x = d["x"]*1000; y = d["y"]*1000
COMPS = ["SX", "SY", "SXY", "SX+SY"]

def relrms(a, b, fmask=None, nmask=None):
    a = np.abs(d["rec_"+a]); b = np.abs(d["tru_"+b])
    if fmask is not None: a, b = a[fmask], b[fmask]
    if nmask is not None: a, b = a[:, nmask], b[:, nmask]
    return np.sqrt(np.sum((a-b)**2)/np.sum(b**2))*100

print("resonance fk=%d -> %.1f Hz | critical node cp=%d at (%.1f,%.1f) mm"
      % (fk, freqs[fk], int(d["cp"]), x[int(d["cp"])], y[int(d["cp"])]))

# where are the truth resonances (spatial energy of invariant)?
se = np.sum(np.abs(d["tru_SX+SY"]), axis=1)
peaks = np.argsort(se)[::-1][:6]
print("Top truth-energy freq bins:", sorted([round(float(freqs[p]),1) for p in peaks]))

full = slice(None)
band = (freqs >= 45) & (freqs <= 250)     # the paper's analysis band
print("\nrel-RMS [%]   full 0-1000Hz   |   band 45-250Hz")
for c in COMPS:
    print(f"  {c:6s}   {relrms(c,c):7.1f}        |   {relrms(c,c,fmask=band):7.1f}")

# stress-conditioned (band-limited): error at high-stress nodes
df = freqs[1]-freqs[0]
def band_rms_node(tag, kind):
    P = np.abs(d[f"{kind}_{tag}"])[band]
    return np.sqrt(np.sum(P, axis=0)*df)
print("\nBand-limited per-node error vs stress level:")
for c in ["SX","SY","SXY","SX+SY"]:
    tru = band_rms_node(c,"tru"); rec = band_rms_node(c,"rec")
    err = np.abs(rec-tru)/tru*100
    order = np.argsort(tru); N=len(tru)
    top10 = order[int(0.9*N):]; bot50 = order[:int(N//2)]
    # damage weighted b=5
    w = tru**5; dw = np.sum(w*np.abs(rec-tru)/tru)/np.sum(w)*100
    print(f"  {c:6s}: top10%%={np.median(err[top10]):5.1f}%  bottom50%%={np.median(err[bot50]):6.1f}%  damage-wtd(b5)={dw:4.1f}%")

# tau_xy activity at the critical node now (was ~0 on the old symmetry axis)
cp = int(d["cp"])
print(f"\nAt critical node cp={cp}: peak tru |SXY|={np.abs(d['tru_SXY'][:,cp]).max():.2e}, "
      f"|SY|={np.abs(d['tru_SY'][:,cp]).max():.2e}, ratio={np.abs(d['tru_SXY'][:,cp]).max()/np.abs(d['tru_SY'][:,cp]).max():.3f}")
