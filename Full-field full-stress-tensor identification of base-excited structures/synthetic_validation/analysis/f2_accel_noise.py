import sys, numpy as np
from scipy.signal import welch, csd
sys.path.insert(0, ".")
from ansys.mapdl import reader as pymr
from synthetic_validation.forward_model import synthesize_base_excitation, base_excitation_frf
from synthetic_validation.expansion import normalize_mode_shapes, modal_decompose, expand_components, stress_psd
from synthetic_validation.noise import inject_accel_noise, inject_camera_noise
from synthetic_validation.metrics import nrmse

r=pymr.read_binary("../plosca_base_excitation_files/dp0/SYS/MECH/file.rst")
f=np.array([float(v) for v in r.time_values]);nm=len(f);nn=r.mesh.nodes.shape[0]
SX=np.zeros((nm,nn));SY=np.zeros((nm,nn));SXY=np.zeros((nm,nn));UZ=np.zeros((nm,nn));d2=np.zeros(nm)
for i in range(nm):
    _,s=r.nodal_stress(i);_,d=r.nodal_displacement(i)
    SX[i]=np.nan_to_num(s[:,0]);SY[i]=np.nan_to_num(s[:,2]);SXY[i]=np.nan_to_num(s[:,5]);UZ[i]=d[:,1]
    d2[i]=(d**2).sum()
g=np.array([UZ[i].sum()/(d2[i]+1e-30) for i in range(nm)])
nc=r.mesh.nodes
md={"modal_freqs":f,"modal_omega":2*np.pi*f,"zeta":np.full(nm,0.005),
    "node_coords":np.column_stack([nc[:,0],nc[:,2],np.zeros(nn)]),
    "modal_sx":SX,"modal_sy":SY,"modal_sxy":SXY,"gamma_base":g,"modal_mass":d2}
factor=-3.6e8; fs=2000; N=10000
out=synthesize_base_excitation(md,fs,N,1.0,np.random.default_rng(0),factor,(nn,1))
cam=out["cam_frames"].reshape(N,nn)*factor; accel0=out["accel"]; a_std=accel0.std()
wk=dict(fs=fs,nperseg=int(2*N/9),window="boxcar",detrend=False)  # 8 segments
frq,Saa_clean=welch(accel0,**wk)
psi=normalize_mode_shapes({k:v.T[:, :2] for k,v in
     {"SX":SX,"SY":SY,"SXY":SXY,"SX+SY":SX+SY}.items()})
truthPSD={c:(np.abs(v)**2)*Saa_clean[:,None] for c,v in base_excitation_frf(md,frq).items()}
def recover(accel):
    Saa=welch(accel,**wk)[1]
    T=np.empty((len(frq),nn),complex)
    for i in range(nn): T[:,i]=csd(cam[:,i],accel,**wk)[1]/Saa
    Tc=expand_components(modal_decompose(psi["SX+SY"],T),psi)
    return stress_psd(Tc,Saa)
print("ACCEL-noise sweep (camera clean): NRMSE vs TRUE PSD, 8-seg Welch")
print(" accel_noise(%signal) | SX      SY      SXY     SX+SY")
for frac in [0.0,0.05,0.20,0.50,1.0]:
    rng=np.random.default_rng(7)
    an=inject_accel_noise(accel0,frac*a_std,rng) if frac>0 else accel0
    rec=recover(an)
    print("   %5.0f%%             | "%(frac*100)+"  ".join("%.4f"%nrmse(rec[c],truthPSD[c]) for c in ["SX","SY","SXY","SX+SY"]))
print("\n(compare: CAMERA-noise at 100%% signal gave ~0.005 NRMSE = negligible)")

# --- peak-ratio metric: mean(recovered/truth) over nodes at the resonance bin ---
fk=int(np.argmax(np.abs(base_excitation_frf(md,frq)["SX+SY"]).sum(axis=1)))
print("\nSame sweep, but PEAK-RATIO metric (recovered/truth at f=%.1f Hz, median over nodes):"%frq[fk])
print(" accel_noise(%signal) | SX-ratio  SY-ratio  (1.0=perfect, <1=H1 underestimate)")
for frac in [0.0,0.05,0.20,0.50,1.0,2.0]:
    an=inject_accel_noise(accel0,frac*a_std,np.random.default_rng(7)) if frac>0 else accel0
    rec=recover(an)
    rr=[np.median(rec[c][fk]/(truthPSD[c][fk]+1e-30)) for c in ["SX","SY"]]
    print("   %5.0f%%             |  %.3f     %.3f"%(frac*100, rr[0], rr[1]))
