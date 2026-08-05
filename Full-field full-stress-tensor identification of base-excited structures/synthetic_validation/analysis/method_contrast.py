import sys, numpy as np
sys.path.insert(0,".")
from ansys.mapdl import reader as pymr
from synthetic_validation.harness import run_case_from_modal, run_direct_psd_from_modal
from synthetic_validation.metrics import rel_rms_error, peak_ratio
r=pymr.read_binary("../plosca_base_excitation_files/dp0/SYS/MECH/file.rst")
f=np.array([float(v) for v in r.time_values]);nm=len(f);nn=r.mesh.nodes.shape[0]
SX=np.zeros((nm,nn));SY=np.zeros((nm,nn));SXY=np.zeros((nm,nn));UZ=np.zeros((nm,nn));d2=np.zeros(nm)
for i in range(nm):
    _,s=r.nodal_stress(i);_,d=r.nodal_displacement(i)
    SX[i]=np.nan_to_num(s[:,0]);SY[i]=np.nan_to_num(s[:,2]);SXY[i]=np.nan_to_num(s[:,5]);UZ[i]=d[:,1];d2[i]=(d**2).sum()
g=np.array([UZ[i].sum()/(d2[i]+1e-30) for i in range(nm)]);nc=r.mesh.nodes
md={"modal_freqs":f,"modal_omega":2*np.pi*f,"zeta":np.full(nm,0.005),
    "node_coords":np.column_stack([nc[:,0],nc[:,2],np.zeros(nn)]),
    "modal_sx":SX,"modal_sy":SY,"modal_sxy":SXY,"gamma_base":g,"modal_mass":d2}
def line(tag,res):
    m=res["metrics"]; print("  %-26s "%tag+"  ".join("%s:rel=%.3f pk=%.2f"%(c,m["rel_rms"][c],m["peak_ratio"][c]) for c in ["SX","SXY"]))
print("METHOD CONTRAST on real model (auto-PSD recovery: rel_rms + peak_ratio)")
for noise,seg in [(0.0,None),(0.02,8),(0.10,8)]:
    print("\n-- camera noise=%.0f%% signal, n_segments=%s --"%(noise*100,seg))
    kw=dict(saa_level=1.0,fs=2000,n_frames=10000,seed=0,n_modes=2,noise_sigma_T=noise*0.027,n_segments=seg)
    line("transmissibility(+accel)", run_case_from_modal(md,md,**kw))
    line("direct-PSD(camera-only)", run_direct_psd_from_modal(md,md,**kw))
print("\nNOTE: direct-PSD gives auto-PSDs only; cross-spectra (3x3 matrix) need transmissibility.")
