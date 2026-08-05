import sys, os, numpy as np
sys.path.insert(0,".")
from ansys.mapdl import reader as pymr
from synthetic_validation.harness import run_case_from_modal, run_direct_psd_from_modal
from synthetic_validation.forward_model import base_excitation_frf
from synthetic_validation.metrics import rel_rms_error, peak_ratio
OUT="synthetic_validation/figures_data"; os.makedirs(OUT,exist_ok=True)

# ---- real-model truth modal data (x<-x, y<-z, oop<-y; sxx<-SX,syy<-SZ,txy<-SXZ) ----
r=pymr.read_binary("../plosca_base_excitation_files/dp0/SYS/MECH/file.rst")
f=np.array([float(v) for v in r.time_values]);nm=len(f);nn=r.mesh.nodes.shape[0]
SX=np.zeros((nm,nn));SY=np.zeros((nm,nn));SXY=np.zeros((nm,nn));UZ=np.zeros((nm,nn));d2=np.zeros(nm)
for i in range(nm):
    _,s=r.nodal_stress(i);_,d=r.nodal_displacement(i)
    SX[i]=np.nan_to_num(s[:,0]);SY[i]=np.nan_to_num(s[:,2]);SXY[i]=np.nan_to_num(s[:,5]);UZ[i]=d[:,1];d2[i]=(d**2).sum()
g=np.array([UZ[i].sum()/(d2[i]+1e-30) for i in range(nm)]);nc=r.mesh.nodes
xh,yh=nc[:,0],nc[:,2]
md={"modal_freqs":f,"modal_omega":2*np.pi*f,"zeta":np.full(nm,0.005),
    "node_coords":np.column_stack([xh,yh,np.zeros(nn)]),
    "modal_sx":SX,"modal_sy":SY,"modal_sxy":SXY,"gamma_base":g,"modal_mass":d2}

# ===== (1) Study C: FE-error sensitivity (measured, full-SEMM run; proportional vs NRMSE) =====
# real measured values from the full-path Study C re-summary (docs: synthetic_validation_results.md R2.5)
np.savez(f"{OUT}/study_c_sensitivity.npz",
    labels=np.array(["nominal\n(E,mass)","nu=0.25","nu=0.40","thick+15%"]),
    nrmse_SX=np.array([0.0001,0.0019,0.0018,0.0002]),
    relrms_SX=np.array([0.030,0.385,0.361,0.031]),
    relrms_SXY=np.array([0.089,0.427,0.289,0.105]),
    peakratio_SX=np.array([1.026,0.597,1.474,1.020]))

# ===== (2) method contrast vs camera noise (live, analytic) =====
noises=np.array([0.0,0.02,0.05,0.10]); comps=["SX","SXY"]
trans=np.zeros((len(noises),2)); direct=np.zeros((len(noises),2))
for k,nz in enumerate(noises):
    kw=dict(saa_level=1.0,fs=2000,n_frames=10000,seed=0,n_modes=2,noise_sigma_T=nz*0.027,n_segments=8)
    rt=run_case_from_modal(md,md,**kw); rd=run_direct_psd_from_modal(md,md,**kw)
    for j,c in enumerate(comps):
        trans[k,j]=rt["metrics"]["rel_rms"][c]; direct[k,j]=rd["metrics"]["rel_rms"][c]
np.savez(f"{OUT}/method_contrast.npz",noise_pct=noises*100,comps=np.array(comps),
         trans_relrms=trans,direct_relrms=direct)

# ===== (3) F5 cross-spectral 3x3 matrix + coherence + eigenvalues (recovered vs truth) =====
res=run_case_from_modal(md,md,saa_level=1.0,fs=2000,n_frames=10000,seed=0,n_modes=2)
Tr=res["T_by_comp"];Saa=res["S_aa"];fr=res["freqs"];Tt=base_excitation_frf(md,fr)
fk=int(np.argmax(np.abs(Tt["SX+SY"]).sum(axis=1)))
# node with significant shear so all 3 comps are nonzero
node=int(np.argmax(np.abs(Tt["SXY"][fk])))
def mat(T):
    v=np.array([T[cc][fk,node] for cc in ["SX","SY","SXY"]]); return np.outer(v,np.conj(v))*Saa[fk]
Mr,Mt=mat(Tr),mat(Tt)
def coh(M): return np.array([[abs(M[i,j])**2/(abs(M[i,i])*abs(M[j,j])+1e-60) for j in range(3)] for i in range(3)])
np.savez(f"{OUT}/cross_spectra.npz",comp_names=np.array(["sigxx","sigyy","tauxy"]),
    S_recovered=Mr,S_truth=Mt,coherence_recovered=coh(Mr),coherence_truth=coh(Mt),
    eig_recovered=np.sort(np.abs(np.linalg.eigvalsh(Mr)))[::-1],
    eig_truth=np.sort(np.abs(np.linalg.eigvalsh(Mt)))[::-1],
    freq=fr[fk],node=node)

# ===== (4) spatial full-field stress-PSD maps at resonance (truth), 31x31 grid =====
truthPSD={c:(np.abs(v)**2)*Saa[:,None] for c,v in Tt.items()}
np.savez(f"{OUT}/spatial_fields.npz", x=xh, y=yh, freq=fr[fk],
    sigxx=truthPSD["SX"][fk], sigyy=truthPSD["SY"][fk], tauxy=truthPSD["SXY"][fk],
    invariant=truthPSD["SX+SY"][fk])
print("saved figure data to",OUT); print(os.listdir(OUT))
print("cross-spectra resonance f=%.1f node=%d; eig(recovered)=%s"%(fr[fk],node,np.sort(np.abs(np.linalg.eigvalsh(Mr)))[::-1]))
