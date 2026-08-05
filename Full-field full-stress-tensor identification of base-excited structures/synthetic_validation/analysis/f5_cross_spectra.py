import sys, numpy as np
sys.path.insert(0, ".")
from ansys.mapdl import reader as pymr
from synthetic_validation.harness import run_case_from_modal
from synthetic_validation.forward_model import base_excitation_frf

# --- load real base_excitation model as truth (x<-x, y<-z, oop<-y; sxx<-SX,syy<-SZ,txy<-SXZ) ---
r = pymr.read_binary("../plosca_base_excitation_files/dp0/SYS/MECH/file.rst")
f = np.array([float(v) for v in r.time_values]); nm=len(f); nn=r.mesh.nodes.shape[0]
SX=np.zeros((nm,nn));SY=np.zeros((nm,nn));SXY=np.zeros((nm,nn));UZ=np.zeros((nm,nn));d2=np.zeros(nm)
for i in range(nm):
    _,s=r.nodal_stress(i); _,d=r.nodal_displacement(i)
    SX[i]=np.nan_to_num(s[:,0]);SY[i]=np.nan_to_num(s[:,2]);SXY[i]=np.nan_to_num(s[:,5]);UZ[i]=d[:,1];d2[i]=(d[:,1]**2).sum()+ (d[:,0]**2).sum()+(d[:,2]**2).sum()
g=np.array([UZ[i].sum()/(d2[i]+1e-30) for i in range(nm)])
nc=r.mesh.nodes
md={"modal_freqs":f,"modal_omega":2*np.pi*f,"zeta":np.full(nm,0.005),
    "node_coords":np.column_stack([nc[:,0],nc[:,2],np.zeros(nn)]),
    "modal_sx":SX,"modal_sy":SY,"modal_sxy":SXY,"gamma_base":g,"modal_mass":d2}

res = run_case_from_modal(md, md, saa_level=1.0, fs=2000, n_frames=10000, seed=0, n_modes=2)
Tr = res["T_by_comp"]; Saa=res["S_aa"]; fr=res["freqs"]
Tt = base_excitation_frf(md, fr)              # truth transmissibilities
# resonance freq + representative node (max invariant response there)
fk = int(np.argmax(np.abs(Tr["SX+SY"]).sum(axis=1)))
node = int(np.argmax(np.abs(Tt["SX+SY"][fk])))
print("representative point: f=%.1f Hz, node=%d"%(fr[fk],node))
comps=["SX","SY","SXY"]
def mat(T):
    v=np.array([T[c][fk,node] for c in comps])
    return np.outer(v, np.conj(v))*Saa[fk]     # 3x3 cross-spectral stress-PSD [Pa^2/Hz]
for lab,T in [("RECOVERED",Tr),("TRUTH",Tt)]:
    M=mat(T)
    print("\n=== %s 3x3 stress-PSD matrix at (f,node) ==="%lab)
    print(" |S_ij| (Pa^2/Hz):"); 
    for i in range(3): print("   ",["%.3e"%abs(M[i,j]) for j in range(3)])
    print(" phase(S_ij) deg:")
    for i in range(3): print("   ",["%+7.1f"%np.degrees(np.angle(M[i,j])) for j in range(3)])
    coh=np.array([[abs(M[i,j])**2/(abs(M[i,i])*abs(M[j,j])+1e-60) for j in range(3)] for i in range(3)])
    print(" coherence gamma^2:"); 
    for i in range(3): print("   ",["%.3f"%coh[i,j] for j in range(3)])
    ev=np.linalg.eigvalsh(M); ev=np.sort(np.abs(ev))[::-1]
    print(" eigenvalues (sorted):",["%.3e"%e for e in ev], " -> rank~%d"%(np.sum(ev>ev[0]*1e-6)))
# recovery accuracy of the off-diagonal (cross) terms over all freqs/nodes
Mr_off=Tr["SX"]*np.conj(Tr["SY"]); Mt_off=Tt["SX"]*np.conj(Tt["SY"])
err=np.linalg.norm(Mr_off-Mt_off)/np.linalg.norm(Mt_off)
print("\ncross-term SX*conj(SY) recovered-vs-truth rel error (all f,nodes): %.2e"%err)
