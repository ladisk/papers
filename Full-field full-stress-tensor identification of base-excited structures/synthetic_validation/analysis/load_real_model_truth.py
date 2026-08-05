import sys, numpy as np
sys.path.insert(0, ".")
from ansys.mapdl import reader as pymr
from synthetic_validation.harness import run_case_from_modal

RST = "../plosca_base_excitation_files/dp0/SYS/MECH/file.rst"
r = pymr.read_binary(RST)
freqs = np.array([float(f) for f in r.time_values])
nodes = r.mesh.nodes
# plate in x-z plane; harness (x,y)=(model x, model z); out-of-plane = model y
xh, yh = nodes[:,0], nodes[:,2]
nnodes = nodes.shape[0]; nmodes = len(freqs)
SX=np.zeros((nmodes,nnodes)); SY=np.zeros((nmodes,nnodes)); SXY=np.zeros((nmodes,nnodes)); UZ=np.zeros((nmodes,nnodes))
for i in range(nmodes):
    _,s = r.nodal_stress(i)         # [SX,SY,SZ,SXY,SYZ,SXZ]
    _,d = r.nodal_displacement(i)   # [ux,uy,uz]
    SX[i]=np.nan_to_num(s[:,0]); SY[i]=np.nan_to_num(s[:,2]); SXY[i]=np.nan_to_num(s[:,5])  # sxx<-SX, syy<-SZ, txy<-SXZ
    UZ[i]=d[:,1]                     # out-of-plane <- model uy
# base participation (uniform plate mass cancels): gamma_r = sum(uz)/sum(|u|^2)
disp2 = np.array([ (r.nodal_displacement(i)[1]**2).sum() for i in range(nmodes)])
gamma = np.array([ UZ[i].sum()/ (disp2[i]+1e-30) for i in range(nmodes)])
gnorm = np.abs(gamma)/np.abs(gamma).max()
print("real-model modes: freq | gamma_norm (base activity)")
for i in range(nmodes):
    print("  %6.1f Hz | %.3f %s"%(freqs[i], gnorm[i], "<-- base-active" if gnorm[i]>0.15 else ""))
md = {"modal_freqs":freqs,"modal_omega":2*np.pi*freqs,"zeta":np.full(nmodes,0.005),
      "node_coords":np.column_stack([xh,yh,np.zeros(nnodes)]),
      "modal_sx":SX,"modal_sy":SY,"modal_sxy":SXY,"gamma_base":gamma,"modal_mass":disp2}
# oracle recovery (truth=prior) on the REAL model, in-band modes (n_modes covering 45-300)
nin = int((freqs<300).sum())
res = run_case_from_modal(md, md, saa_level=1.0, fs=2000, n_frames=10000, seed=0, n_modes=max(1,nin))
print("\nORACLE recovery on REAL base_excitation model (n_modes=%d in <300Hz):"%nin)
for c in ["SX","SY","SXY","SX+SY"]:
    print("  %-6s NRMSE=%.5f MAC=%.4f"%(c,res["metrics"]["nrmse"][c],res["metrics"]["mac"][c]))
