import sys, dataclasses, time
import numpy as np
sys.path.insert(0,".")
from synthetic_validation.config import load_config
from synthetic_validation.studies import study_a_recovery, study_c_fe_discrepancy
truth=load_config("synthetic_validation/configs/truth.json")
parent=load_config("synthetic_validation/configs/parent.json")
COMPS=["SX","SY","SXY","SX+SY"]; BASE=dict(saa_level=1.0,fs=2000,n_frames=10000,seed=0)
t0=time.time()
def show(tag,m):
    print(" %s"%tag)
    for c in COMPS:
        print("    %-6s NRMSE=%.4f  rel_rms=%.4f  peak_ratio=%.3f  MAC=%.4f"%(
              c,m["nrmse"][c],m["rel_rms"][c],m["peak_ratio"][c],m["mac"][c]))
print("=== STUDY A re-summary (full path, noise-free, single-segment) ===")
A=study_a_recovery(truth,parent,analytic=False,n_modes=2,noise=False,**BASE)
show("realistic:",A["realistic"]["metrics"]); show("oracle:",A["oracle"]["metrics"])
print("\n=== STUDY C re-summary (ratio-changing variants) ===")
variants=[parent, dataclasses.replace(parent,nu=0.25), dataclasses.replace(parent,nu=0.40),
          dataclasses.replace(parent,thickness=0.00345)]
names=["nominal","nu=0.25","nu=0.40","thick+15%"]
C=study_c_fe_discrepancy(truth,variants,analytic=False,nominal_idx=0,n_modes=2,noise=False,**BASE)
for nm,vr in zip(names,C["variants"]):
    m=vr["metrics"]
    print("  %-10s NRMSE[SX]=%.4f rel_rms[SX]=%.4f peak_ratio[SX]=%.3f | rel_rms[SXY]=%.4f"%(
          nm,m["nrmse"]["SX"],m["rel_rms"]["SX"],m["peak_ratio"]["SX"],m["rel_rms"]["SXY"]))
print("(%.1f min)"%((time.time()-t0)/60))
