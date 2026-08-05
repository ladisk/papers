import sys, time, json, dataclasses
import numpy as np
sys.path.insert(0, ".")
from synthetic_validation.config import load_config
from synthetic_validation.fe_models import generate_fe
from synthetic_validation.forward_model import synthesize_base_excitation
from synthetic_validation.harness import run_case
from synthetic_validation import expansion as EXP

TMP = sys.argv[1]
truth = load_config("synthetic_validation/configs/truth.json")
parent = load_config("synthetic_validation/configs/parent.json")
COMPS = ["SX","SY","SXY","SX+SY"]
BASE = dict(saa_level=1.0, fs=2000, n_frames=10000, seed=0)
log = []
def P(*a):
    s=" ".join(str(x) for x in a); print(s, flush=True); log.append(s)

def metrics_line(tag, res):
    m=res["metrics"]; P("  %-22s cond=%.3g" % (tag, res["condition_number"]))
    for c in COMPS: P("     %-6s NRMSE=%.4f MAC=%.4f"%(c,m["nrmse"][c],m["mac"][c]))

# ---- measure synthetic camera signal RMS (for signal-relative SNR) ----
tmd = generate_fe(truth, "center_middle")["modal_data"]
syn = synthesize_base_excitation(tmd, fs=2000, n_frames=10000, saa_level=1.0,
        rng=np.random.default_rng(0), camera_to_stress_factor=-3.6e8,
        grid_shape=(29,29))
sig_std = float(np.std(syn["cam_frames"]))
P("Synthetic camera signal std = %.4g K\n"%sig_std)

results={"signal_std":sig_std}

# ================= Study D: modal convergence =================
P("="*60); P("STUDY D - modal convergence (noise-free)")
t0=time.time(); D={}
for nm in [1,2,3]:
    r=run_case(truth,parent,noise=False,n_modes=nm,**BASE)
    D[nm]={c:float(r["metrics"]["nrmse"][c]) for c in COMPS}
    P("  n_modes=%d : "%nm+" ".join("%s=%.4f"%(c,D[nm][c]) for c in COMPS))
results["D"]=D; P("  (%.1f min)\n"%((time.time()-t0)/60))

# ================= Study E: conditioning =================
P("="*60); P("STUDY E - conditioning (realistic mode basis, n_modes=2)")
r2=run_case(truth,parent,noise=False,n_modes=2,**BASE)
results["E"]={"condition_number_realistic":float(r2["condition_number"])}
P("  condition number (realistic) = %.4g"%r2["condition_number"])
P("  (oracle basis cond = 1.0 by construction)\n")

# ================= Study B: noise Monte-Carlo (signal-relative) =================
P("="*60); P("STUDY B - noise Monte-Carlo (signal-relative SNR)")
t0=time.time(); B={}
for frac in [0.02, 0.10, 0.30]:
    sigma=frac*sig_std; reps=[]
    for rep in range(4):
        r=run_case(truth,parent,noise=True,noise_sigma_T=sigma,n_modes=2,
                   n_segments=8, saa_level=1.0, fs=2000, n_frames=10000, seed=rep)
        reps.append({c:float(r["metrics"]["nrmse"][c]) for c in COMPS})
    mean={c:float(np.mean([x[c] for x in reps])) for c in COMPS}
    std ={c:float(np.std ([x[c] for x in reps])) for c in COMPS}
    B["%.2f"%frac]={"sigma_K":sigma,"mean":mean,"std":std}
    P("  noise=%.0f%% signal : "%(frac*100)+" ".join("%s=%.3f±%.3f"%(c,mean[c],std[c]) for c in COMPS))
results["B"]=B; P("  (%.1f min)\n"%((time.time()-t0)/60))

# ================= Study C: ratio-changing FE discrepancy =================
P("="*60); P("STUDY C - FE discrepancy that changes component RATIOS (noise-free)")
t0=time.time()
variants={
 "nominal(=parent)": parent,
 "nu=0.25":          dataclasses.replace(parent, nu=0.25),
 "nu=0.40":          dataclasses.replace(parent, nu=0.40),
 "pmass_offcenter":  dataclasses.replace(parent, point_mass=0.1, point_mass_xy=(0.03,0.03)),
 "thick+15%":        dataclasses.replace(parent, thickness=0.00345),
}
oracle=run_case(truth,truth,noise=False,n_modes=2,use_truth_as_prior=True,**BASE)
orc={c:float(oracle["metrics"]["nrmse"][c]) for c in COMPS}
P("  ORACLE NRMSE: "+" ".join("%s=%.4f"%(c,orc[c]) for c in COMPS))
C={}
for name,cfg in variants.items():
    r=run_case(truth,cfg,noise=False,n_modes=2,**BASE)
    nr={c:float(r["metrics"]["nrmse"][c]) for c in COMPS}
    gap={c:nr[c]-orc[c] for c in COMPS}
    C[name]={"nrmse":nr,"gap_vs_oracle":gap}
    P("  %-18s NRMSE: "%name+" ".join("%s=%.4f"%(c,nr[c]) for c in COMPS))
results["C"]=C; results["C_oracle"]=orc; P("  (%.1f min)\n"%((time.time()-t0)/60))

json.dump(results, open("%s/studies_results.json"%TMP,"w"), indent=2)
open("%s/studies_log.txt"%TMP,"w").write("\n".join(log))
P("SAVED studies_results.json")
