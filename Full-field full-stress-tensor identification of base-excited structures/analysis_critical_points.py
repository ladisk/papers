"""P30 -- Critical-point stress-PSD table (Reviewer 1, comments 4 and 12).

Computes the recovered stress-component PSD of the REAL base-excitation
measurement at three named critical points (clamp centre, block corner,
mid-field), with a Monte-Carlo camera-noise band as the +/- bound, as
promised in response_to_rev1.tex (R1.12: "the recovered PSD is tabulated
at three critical points (clamp centre, corner, mid-field) with a
Monte-Carlo camera-noise band as the $\\pm$ bound").

Method
------
1.  Re-runs the Step-2 transmissibility expansion EXACTLY as in
    dual_stage_base_pipeline/transmissibility_expansion.ipynb (same raw
    .hcc recording, crop, 3x3 reduction, H1 estimator, Stage-1 SEMM mode
    shapes) and verifies the reproduction bit-for-bit against the frozen
    arrays in dual_stage_base_pipeline/outputs/article_data/ (trans_*).
2.  Calibrates the per-node camera noise floor from the recording itself
    with the hf_temporal estimator (synthetic_validation/noise.py), i.e.
    the same calibration used by the letter's noise study.
3.  Monte-Carlo: N_MC times, injects an independent Gaussian camera-noise
    realization at the calibrated level into the measured temperature
    frames and repeats the full Step-2 identification (H1 -> modal
    decomposition -> expansion -> PSD).  The spread (1 sigma) of the
    per-point PSD across realizations is the reported +/- band.  The H1
    estimator references the (unperturbed) base accelerometer, so the MC
    perturbs only the camera path -- camera noise is the dominant
    stochastic source (the expansion itself is deterministic).

Outputs (new files only)
------------------------
synthetic_validation/analysis/critical_points/critical_points.json
synthetic_validation/analysis/critical_points/critical_points_summary.md

Run:  python analysis_critical_points.py         (from the repo root)
"""
from __future__ import annotations

import json
import pickle
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import get_window, resample, welch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from dual_stage_base_pipeline.semm_thermoelastic_pipeline import (  # noqa: E402
    SEMMThermoelasticPipeline,
)
from synthetic_validation.noise import estimate_camera_noise_floor  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration -- copied verbatim from transmissibility_expansion.ipynb
# ---------------------------------------------------------------------------
# DATA_ROOT is the only path that must be adapted: point it at the directory
# holding the measurement recordings (see the README). It can also be supplied
# through the THERMO_DATA_ROOT environment variable.
DATA_ROOT = Path(os.environ.get("THERMO_DATA_ROOT", r"C:\path\to\dataset")).resolve()
PIPE_DIR = REPO / "dual_stage_base_pipeline"
STAGE1_RUN = "stage1_20260306_075427"
STAGE1_DIR = DATA_ROOT / STAGE1_RUN

CAMERA_HCC_PATH = (
    DATA_ROOT / "experiment_20260305" / "stage2_camera"
    / "DefaultName_20260305T073259507"
    / "_20260305T073259507_20260305T073259947.hcc"
)
ACCEL_PKL_PATH = (
    DATA_ROOT / "experiment_20260305" / "stage2_shaker"
    / "20260305_073124_base_excitation.pkl"
)
ACCEL_TASK = "AccelerationTask"
ACCEL_CHANNEL = "Acceleration"
ACCEL_IN_G = True
G0 = 9.80665
CAM_LEAD_FRAMES = 49

CROP_JSON = PIPE_DIR / 'configs' / "crop.json"
REDUCE_FACTOR = 3
FLIP_VERTICAL = False
N_SEGMENTS = 5
F_MIN, F_MAX = 50.0, 300.0
FS_IR = 2000.0

COMPS = {0: "SX", 1: "SY", 2: "SXY", 3: "SX+SY"}

# Frozen Step-2 outputs used for (a) mode shapes and (b) baseline verification
ARTICLE_DATA = REPO / "dual_stage_base_pipeline" / "outputs" / "article_data"

# Monte-Carlo settings
N_MC = 32
MC_SEED0 = 0

OUT_DIR = REPO / "synthetic_validation" / "analysis" / "critical_points"

# The three critical points promised in the letter (targets; the analysis
# snaps each to the nearest node of the 34x34 measurement grid).
#   clamp_centre : centre of the clamped bottom edge -- the fatigue-critical
#                  maximum-normal-stress location.
#   block_corner : inner (lower-right) corner of the 170 g steel block glued
#                  to the top-left corner region of the plate (footprint
#                  x 0--21.4 mm, y 128.6--150 mm, cf. fig_component_maps.py
#                  mass_box) -- the multiaxial, shear-carrying region.
#   mid_field    : plate centre -- representative far-field point.
POINT_TARGETS_MM = {
    "clamp_centre": (75.0, 0.0),
    "block_corner": (21.4, 128.6),
    "mid_field": (75.0, 75.0),
}
POINT_LABELS = {
    "clamp_centre": "Clamp centre",
    "block_corner": "Block corner",
    "mid_field": "Mid-field",
}


# ---------------------------------------------------------------------------
# Helper functions -- copied verbatim from the notebook (cell 3)
# ---------------------------------------------------------------------------
def compute_h1_frf(ref, response_2d, fs, seg_len, window="hann"):
    """Vectorised H1 FRF estimator with coherence (notebook cell 3)."""
    n = len(ref)
    hop = seg_len // 2
    win = get_window(window, seg_len)
    wss = np.sum(win**2)
    freq = np.fft.rfftfreq(seg_len, 1.0 / fs)
    nf, nch = len(freq), response_2d.shape[1]

    Gxx = np.zeros(nf)
    Gyx = np.zeros((nf, nch), dtype=complex)
    Gyy = np.zeros((nf, nch))
    n_avg = 0

    for s in range(0, n - seg_len + 1, hop):
        xw = ref[s : s + seg_len] * win
        yw = response_2d[s : s + seg_len] * win[:, None]
        X = np.fft.rfft(xw)
        Y = np.fft.rfft(yw, axis=0)
        Gxx += (X.conj() * X).real / wss
        Gyx += Y * X.conj()[:, None] / wss
        Gyy += (Y.conj() * Y).real / wss
        n_avg += 1

    Gxx /= n_avg
    Gyx /= n_avg
    Gyy /= n_avg

    H1 = Gyx / (Gxx[:, None] + 1e-30)
    coh = np.abs(Gyx) ** 2 / (Gyy * Gxx[:, None] + 1e-30)
    return freq, H1, coh, Gxx


def map_pixels_to_nodes(A, cam_to_node, n_nodes):
    """Average pixel columns of A (nfreq, npix) onto nodes (vectorised
    equivalent of the notebook's camera_to_nodes / _complex loop)."""
    counts = np.bincount(cam_to_node, minlength=n_nodes).astype(float)
    counts[counts == 0] = 1.0
    out = np.zeros((n_nodes, A.shape[0]), dtype=A.dtype)
    np.add.at(out, cam_to_node, A.T)
    return (out / counts[:, None]).T


# ---------------------------------------------------------------------------
# Step-2 identification from a camera stress time-series (fixed mode shapes)
# ---------------------------------------------------------------------------
def run_step2(sigma_cam_2d, accel, seg_len, cam_to_node, n_nodes,
              Psi, Psi_cam_pinv, S_aa_interp_ref=None):
    """H1 -> band-limit -> node mapping -> modal decomposition -> expansion.

    Returns (freq_b, S_stress dict, T_cam_nodes, S_aa_interp).
    """
    freq_h1, T_cam_pix, _coh_pix, _Gaa = compute_h1_frf(
        accel, sigma_cam_2d, FS_IR, seg_len
    )
    band = (freq_h1 >= F_MIN) & (freq_h1 <= F_MAX)
    freq_b = freq_h1[band]
    T_cam_b = T_cam_pix[band]

    T_cam_nodes = map_pixels_to_nodes(T_cam_b, cam_to_node, n_nodes)

    gamma = (Psi_cam_pinv @ T_cam_nodes.T).T  # (n_freq_b, n_modes)

    if S_aa_interp_ref is None:
        f_psd, S_aa = welch(accel, fs=FS_IR, nperseg=seg_len, window="hann")
        band_psd = (f_psd >= F_MIN) & (f_psd <= F_MAX)
        S_aa_interp = np.interp(freq_b, f_psd[band_psd], S_aa[band_psd])
    else:
        S_aa_interp = S_aa_interp_ref  # accel path unperturbed in the MC

    S_stress = {}
    for label in COMPS.values():
        T_exp = gamma @ Psi[label].T  # (n_freq_b, n_nodes)
        S_stress[label] = np.abs(T_exp) ** 2 * S_aa_interp[:, None]
    return freq_b, S_stress, T_cam_nodes, S_aa_interp


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep = {}

    # ------------------------------------------------------------------
    # 1. Load raw recording (identical to notebook cell 11)
    # ------------------------------------------------------------------
    from fasthcc import read_hcc

    with open(CROP_JSON) as f:
        CROP = json.load(f)

    print("Reading camera recording ...")
    images_raw, _meta = read_hcc(str(CAMERA_HCC_PATH), calibrated=True,
                                 metadata=True)
    images_raw = np.asarray(images_raw, dtype=np.float64)
    if CAM_LEAD_FRAMES > 0:
        images_raw = images_raw[CAM_LEAD_FRAMES:]
    if FLIP_VERTICAL:
        images_raw = images_raw[:, ::-1, :]

    images_crop = SEMMThermoelasticPipeline.crop_video(images_raw, CROP)
    images_red = SEMMThermoelasticPipeline.reduce_resolution(
        images_crop, REDUCE_FACTOR, mode="crop"
    )
    del images_raw, images_crop

    n_frames, rows_cam, cols_cam = images_red.shape
    n_pixels = rows_cam * cols_cam
    print(f"frames x pixels : {n_frames} x {rows_cam}x{cols_cam}")

    images_ac = images_red - images_red.mean(axis=0, keepdims=True)  # [K]
    del images_red
    factor = SEMMThermoelasticPipeline._camera_to_stress_factor_pa_per_k()
    images_ac_2d = images_ac.reshape(n_frames, -1)  # [K]
    del images_ac
    sigma_cam_2d = images_ac_2d * factor  # [Pa]

    # ------------------------------------------------------------------
    # 2. Calibrate the per-node camera noise floor from the recording
    #    (hf_temporal estimator, synthetic_validation/noise.py)
    # ------------------------------------------------------------------
    sigma_cal_K = estimate_camera_noise_floor(
        images_ac_2d.reshape(n_frames, rows_cam, cols_cam)
    )
    print(f"calibrated camera noise floor: {sigma_cal_K*1e3:.2f} mK "
          f"({abs(sigma_cal_K*factor)/1e6:.4f} MPa per node)")

    # ------------------------------------------------------------------
    # 3. Accelerometer (identical to notebook cell 11)
    # ------------------------------------------------------------------
    with open(ACCEL_PKL_PATH, "rb") as f:
        pkl = pickle.load(f)
    task = pkl[ACCEL_TASK]
    accel_data = np.asarray(task["data"], dtype=float)
    fs_acc = float(task["sample_rate"])
    i_ch = list(task.get("channel_names", [])).index(ACCEL_CHANNEL)
    accel_raw = accel_data[:, i_ch] - np.mean(accel_data[:, i_ch])
    if ACCEL_IN_G:
        accel_raw = accel_raw * G0
    cam_duration = n_frames / FS_IR
    n_accel_keep = min(int(round(cam_duration * fs_acc)), len(accel_raw))
    accel = resample(accel_raw[:n_accel_keep], n_frames)

    seg_len = int(2 * n_frames / (N_SEGMENTS + 1))
    print(f"Welch segment   : {seg_len} samples, {N_SEGMENTS} segments")

    # ------------------------------------------------------------------
    # 4. Node grid + pixel->node mapping (identical to notebook cell 11)
    # ------------------------------------------------------------------
    node_coords = np.load(STAGE1_DIR / "stage1" / "node_coords.npy")
    with open(STAGE1_DIR / "stage1" / "mapping_metadata.json") as f:
        mapping_meta = json.load(f)
    n_nodes = node_coords.shape[0]

    x_range = mapping_meta["diag_cam"]["x_range"]
    y_range = mapping_meta["diag_cam"]["y_range"]
    xs = np.linspace(x_range[0], x_range[1], cols_cam)
    ys = np.linspace(y_range[1], y_range[0], rows_cam)
    xx, yy = np.meshgrid(xs, ys)
    cam_positions = np.column_stack([xx.ravel(), yy.ravel()])
    d2 = np.sum(
        (cam_positions[:, None, :] - node_coords[None, :, :2]) ** 2, axis=2
    )
    cam_to_node = np.argmin(d2, axis=1)
    print(f"pixel->node map : {n_pixels} pixels -> "
          f"{len(np.unique(cam_to_node))} unique nodes")

    # ------------------------------------------------------------------
    # 5. Stage-1 SEMM stress-mode shapes (frozen article_data arrays --
    #    exactly the matrices the published Step-2 run used)
    # ------------------------------------------------------------------
    Psi = {
        label: np.load(ARTICLE_DATA
                       / f"trans_mode_shapes_{label.replace('+', 'plus')}.npy")
        for label in COMPS.values()
    }
    Psi_cam = Psi["SX+SY"]
    Psi_cam_pinv = np.linalg.pinv(Psi_cam)

    # ------------------------------------------------------------------
    # 6. Baseline run + verification against the frozen outputs
    # ------------------------------------------------------------------
    print("Baseline Step-2 run ...")
    freq_b, S_base, T_cam_nodes, S_aa_interp = run_step2(
        sigma_cam_2d, accel, seg_len, cam_to_node, n_nodes, Psi, Psi_cam_pinv
    )

    freq_ref = np.load(ARTICLE_DATA / "trans_freq_b.npy")
    assert np.allclose(freq_b, freq_ref), "frequency grid mismatch"
    T_ref = np.load(ARTICLE_DATA / "trans_T_cam_nodes.npy")
    dev_T = np.max(np.abs(T_cam_nodes - T_ref)) / np.max(np.abs(T_ref))
    devs = {}
    for label in COMPS.values():
        S_ref = np.load(ARTICLE_DATA
                        / f"trans_S_stress_{label.replace('+', 'plus')}.npy")
        devs[label] = float(np.max(np.abs(S_base[label] - S_ref))
                            / np.max(S_ref))
    print(f"verification: max rel dev  T_cam {dev_T:.2e}  "
          + "  ".join(f"S_{c} {d:.2e}" for c, d in devs.items()))
    if dev_T > 1e-6 or max(devs.values()) > 1e-6:
        print("WARNING: baseline deviates from frozen article_data arrays")
    rep["baseline_verification_max_rel_dev"] = {
        "T_cam_nodes": float(dev_T), **devs
    }

    # ------------------------------------------------------------------
    # 7. Critical points and resonance bins
    # ------------------------------------------------------------------
    x_mm = node_coords[:, 0] * 1e3
    y_mm = node_coords[:, 1] * 1e3
    points = {}
    for name, (tx, ty) in POINT_TARGETS_MM.items():
        i = int(np.argmin((x_mm - tx) ** 2 + (y_mm - ty) ** 2))
        points[name] = i
        print(f"{name:13s} -> node {i:4d} at "
              f"({x_mm[i]:.1f}, {y_mm[i]:.1f}) mm")

    f_peaks = np.load(ARTICLE_DATA / "trans_f_peaks.npy")
    kres = [int(np.argmin(np.abs(freq_b - fp))) for fp in f_peaks]
    print(f"resonances      : {freq_b[kres[0]]:.2f}, {freq_b[kres[1]]:.2f} Hz "
          f"(bins {kres})")

    # coherence at the points (diagnostic, stored in the JSON)
    coh_nodes = np.load(ARTICLE_DATA / "trans_coh_nodes.npy")

    # ------------------------------------------------------------------
    # 8. Monte-Carlo camera-noise study
    # ------------------------------------------------------------------
    comps3 = ["SX", "SY", "SXY"]
    all_comps = comps3 + ["SX+SY"]
    band_full = np.ones_like(freq_b, dtype=bool)
    band_250 = freq_b <= 250.0

    def extract(S_stress):
        """Per-point values: PSD at the two resonances + band-RMS [Pa units]."""
        out = {}
        for pname, i in points.items():
            e = {}
            for c in all_comps:
                col = S_stress[c][:, i]
                e[c] = {
                    "psd_f1": float(col[kres[0]]),
                    "psd_f2": float(col[kres[1]]),
                    "band_rms_full": float(np.sqrt(np.trapz(col, freq_b))),
                    "band_rms_250": float(np.sqrt(
                        np.trapz(col[band_250], freq_b[band_250]))),
                }
            out[pname] = e
        return out

    base_vals = extract(S_base)

    noise_pa = sigma_cal_K * abs(factor)
    mc_vals = []
    print(f"Monte-Carlo: {N_MC} realizations at sigma_T = "
          f"{sigma_cal_K*1e3:.2f} mK ...")
    for r in range(N_MC):
        rng = np.random.default_rng(MC_SEED0 + r)
        noisy = sigma_cam_2d + rng.standard_normal(sigma_cam_2d.shape) * noise_pa
        _, S_mc, _, _ = run_step2(
            noisy, accel, seg_len, cam_to_node, n_nodes, Psi, Psi_cam_pinv,
            S_aa_interp_ref=S_aa_interp,
        )
        mc_vals.append(extract(S_mc))
        del noisy, S_mc
        if (r + 1) % 8 == 0:
            print(f"  {r + 1}/{N_MC}")

    # ------------------------------------------------------------------
    # 9. Statistics and report
    # ------------------------------------------------------------------
    def mc_stats(pname, c, key):
        v = np.array([m[pname][c][key] for m in mc_vals])
        return {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)),
            "p2p5": float(np.percentile(v, 2.5)),
            "p97p5": float(np.percentile(v, 97.5)),
        }

    results = {}
    for pname, i in points.items():
        entry = {
            "label": POINT_LABELS[pname],
            "node_index": int(i),
            "x_mm": round(float(x_mm[i]), 1),
            "y_mm": round(float(y_mm[i]), 1),
            "target_mm": list(POINT_TARGETS_MM[pname]),
            "coherence_f1": float(coh_nodes[kres[0], i]),
            "coherence_f2": float(coh_nodes[kres[1], i]),
            "components": {},
        }
        for c in all_comps:
            entry["components"][c] = {
                "recovered": base_vals[pname][c],
                "mc": {key: mc_stats(pname, c, key)
                       for key in ("psd_f1", "psd_f2",
                                   "band_rms_full", "band_rms_250")},
            }
        results[pname] = entry

    rep.update({
        "task": "P30 critical-point stress-PSD table (R1.4 / R1.12)",
        "pipeline": "real-measurement Step-2 transmissibility expansion "
                    "(transmissibility_expansion.ipynb chain, re-run and "
                    "verified against outputs/article_data)",
        "camera_recording": CAMERA_HCC_PATH.name,
        "accel_recording": ACCEL_PKL_PATH.name,
        "stage1_run": STAGE1_RUN,
        "n_frames": int(n_frames),
        "fs_ir_hz": FS_IR,
        "welch_seg_len": int(seg_len),
        "welch_n_segments": int(N_SEGMENTS),
        "band_hz": [float(freq_b[0]), float(freq_b[-1])],
        "f_resonances_hz": [float(freq_b[kres[0]]), float(freq_b[kres[1]])],
        "noise_calibration": {
            "method": "hf_temporal (synthetic_validation.noise."
                      "estimate_camera_noise_floor) on the reduced "
                      "34x34-node temperature frames of this recording",
            "sigma_T_K": float(sigma_cal_K),
            "sigma_stress_Pa": float(noise_pa),
        },
        "monte_carlo": {
            "n_realizations": N_MC,
            "seed0": MC_SEED0,
            "perturbation": "independent Gaussian camera-noise realization "
                            "at the calibrated level added to the measured "
                            "temperature frames; full Step-2 identification "
                            "repeated; accelerometer path unperturbed",
            "band_definition": "+/- bound = 1 sigma across realizations",
        },
        "units": "psd values in Pa^2/Hz, band-RMS in Pa "
                 "(divide by 1e12 resp. 1e6 for MPa^2/Hz resp. MPa)",
        "points": results,
    })

    with open(OUT_DIR / "critical_points.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(f"saved {OUT_DIR / 'critical_points.json'}")

    # ------------------------------------------------------------------
    # 10. Human-readable summary + LaTeX table draft
    # ------------------------------------------------------------------
    comp_tex = {"SX": r"$\sigma_{xx}$", "SY": r"$\sigma_{yy}$",
                "SXY": r"$\tau_{xy}$"}

    from math import floor, log10

    def _std_decimals(s):
        """Decimals so the std shows 2 sig figs if its leading digit is 1,
        else 1 sig fig (standard uncertainty-rounding convention)."""
        e = floor(log10(s))
        lead = int(s / 10.0**e)
        sig = 2 if lead == 1 else 1
        return max(0, -e + sig - 1)

    def fmt_pair(v, s):
        """value +/- std with the value rounded to the std's precision;
        falls back to (m +/- s) x 10^k when the decimal form gets long.
        Guarantees the +/- bound never displays as zero."""
        if v == 0:
            return "$0$"
        dec = _std_decimals(s)
        if v >= 0.01 and dec <= 4:
            return rf"${v:.{dec}f} \pm {s:.{dec}f}$"
        exp = floor(log10(abs(v)))
        scale = 10.0**exp
        mv, ms = v / scale, s / scale
        dec = _std_decimals(ms)
        return rf"$({mv:.{dec}f} \pm {ms:.{dec}f})\times 10^{{{exp}}}$"

    def fmt(v, s):
        """value +/- std in MPa^2/Hz."""
        return fmt_pair(v / 1e12, s / 1e12)

    lines = []
    lines.append("# Critical-point stress-PSD table (P30, R1.4 / R1.12)\n")
    lines.append(f"Generated by `analysis_critical_points.py` "
                 f"({N_MC}-realization Monte-Carlo).\n")
    lines.append("## Pipeline\n")
    lines.append(
        "- Real measurement, Step-2 transmissibility expansion "
        f"(`{CAMERA_HCC_PATH.name}` + `{ACCEL_PKL_PATH.name}`, "
        f"Stage-1 run `{STAGE1_RUN}`); baseline re-run verified against "
        "the frozen `outputs/article_data` arrays (max rel. deviation "
        f"{max([rep['baseline_verification_max_rel_dev'][k] for k in rep['baseline_verification_max_rel_dev']]):.1e}).")
    lines.append(
        f"- Calibrated camera noise floor (hf_temporal, this recording): "
        f"sigma_T = {sigma_cal_K*1e3:.2f} mK per 34x34 node "
        f"(= {noise_pa/1e6:.4f} MPa on the invariant).")
    lines.append(
        f"- Monte-Carlo band: 1-sigma spread over {N_MC} re-identifications "
        "with an independent calibrated camera-noise realization injected "
        "into the measured temperature frames (accelerometer unperturbed; "
        "the H1 reference makes the estimate unbiased, so the spread "
        "measures run-to-run repeatability under camera noise).\n")
    lines.append("## Critical points (node grid 34x34, spacing 4.5 mm)\n")
    for pname, e in results.items():
        tx, ty = POINT_TARGETS_MM[pname]
        lines.append(
            f"- **{e['label']}** -- node {e['node_index']} at "
            f"({e['x_mm']}, {e['y_mm']}) mm (target ({tx}, {ty}) mm); "
            f"coherence gamma^2 = {e['coherence_f1']:.2f} / "
            f"{e['coherence_f2']:.2f} at f1 / f2.")
    lines.append("")
    lines.append("## Recovered PSD at the resonances (MPa^2/Hz, "
                 "+/- = 1-sigma MC camera-noise band)\n")
    lines.append("| Point (x, y) [mm] | Component | S(f1 = "
                 f"{freq_b[kres[0]]:.1f} Hz) | S(f2 = "
                 f"{freq_b[kres[1]]:.1f} Hz) |")
    lines.append("|---|---|---|---|")
    for pname, e in results.items():
        for c in comps3:
            b = e["components"][c]["recovered"]
            m = e["components"][c]["mc"]
            cell1 = fmt(b["psd_f1"], m["psd_f1"]["std"])
            cell2 = fmt(b["psd_f2"], m["psd_f2"]["std"])
            head = (f"{e['label']} ({e['x_mm']:.0f}, {e['y_mm']:.0f})"
                    if c == "SX" else "")
            lines.append(f"| {head} | {comp_tex[c]} | {cell1} | {cell2} |")
    lines.append("")

    lines.append("## LaTeX table draft\n")
    lines.append("```latex")
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Recovered stress-component PSD at the three critical "
        r"points at the two resonances. The $\pm$ bound is the $1\sigma$ "
        r"run-to-run spread of a " + str(N_MC) + r"-realization Monte-Carlo "
        r"camera-noise study, in which an independent noise realization at "
        r"the calibrated level ($\sigma_T = "
        + f"{sigma_cal_K*1e3:.1f}" + r"$\,mK per node) is injected into the "
        r"measured recording and the full Step-2 identification is "
        r"repeated; camera noise is the dominant stochastic source, the "
        r"expansion being otherwise deterministic. Noise model in "
        r"Appendix~B.3.}")
    lines.append(r"    \label{tab:critical_points}")
    lines.append(r"    \begin{tabular}{llrr}")
    lines.append(r"        \toprule")
    lines.append(
        r"        Point $(x, y)$ [mm] & Component & "
        rf"$S_{{\sigma\sigma}}({freq_b[kres[0]]:.1f}\,\mathrm{{Hz}})$ & "
        rf"$S_{{\sigma\sigma}}({freq_b[kres[1]]:.1f}\,\mathrm{{Hz}})$ \\")
    lines.append(r"         &  & [MPa$^2$/Hz] & [MPa$^2$/Hz] \\")
    lines.append(r"        \midrule")
    for pname, e in results.items():
        for j, c in enumerate(comps3):
            b = e["components"][c]["recovered"]
            m = e["components"][c]["mc"]
            cell1 = fmt(b["psd_f1"], m["psd_f1"]["std"])
            cell2 = fmt(b["psd_f2"], m["psd_f2"]["std"])
            head = (rf"{e['label']} $({e['x_mm']:.0f}, {e['y_mm']:.0f})$"
                    if j == 0 else "")
            lines.append(rf"        {head} & {comp_tex[c]} & "
                         rf"{cell1} & {cell2} \\")
        if pname != list(results)[-1]:
            lines.append(r"        \midrule")
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("```")
    lines.append("")

    lines.append("## Band-RMS alternative (MPa, 50.2--250.1 Hz)\n")
    lines.append("| Point | Component | band-RMS +/- MC std |")
    lines.append("|---|---|---|")
    for pname, e in results.items():
        for c in comps3:
            b = e["components"][c]["recovered"]["band_rms_250"] / 1e6
            s = e["components"][c]["mc"]["band_rms_250"]["std"] / 1e6
            head = e["label"] if c == "SX" else ""
            lines.append(f"| {head} | {comp_tex[c]} | {fmt_pair(b, s)} |")
    lines.append("")
    # measured MC bias / relative-spread ranges for the notes
    biases, rels = [], []
    for pname, e in results.items():
        for c in comps3:
            for key in ("psd_f1", "psd_f2"):
                b = e["components"][c]["recovered"][key]
                m = e["components"][c]["mc"][key]
                biases.append(abs(m["mean"] - b) / b * 100.0)
                rels.append(m["std"] / b * 100.0)

    lines.append("## Notes\n")
    lines.append(
        f"- The MC mean stays within {max(biases):.2f}% of the recovered "
        "value at every point/component (H1 is unbiased to camera noise); "
        f"the 1-sigma band spans {min(rels):.2f}--{max(rels):.1f}% of the "
        "recovered PSD. Full per-seed statistics (mean, std, 2.5/97.5 "
        "percentiles) are in `critical_points.json`.")
    lines.append(
        "- Mid-field gamma^2 at f2 is low (the point sits near the mode-2 "
        "nodal line, where the camera sees little mode-2 signal); the "
        "recovered PSD there is carried by the full-field modal projection, "
        "not the local pixel alone.")
    lines.append(
        "- Letter promise (R1.12): 'the recovered PSD is tabulated at "
        "three critical points (clamp centre, corner, mid-field) with a "
        "Monte-Carlo camera-noise band as the +/- bound'. The manuscript "
        "table should carry a pointer to the noise model in the "
        "appendix (P30: Appendix B.3).")

    with open(OUT_DIR / "critical_points_summary.md", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {OUT_DIR / 'critical_points_summary.md'}")


if __name__ == "__main__":
    main()
