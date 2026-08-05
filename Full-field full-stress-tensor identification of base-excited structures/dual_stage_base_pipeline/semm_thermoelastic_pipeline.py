from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import get_window, resample

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


@dataclass
class Stage1Config:
    f_min: float = 45.0
    f_max: float = 300.0
    n_segments: int = 1
    fs_ir_fallback: float = 2000.0
    crop: Optional[Dict[str, int]] = None
    reduce_factor: int = 3
    flip_vertical: bool = True
    flip_horizontal: bool = False
    force_task_name: str = "AccelerationTask"
    force_channel_name: str = "Force"
    use_all_cases: bool = True
    case_idx: int = 0
    plate_lx: float = 0.15
    plate_ly: float = 0.15
    force_points: Optional[Dict[str, Tuple[float, float]]] = None
    enable_impact_sync_crop: bool = True
    impact_pre_ms: float = 5.0
    impact_noise_ref_s: float = 0.25
    impact_start_rms_mult: float = 4.0
    impact_start_rel_peak: float = 0.01
    impact_end_rms_mult: float = 1.0
    impact_end_rel_peak: float = 0.01
    impact_quiet_hold_ms: float = 2.0
    force_zero_post_ms: float = 0.5
    force_end_taper_ms: float = 3.0
    exp_window_residual: float = 0.01  # exponential window end-level (0.01 = 1%); 0 = disabled
    camera_flip_x: bool = False
    camera_flip_y: bool = False


@dataclass
class Stage2Config:
    crop_mode: str = "inherit_stage1"
    stage2_crop: Optional[Dict[str, int]] = None
    fs_ir_fallback: float = 2000.0
    n_segments: int = 5
    reduce_factor: Optional[int] = None
    flip_vertical: bool = True
    flip_horizontal: bool = False
    acc_task_name: str = "AccelerationTask"
    acc_channel_name: str = "Acceleration"
    acc_in_g: bool = True
    acc_g0: float = 9.80665
    remove_dc: bool = True
    cache_stage2_camera_npy: bool = True
    overlay_component: int = 3
    equiv_regularization: float = 1e-12
    # Fit equivalent inputs on every Nth interface DoF (1 = all DoFs).
    # Using >1 prevents over-fitting the same DoFs later used by SEMM.
    equiv_fit_every_n_dof: int = 1
    parent_basis: str = "Y_SEMM"
    # 'virtual_base': build q only from stage-1 metadata (no stage-2 overlay fit)
    # 'fit_overlay': identify q by fitting stage-2 overlay interface
    parent_q_mode: str = "virtual_base"
    # Scaling from unit base acceleration to equivalent force basis [N/(m/s^2)].
    virtual_base_mass_kg: float = 1.0
    # Optional Gaussian width for spatial weighting of hammer inputs [m].
    virtual_base_sigma_m: Optional[float] = None
    # Optional explicit weights by input case label, e.g. {"left_top": 0.2, ...}.
    virtual_base_input_weights: Optional[Dict[str, float]] = None
    virtual_input_case_name: str = "base_excitation"
    virtual_input_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class PathsConfig:
    project_root: str = "."
    solver_script: str = "plate_stress_modal_superposition_force_pointmass_v2_camfreq_20260206_1239.py"
    stage2_camera_hcc: str = r"base_excitation_measurements\best_one\_20260217T123846487_20260217T123846978.hcc"
    stage2_camera_npy: Optional[str] = None
    stage2_accel_pkl: str = r"base_excitation_measurements\best_one\20260217_123809_base_excitation.pkl"
    crop_json: str = "dual_stage_base_pipeline/configs/crop.json"
    stage2_crop_json: str = "dual_stage_base_pipeline/configs/stage2_crop.json"
    output_root: str = "dual_stage_base_pipeline/outputs"
    stage1_run_root: str = "dual_stage_base_pipeline/outputs/_stage1_numerical_runs"
    stage1_cases: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "center_middle": {
            "pkl_path": r"hammer_measurement_6_points/force/middle_center.pkl",
            "npy_path": r"hammer_measurement_6_points/camera/npy/center_middle_ir_images.npy",
        },
        "center_top": {
            "pkl_path": r"hammer_measurement_6_points/force/top_center.pkl",
            "npy_path": r"hammer_measurement_6_points/camera/npy/center_top_ir_images.npy",
        },
        "left_top": {
            "pkl_path": r"hammer_measurement_6_points/force/top_left.pkl",
            "npy_path": r"hammer_measurement_6_points/camera/npy/left_top_ir_images.npy",
        },
        "left_middle": {
            "pkl_path": r"hammer_measurement_6_points/force/middle_left.pkl",
            "npy_path": r"hammer_measurement_6_points/camera/npy/left_middle_ir_images.npy",
        },
        "right_middle": {
            "pkl_path": r"hammer_measurement_6_points/force/middle_right.pkl",
            "npy_path": r"hammer_measurement_6_points/camera/npy/right_middle_ir_images.npy",
        },
        "right_top": {
            "pkl_path": r"hammer_measurement_6_points/force/top_right.pkl",
            "npy_path": r"hammer_measurement_6_points/camera/npy/right_top_ir_images.npy",
        },
    })


@dataclass
class SEMMConfig:
    f_semm_min: float = 40.0
    f_semm_max: float = 300.0
    semm_lines_per_chunk: int = 1
    log_every_n_chunks: int = 5
    semm_type: str = "fully-extend-svd"


class SEMMThermoelasticPipeline:
    def __init__(
        self,
        paths: Optional[PathsConfig] = None,
        stage1: Optional[Stage1Config] = None,
        stage2: Optional[Stage2Config] = None,
        semm: Optional[SEMMConfig] = None,
    ) -> None:
        self.paths = paths or PathsConfig()
        self.stage1_cfg = stage1 or Stage1Config()
        self.stage2_cfg = stage2 or Stage2Config()
        self.semm_cfg = semm or SEMMConfig()

        self._root = Path(self.paths.project_root).resolve()
        self.state: Dict[str, Any] = {"stage1": {}, "stage2": {}, "meta": {}}

        # Auto-load crop from JSON if not set explicitly
        if self.stage1_cfg.crop is None:
            crop_file = self._resolve_path(self.paths.crop_json)
            if crop_file.exists():
                self.load_crop(crop_file)

    # -------------------------- basic utilities --------------------------
    def _resolve_path(self, p: Any) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else self._root / pp

    @staticmethod
    def _jsonable(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): SEMMThermoelasticPipeline._jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [SEMMThermoelasticPipeline._jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, Path):
            return str(x)
        return x

    def _write_json(self, p: Path, d: Any) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self._jsonable(d), f, indent=2)

    @staticmethod
    def _normalize_crop(crop: Dict[str, Any]) -> Dict[str, int]:
        keys = ["row_start", "row_end", "col_start", "col_end"]
        for k in keys:
            if k not in crop:
                raise ValueError(f"Missing crop key: {k}")
        out = {k: int(crop[k]) for k in keys}
        if out["row_end"] <= out["row_start"] or out["col_end"] <= out["col_start"]:
            raise ValueError(f"Invalid crop bounds: {out}")
        return out

    @staticmethod
    def reduce_resolution(data: np.ndarray, factor: Optional[int], mode: str = "crop") -> np.ndarray:
        data = np.asarray(data)
        if data.ndim != 3:
            raise ValueError("Input data must be 3D: (frequency, row, col).")
        if factor is None or int(factor) <= 1:
            return data
        factor = int(factor)
        nf, nr, nc = data.shape
        if mode == "crop":
            nr2 = nr - (nr % factor)
            nc2 = nc - (nc % factor)
            d = data[:, :nr2, :nc2]
        elif mode == "pad":
            pr = (factor - nr % factor) % factor
            pc = (factor - nc % factor) % factor
            d = np.pad(data, ((0, 0), (0, pr), (0, pc)), mode="constant", constant_values=0)
            nr2, nc2 = d.shape[1:]
        else:
            raise ValueError("mode must be 'crop' or 'pad'")
        return d.reshape(nf, nr2 // factor, factor, nc2 // factor, factor).mean(axis=(2, 4))

    @staticmethod
    def frf_ir_h1(ref_signal: np.ndarray, fs_ref: float, ir_data: np.ndarray, fs_ir: float, n_segments: int, exp_window_residual: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        del fs_ref
        x = np.asarray(ref_signal, dtype=float)
        y = np.asarray(ir_data)
        # Remove temporal mean (large thermal offset) from each pixel before FFT
        y = y - np.mean(y, axis=0, keepdims=True)
        n = y.shape[0]
        x = resample(x, n)
        x = x - np.mean(x, dtype=float)

        # Exponential window for impact data: w(t) = exp(-alpha*t), w(T) = residual
        if exp_window_residual > 0 and n > 1:
            alpha = -np.log(exp_window_residual) / (n - 1)
            exp_win = np.exp(-alpha * np.arange(n))
            x = x * exp_win
            y = y * exp_win[:, None, None]

        seg_len = int(2 * n / (n_segments + 1))
        if seg_len <= 1:
            raise ValueError("Segment length too short.")
        hop = seg_len // 2

        win = get_window("boxcar", seg_len) if n_segments <= 1 else get_window("hann", seg_len)
        wn = np.sum(win ** 2)
        freq = np.fft.rfftfreq(seg_len, 1.0 / float(fs_ir))

        rows, cols = y.shape[1:]
        g_xx = np.zeros(len(freq), dtype=float)
        g_yx = np.zeros((len(freq), rows, cols), dtype=np.complex128)
        g_yy = np.zeros((len(freq), rows, cols), dtype=float)

        n_used = 0
        for i in range(0, n - seg_len + 1, hop):
            xx = x[i:i + seg_len] * win
            yy = y[i:i + seg_len, :, :] * win[:, None, None]
            X = np.fft.rfft(xx, seg_len)
            Y = np.fft.rfft(yy, seg_len, axis=0)
            g_xx += (X * np.conj(X)).real / wn
            g_yx += (Y * np.conj(X)[:, None, None]) / wn
            g_yy += (Y * np.conj(Y)).real / wn
            n_used += 1

        if n_used == 0:
            raise RuntimeError("No segments used in FRF estimation.")
        g_xx /= n_used
        g_yx /= n_used
        g_yy /= n_used

        # Guard against division by near-zero G_xx
        g_xx_threshold = np.max(g_xx) * 1e-10
        g_xx_safe = np.where(g_xx > g_xx_threshold, g_xx, g_xx_threshold)

        H1 = g_yx / g_xx_safe[:, None, None]

        # Ordinary coherence: gamma^2 = |G_yx|^2 / (G_xx * G_yy)
        coherence = np.abs(g_yx) ** 2 / (g_xx_safe[:, None, None] * np.where(g_yy > 0, g_yy, 1.0))
        coherence = np.clip(coherence, 0.0, 1.0)

        return freq, H1, coherence

    @staticmethod
    def detect_force_impact_bounds(
        force: np.ndarray,
        fs: float,
        noise_ref_s: float = 0.25,
        start_rms_mult: float = 6.0,
        start_rel_peak: float = 0.02,
        end_rms_mult: float = 3.0,
        end_rel_peak: float = 0.01,
        quiet_hold_ms: float = 2.0,
    ) -> Dict[str, Any]:
        force = np.asarray(force, dtype=float)
        if force.ndim != 1 or len(force) < 10:
            raise ValueError("force must be a 1D array with at least 10 samples")

        absf = np.abs(force)
        i_peak = int(np.argmax(absf))
        peak_abs = float(absf[i_peak])
        if peak_abs <= 0:
            raise RuntimeError("Force peak is zero; cannot detect impact.")

        n_noise = max(8, int(round(float(noise_ref_s) * float(fs))))
        i0 = max(0, i_peak - n_noise)
        noise_seg = absf[i0:i_peak] if i_peak > i0 else absf[:n_noise]
        if noise_seg.size == 0:
            noise_seg = absf[: min(len(absf), n_noise)]
        noise_rms = float(np.sqrt(np.mean(noise_seg**2) + 1e-30))

        thr_start = max(float(start_rms_mult) * noise_rms, float(start_rel_peak) * peak_abs, 1e-12)
        thr_end = max(float(end_rms_mult) * noise_rms, float(end_rel_peak) * peak_abs, 1e-12)

        left = i_peak
        while left > 0 and absf[left] >= thr_start:
            left -= 1
        i_start = 0 if (left == 0 and absf[left] >= thr_start) else min(i_peak, left + 1)

        quiet_n = max(1, int(round(float(quiet_hold_ms) * 1e-3 * float(fs))))
        i_end = len(force) - 1
        found_end = False
        j = i_peak
        while j < len(force) - quiet_n:
            if np.all(absf[j:j + quiet_n] <= thr_end):
                i_end = j
                found_end = True
                break
            j += 1

        return {
            "i_peak": int(i_peak),
            "i_start": int(i_start),
            "i_end": int(i_end),
            "t_peak_s": float(i_peak / fs),
            "t_start_s": float(i_start / fs),
            "t_end_s": float(i_end / fs),
            "peak_abs": peak_abs,
            "noise_rms": noise_rms,
            "thr_start": float(thr_start),
            "thr_end": float(thr_end),
            "quiet_n": int(quiet_n),
            "found_end": bool(found_end),
        }

    @classmethod
    def sync_crop_and_window_force_camera(
        cls,
        force: np.ndarray,
        fs_force: float,
        images: np.ndarray,
        fs_ir: float,
        pre_ms: float = 5.0,
        noise_ref_s: float = 0.25,
        start_rms_mult: float = 6.0,
        start_rel_peak: float = 0.02,
        end_rms_mult: float = 3.0,
        end_rel_peak: float = 0.01,
        quiet_hold_ms: float = 2.0,
        force_zero_post_ms: float = 0.0,
        force_end_taper_ms: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
        force = np.asarray(force, dtype=float)
        images = np.asarray(images)

        det = cls.detect_force_impact_bounds(
            force, fs_force,
            noise_ref_s=noise_ref_s,
            start_rms_mult=start_rms_mult,
            start_rel_peak=start_rel_peak,
            end_rms_mult=end_rms_mult,
            end_rel_peak=end_rel_peak,
            quiet_hold_ms=quiet_hold_ms,
        )

        pre_n = max(0, int(round(float(pre_ms) * 1e-3 * float(fs_force))))
        i_crop_force = max(0, det["i_start"] - pre_n)
        t_crop_s = float(i_crop_force / fs_force)
        i_crop_ir = max(0, int(round(t_crop_s * float(fs_ir))))

        force_crop = force[i_crop_force:].copy()
        images_crop = images[i_crop_ir:, ...].copy()

        i_end_rel = max(0, det["i_end"] - i_crop_force)
        post_n = max(0, int(round(float(force_zero_post_ms) * 1e-3 * float(fs_force))))
        i_zero_start = min(len(force_crop), i_end_rel + post_n)
        taper_n = max(0, int(round(float(force_end_taper_ms) * 1e-3 * float(fs_force))))

        force_proc = force_crop.copy()
        i_zero_apply = i_zero_start
        if taper_n > 0 and i_zero_start < len(force_proc):
            i_taper_end = min(len(force_proc), i_zero_start + taper_n)
            if i_taper_end > i_zero_start:
                u = np.linspace(0.0, 1.0, i_taper_end - i_zero_start, endpoint=False)
                taper = 0.5 * (1.0 + np.cos(np.pi * u))
                force_proc[i_zero_start:i_taper_end] *= taper
                i_zero_apply = i_taper_end
        if i_zero_apply < len(force_proc):
            force_proc[i_zero_apply:] = 0.0

        info = dict(det)
        info.update({
            "i_crop_force": int(i_crop_force),
            "i_crop_ir": int(i_crop_ir),
            "t_crop_s": float(t_crop_s),
            "i_end_rel": int(i_end_rel),
            "i_zero_start_rel": int(i_zero_start),
            "i_zero_apply_rel": int(i_zero_apply),
            "t_zero_apply_s": float(t_crop_s + i_zero_apply / fs_force),
            "n_force_crop": int(len(force_crop)),
            "n_ir_crop": int(images_crop.shape[0]),
        })
        return force_proc, images_crop, info, force_crop

    @staticmethod
    def crop_video(images: np.ndarray, crop: Optional[Dict[str, int]]) -> np.ndarray:
        if crop is None:
            return np.asarray(images)
        return np.asarray(images)[:, crop.get("row_start", None):crop.get("row_end", None), crop.get("col_start", None):crop.get("col_end", None)]

    @staticmethod
    def interp_complex_to_target(freq_src: np.ndarray, H_src_flat: np.ndarray, freq_tgt: np.ndarray) -> np.ndarray:
        freq_src = np.asarray(freq_src, dtype=float)
        freq_tgt = np.asarray(freq_tgt, dtype=float)
        H = np.asarray(H_src_flat)

        if np.any(np.diff(freq_src) <= 0):
            idx_sort = np.argsort(freq_src)
            freq_src = freq_src[idx_sort]
            H = H[idx_sort, :]

        # Decompose into magnitude and unwrapped phase
        mag = np.abs(H)
        phase = np.unwrap(np.angle(H), axis=0)

        idx_hi = np.searchsorted(freq_src, freq_tgt, side="left")
        idx_lo = idx_hi - 1
        oob = (idx_lo < 0) | (idx_hi >= len(freq_src))

        idx_lo = np.clip(idx_lo, 0, len(freq_src)-1)
        idx_hi = np.clip(idx_hi, 0, len(freq_src)-1)

        x0 = freq_src[idx_lo]
        x1 = freq_src[idx_hi]
        denom = np.where((x1-x0)==0.0, 1.0, (x1-x0))
        w = (freq_tgt - x0) / denom

        # Interpolate magnitude and phase separately
        mag_interp = (1.0 - w)[:, None] * mag[idx_lo, :] + w[:, None] * mag[idx_hi, :]
        phase_interp = (1.0 - w)[:, None] * phase[idx_lo, :] + w[:, None] * phase[idx_hi, :]

        # Convert back to complex
        H_tgt = mag_interp * np.exp(1j * phase_interp)
        H_tgt[oob, :] = np.nan + 1j*np.nan
        return H_tgt

    @staticmethod
    def build_camera_df_from_grid(node_coords_xyz: np.ndarray, rows: int, cols: int, flip_x: bool = False, flip_y: bool = False) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Invalid grid size rows={rows}, cols={cols}")
        node_coords_xyz = np.asarray(node_coords_xyz, dtype=float)
        node_unique = np.unique(node_coords_xyz, axis=0)

        x_min, x_max = np.min(node_unique[:, 0]), np.max(node_unique[:, 0])
        y_min, y_max = np.min(node_unique[:, 1]), np.max(node_unique[:, 1])
        z_ref = float(np.nanmedian(node_unique[:, 2]))

        x_axis = np.linspace(x_min, x_max, cols)
        y_axis = np.linspace(y_max, y_min, rows)
        if flip_x:
            x_axis = x_axis[::-1]
        if flip_y:
            y_axis = y_axis[::-1]

        xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
        camera_positions = np.column_stack([xx.reshape(-1), yy.reshape(-1), np.full(rows * cols, z_ref, dtype=float)])

        df_camera = pd.DataFrame(camera_positions, columns=["Position_1", "Position_2", "Position_3"])
        df_camera["Direction_1"] = 0.5
        df_camera["Direction_2"] = 0.5
        df_camera["Direction_3"] = 0.0

        num_xy = node_unique[:, :2]
        cam_xy = camera_positions[:, :2]
        d2 = np.sum((cam_xy[:, None, :] - num_xy[None, :, :]) ** 2, axis=2)
        min_dist = np.sqrt(np.min(d2, axis=1))

        diagnostics = {
            "rows": int(rows),
            "cols": int(cols),
            "n_camera_points": int(camera_positions.shape[0]),
            "n_numerical_nodes_unique": int(node_unique.shape[0]),
            "x_range": (float(x_min), float(x_max)),
            "y_range": (float(y_min), float(y_max)),
            "flip_x": bool(flip_x),
            "flip_y": bool(flip_y),
            "nearest_node_dist_mean": float(np.mean(min_dist)),
            "nearest_node_dist_max": float(np.max(min_dist)),
            "first_point": camera_positions[0].tolist(),
            "last_point": camera_positions[-1].tolist(),
        }
        return df_camera, camera_positions, diagnostics
    def load_signal_from_pkl(
        self,
        pkl_path: Path,
        task_name: str,
        channel_name: str,
        remove_dc: bool = True,
    ) -> Tuple[np.ndarray, float, np.ndarray, Dict[str, Any]]:
        with pkl_path.open("rb") as f:
            d = pickle.load(f)
        if task_name not in d:
            raise KeyError(f"Task '{task_name}' not found in {pkl_path}. Keys: {list(d.keys())}")
        task = d[task_name]
        data = np.asarray(task["data"])
        fs = float(task["sample_rate"])
        time_axis = np.asarray(task.get("time", np.arange(len(data)) / fs), dtype=float)
        ch_names = list(task.get("channel_names", []))
        if channel_name not in ch_names:
            raise KeyError(f"'{channel_name}' not in channel_names={ch_names} for {pkl_path}")
        i_ch = ch_names.index(channel_name)
        sig = data[:, i_ch].astype(float)
        if remove_dc:
            sig = sig - np.mean(sig)
        return sig, fs, time_axis, {"fs": fs, "i_channel": i_ch, "channel_names": ch_names, "n": len(sig)}

    def load_force_from_pkl(
        self,
        pkl_path: Path,
        task_name: str = "AccelerationTask",
        force_name: str = "Force",
    ) -> Tuple[np.ndarray, float, np.ndarray, Dict[str, Any]]:
        return self.load_signal_from_pkl(Path(pkl_path), task_name, force_name, remove_dc=True)

    def _load_hcc_images(
        self,
        hcc_path: Path,
        fs_ir_fallback: float,
        flip_vertical: bool,
        flip_horizontal: bool,
    ) -> Tuple[np.ndarray, float]:
        try:
            from fasthcc import read_hcc
        except Exception as exc:
            raise ImportError("fasthcc is required for loading .hcc files. Install with: pip install fasthcc") from exc

        frames, meta = read_hcc(str(hcc_path), calibrated=True, metadata=True)
        images = np.asarray(frames, dtype=float)

        # Estimate frame rate from metadata timestamps
        fs_ir = float(fs_ir_fallback)
        if len(meta) >= 2:
            try:
                t0 = meta[0]['POSIXTime'] + meta[0].get('SubSecondTime', 0) / 1e6
                t1 = meta[-1]['POSIXTime'] + meta[-1].get('SubSecondTime', 0) / 1e6
                dt = (t1 - t0) / (len(meta) - 1)
                if dt > 0:
                    fs_ir = 1.0 / dt
            except (KeyError, TypeError, ZeroDivisionError):
                pass

        if flip_vertical:
            images = images[:, ::-1, :]
        if flip_horizontal:
            images = images[:, :, ::-1]
        return images, fs_ir

    def load_ir_images_for_case(
        self,
        case_dict: Dict[str, Any],
        fs_ir_fallback: float = 2000.0,
        flip_vertical: bool = True,
        flip_horizontal: bool = False,
    ) -> Tuple[np.ndarray, float]:
        npy_path = case_dict.get("npy_path")
        hcc_path = case_dict.get("hcc_path")

        if npy_path is not None and self._resolve_path(npy_path).exists():
            images = np.asarray(np.load(self._resolve_path(npy_path)), dtype=float)
            fs_ir = float(case_dict.get("fs_ir", fs_ir_fallback))
        elif hcc_path is not None:
            images, fs_ir = self._load_hcc_images(
                self._resolve_path(hcc_path),
                fs_ir_fallback=fs_ir_fallback,
                flip_vertical=False,
                flip_horizontal=False,
            )
            fs_ir = float(case_dict.get("fs_ir", fs_ir))
        else:
            raise ValueError("Case must contain existing npy_path or hcc_path.")

        if flip_vertical:
            images = images[:, ::-1, :]
        if flip_horizontal:
            images = images[:, :, ::-1]
        return images, fs_ir

    def _resolve_case_path(self, case_dict: Dict[str, Any], key: str) -> Path:
        if key not in case_dict:
            raise KeyError(f"Case missing key '{key}'")
        return self._resolve_path(case_dict[key])

    @staticmethod
    def _camera_to_stress_factor_pa_per_k() -> float:
        al_alpha_1_per_k = 23.0e-6
        al_rho_kg_per_m3 = 2700.0
        al_cp_j_per_kg_k = 900.0
        t_ref_k = 293.15
        thermoelastic_coeff = -(al_alpha_1_per_k * t_ref_k) / (al_rho_kg_per_m3 * al_cp_j_per_kg_k)
        return 1.0 / thermoelastic_coeff

    @staticmethod
    def _default_force_points(plate_lx: float, plate_ly: float) -> Dict[str, Tuple[float, float]]:
        x_l, x_c, x_r = 0.01, plate_lx / 2.0, plate_lx - 0.01
        y_mid, y_top = plate_ly - 0.08, plate_ly - 0.028
        return {
            "center_middle": (x_c, y_mid),
            "center_top": (x_c, y_top),
            "left_top": (x_l, y_top),
            "left_middle": (x_l, y_mid),
            "right_middle": (x_r, y_mid),
            "right_top": (x_r, y_top),
        }

    @staticmethod
    def _build_df_imp(case_labels: Iterable[str], force_points: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        rows = []
        for label in case_labels:
            if label not in force_points:
                raise ValueError(f"Case label '{label}' not found in force_points")
            x, y = force_points[label]
            rows.append([x, y, 0.0, 0.0, 0.0, 1.0, str(label)])
        return pd.DataFrame(rows, columns=[
            "Position_1", "Position_2", "Position_3", "Direction_1", "Direction_2", "Direction_3", "Case"
        ])

    @staticmethod
    def _build_df_node_coords(node_coords: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        node_coords = np.asarray(node_coords, dtype=float)
        node_coords_repeated = np.repeat(node_coords, 4, axis=0)
        df = pd.DataFrame(node_coords_repeated, columns=["Position_1", "Position_2", "Position_3"])
        direction_pattern = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ])
        df[["Direction_1", "Direction_2", "Direction_3"]] = np.tile(direction_pattern, (node_coords.shape[0], 1))
        return df, node_coords_repeated

    def _run_solver_case(
        self,
        solver_script: Path,
        run_root: Path,
        case_label: str,
        freq_file: Path,
        nx: int,
        ny: int,
        plate_lx: float,
        plate_ly: float,
        force_xy: Tuple[float, float],
    ) -> Path:
        x, y = force_xy
        run_dir = run_root / f"run_{case_label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.pop("FORCE_NODE_ID", None)
        env.pop("FORCE_X", None)
        env.pop("FORCE_Y", None)

        env["GRID_NX"] = str(int(nx))
        env["GRID_NY"] = str(int(ny))
        env["PLATE_LX"] = f"{float(plate_lx):.12g}"
        env["PLATE_LY"] = f"{float(plate_ly):.12g}"
        env["FREQS_FILE"] = str(freq_file)
        env["FORCE_X"] = f"{float(x):.12g}"
        env["FORCE_Y"] = f"{float(y):.12g}"

        subprocess.run([sys.executable, str(solver_script)], cwd=str(run_dir), env=env, check=True)
        return run_dir

    def _load_pyfbs(self):
        try:
            from pyFBSmaster import pyFBS
        except Exception as exc:
            raise ImportError("Failed to import pyFBS from pyFBSmaster") from exc
        return pyFBS

    def _maybe_sync_crop_stage1(
        self,
        force_raw: np.ndarray,
        fs_force: float,
        images_full: np.ndarray,
        fs_ir: float,
        cfg: Stage1Config,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
        if cfg.enable_impact_sync_crop:
            return self.sync_crop_and_window_force_camera(
                force_raw,
                fs_force,
                images_full,
                fs_ir,
                pre_ms=cfg.impact_pre_ms,
                noise_ref_s=cfg.impact_noise_ref_s,
                start_rms_mult=cfg.impact_start_rms_mult,
                start_rel_peak=cfg.impact_start_rel_peak,
                end_rms_mult=cfg.impact_end_rms_mult,
                end_rel_peak=cfg.impact_end_rel_peak,
                quiet_hold_ms=cfg.impact_quiet_hold_ms,
                force_zero_post_ms=cfg.force_zero_post_ms,
                force_end_taper_ms=cfg.force_end_taper_ms,
            )

        info = {
            "t_crop_s": 0.0,
            "t_start_s": 0.0,
            "t_end_s": 0.0,
            "t_zero_apply_s": 0.0,
            "n_force_crop": int(len(force_raw)),
            "n_ir_crop": int(images_full.shape[0]),
        }
        return np.asarray(force_raw), np.asarray(images_full), info, np.asarray(force_raw)

    def _build_overlay_interface(
        self,
        y_parent: np.ndarray,
        y_camera: np.ndarray,
        node_coords: np.ndarray,
        camera_positions: np.ndarray,
        df_node_coords: pd.DataFrame,
        component_overlay: int,
    ) -> Dict[str, Any]:
        y_parent = np.asarray(y_parent)
        y_camera = np.asarray(y_camera)

        if y_parent.shape[0] != y_camera.shape[0]:
            raise ValueError(
                f"Frequency-line mismatch between parent and camera overlay: "
                f"{y_parent.shape[0]} vs {y_camera.shape[0]}"
            )

        nfreq = int(y_camera.shape[0])
        nimp = int(y_camera.shape[2])
        if camera_positions.shape[0] != y_camera.shape[1]:
            raise ValueError("camera_positions length does not match Y_camera channels")

        d2 = np.sum((camera_positions[:, None, :] - node_coords[None, :, :]) ** 2, axis=2)
        cam_to_num_idx = np.argmin(d2, axis=1)
        map_dist = np.sqrt(np.min(d2, axis=1))

        b_nodes = np.unique(cam_to_num_idx)
        b_dof_idx = 4 * b_nodes + int(component_overlay)

        y_overlay_num = np.zeros((nfreq, len(b_dof_idx), nimp), dtype=y_parent.dtype)
        for j_b, n_idx in enumerate(b_nodes):
            cam_sel = np.where(cam_to_num_idx == n_idx)[0]
            y_overlay_num[:, j_b, :] = np.mean(y_camera[:, cam_sel, :], axis=1)

        df_overlay = df_node_coords.iloc[b_dof_idx].copy().reset_index(drop=True)
        df_overlay[["Direction_1", "Direction_2", "Direction_3"]] = [0.5, 0.5, 0.0]

        return {
            "Y_overlay_num": y_overlay_num,
            "df_overlay": df_overlay,
            "cam_to_num_idx": cam_to_num_idx,
            "map_dist_mean": float(np.mean(map_dist)),
            "map_dist_max": float(np.max(map_dist)),
            "b_nodes": b_nodes,
            "b_dof_idx": b_dof_idx,
        }

    def build_hammer_stage1(
        self,
        stage1_config: Optional[Stage1Config] = None,
        cases: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        cfg = stage1_config or self.stage1_cfg
        case_dict = cases or self.paths.stage1_cases
        case_labels = list(case_dict.keys())
        if not case_labels:
            raise ValueError("No stage-1 cases provided")

        first_case = case_dict[case_labels[0]]
        force_raw, fs_force, _, _ = self.load_force_from_pkl(
            self._resolve_case_path(first_case, "pkl_path"),
            task_name=cfg.force_task_name,
            force_name=cfg.force_channel_name,
        )
        images_full, fs_ir = self.load_ir_images_for_case(
            first_case,
            fs_ir_fallback=cfg.fs_ir_fallback,
            flip_vertical=cfg.flip_vertical,
            flip_horizontal=cfg.flip_horizontal,
        )

        force_proc0, images_sync0, sync_case0, _ = self._maybe_sync_crop_stage1(force_raw, fs_force, images_full, fs_ir, cfg)
        images_roi0 = self.crop_video(images_sync0, cfg.crop)
        freq_cam0, ir_frf0, _ = self.frf_ir_h1(force_proc0, fs_force, images_roi0, fs_ir, cfg.n_segments, cfg.exp_window_residual)
        ir_frf0 = self.reduce_resolution(ir_frf0, cfg.reduce_factor, mode="crop")

        mask_band = (freq_cam0 >= cfg.f_min) & (freq_cam0 <= cfg.f_max)
        freq_target = np.asarray(freq_cam0[mask_band], dtype=float)
        if freq_target.size == 0:
            raise ValueError(
                f"No camera frequencies in [{cfg.f_min}, {cfg.f_max}] Hz. Available [{freq_cam0.min():.3f}, {freq_cam0.max():.3f}] Hz"
            )

        rows, cols = ir_frf0.shape[1:]
        n_pix = rows * cols
        y_cam_all = np.zeros((len(freq_target), n_pix, len(case_labels)), dtype=np.complex128)
        sync_info = {}

        for k_case, label in enumerate(case_labels):
            if k_case == 0:
                # Reuse the already-computed first case result
                freq_cam = freq_cam0
                ir_frf = ir_frf0
                sync_case = sync_case0
            else:
                c = case_dict[label]
                force_raw, fs_force, _, _ = self.load_force_from_pkl(
                    self._resolve_case_path(c, "pkl_path"),
                    task_name=cfg.force_task_name,
                    force_name=cfg.force_channel_name,
                )
                images_full, fs_ir = self.load_ir_images_for_case(
                    c,
                    fs_ir_fallback=cfg.fs_ir_fallback,
                    flip_vertical=cfg.flip_vertical,
                    flip_horizontal=cfg.flip_horizontal,
                )
                force_proc, images_sync, sync_case, _ = self._maybe_sync_crop_stage1(force_raw, fs_force, images_full, fs_ir, cfg)
                images_roi = self.crop_video(images_sync, cfg.crop)
                freq_cam, ir_frf, _ = self.frf_ir_h1(force_proc, fs_force, images_roi, fs_ir, cfg.n_segments, cfg.exp_window_residual)
                ir_frf = self.reduce_resolution(ir_frf, cfg.reduce_factor, mode="crop")
            if ir_frf.shape[1:] != (rows, cols):
                raise ValueError(f"Spatial mismatch for case '{label}': {ir_frf.shape[1:]} vs {(rows, cols)}")

            h_flat = ir_frf.reshape(len(freq_cam), n_pix)
            y_cam_all[:, :, k_case] = self.interp_complex_to_target(freq_cam, h_flat, freq_target)
            sync_info[label] = sync_case

        out = {
            "cases": case_dict,
            "case_labels": case_labels,
            "freq_target": freq_target,
            "Y_cam_all": y_cam_all,
            "rows": int(rows),
            "cols": int(cols),
            "sync_info": sync_info,
            "stage1_crop": cfg.crop,
        }
        self.state["stage1"].update(out)
        return out
    def _run_chunked_semm(
        self,
        y_parent: np.ndarray,
        y_overlay: np.ndarray,
        df_parent: pd.DataFrame,
        df_imp_parent: pd.DataFrame,
        df_overlay: pd.DataFrame,
        df_imp_overlay: pd.DataFrame,
        freq_axis: np.ndarray,
        scfg: SEMMConfig,
        label: str,
    ) -> np.ndarray:
        pyfbs = self._load_pyfbs()
        y_parent = np.asarray(y_parent)
        y_overlay = np.asarray(y_overlay)
        freq_axis = np.asarray(freq_axis, dtype=float).reshape(-1)

        if y_parent.shape[0] != y_overlay.shape[0]:
            raise ValueError(f"Frequency-line mismatch: parent={y_parent.shape[0]}, overlay={y_overlay.shape[0]}")
        if len(freq_axis) != y_parent.shape[0]:
            raise ValueError(f"Frequency axis length {len(freq_axis)} does not match tensors {y_parent.shape[0]}")

        mask = (freq_axis >= float(scfg.f_semm_min)) & (freq_axis <= float(scfg.f_semm_max))
        sel = np.where(mask)[0]
        if sel.size == 0:
            raise ValueError(
                f"No frequencies in band [{scfg.f_semm_min}, {scfg.f_semm_max}] Hz. "
                f"Available [{freq_axis.min():.3f}, {freq_axis.max():.3f}] Hz"
            )

        n_per = int(scfg.semm_lines_per_chunk)
        if n_per < 1:
            raise ValueError("semm_lines_per_chunk must be >= 1")

        y_out = np.zeros_like(y_parent)
        n_total = int(sel.size)
        n_chunks = int(np.ceil(n_total / n_per))

        print("=" * 88)
        print(label)
        print(f"Start             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Band              : {freq_axis[sel[0]]:.3f} .. {freq_axis[sel[-1]]:.3f} Hz")
        print(f"Selected lines    : {n_total}")
        print(f"Chunk size        : {n_per}")
        print(f"Total chunks      : {n_chunks}")
        print(f"SEMM type         : {scfg.semm_type}")
        print("=" * 88)

        pbar = None
        if _HAS_TQDM:
            pbar = tqdm(total=n_total, desc=label, unit="line", dynamic_ncols=True, leave=True)

        t0 = time.perf_counter()
        for chunk_no, start in enumerate(range(0, n_total, n_per), start=1):
            idx = sel[start:start + n_per]
            f0 = float(freq_axis[idx[0]])
            f1 = float(freq_axis[idx[-1]])

            t_chunk = time.perf_counter()
            y_chunk = pyfbs.SEMM(
                y_parent[idx],
                y_overlay[idx],
                df_parent,
                df_imp_parent,
                df_overlay,
                df_imp_overlay,
                SEMM_type=scfg.semm_type,
            )
            y_out[idx] = y_chunk

            done = start + len(idx)
            frac = done / max(n_total, 1)
            elapsed = time.perf_counter() - t0
            chunk_dt = time.perf_counter() - t_chunk
            eta = (elapsed / frac - elapsed) if frac > 0 else np.nan

            if pbar is not None:
                pbar.update(len(idx))
                pbar.set_postfix(
                    {
                        "chunk": f"{chunk_no}/{n_chunks}",
                        "f_Hz": f"{f0:.1f}-{f1:.1f}",
                        "eta_s": f"{eta:.1f}",
                    },
                    refresh=False,
                )

            if chunk_no == 1 or chunk_no == n_chunks or (chunk_no % int(max(1, scfg.log_every_n_chunks)) == 0):
                print(
                    f"[{chunk_no:>4}/{n_chunks}] "
                    f"lines {done-len(idx)+1:>4}-{done:>4}/{n_total} | "
                    f"f={f0:8.3f}..{f1:8.3f} Hz | chunk={chunk_dt:6.2f}s | "
                    f"elapsed={elapsed:7.1f}s | eta={eta:7.1f}s"
                )

        if pbar is not None:
            pbar.close()

        dt = time.perf_counter() - t0
        print("-" * 88)
        print(f"Completed         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total runtime     : {dt:.2f} s")
        print(f"Output shape      : {y_out.shape}")
        print("=" * 88)
        return y_out

    def run_stage1_numerical_and_semm(
        self,
        stage1_data: Optional[Dict[str, Any]] = None,
        semm_config: Optional[SEMMConfig] = None,
    ) -> Dict[str, Any]:
        data = stage1_data or self.state.get("stage1", {})
        cfg = self.stage1_cfg
        scfg = semm_config or self.semm_cfg

        required = ["freq_target", "Y_cam_all", "rows", "cols", "case_labels"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing stage-1 data keys: {missing}. Run build_hammer_stage1 first.")

        freq_target = np.asarray(data["freq_target"], dtype=float)
        y_cam_all = np.asarray(data["Y_cam_all"])
        rows = int(data["rows"])
        cols = int(data["cols"])
        case_labels = list(data["case_labels"])

        solver_script = self._resolve_path(self.paths.solver_script)
        if not solver_script.exists():
            raise FileNotFoundError(f"Solver script not found: {solver_script}")

        stage1_run_root = self._resolve_path(self.paths.stage1_run_root)
        stage1_run_root.mkdir(parents=True, exist_ok=True)

        freqs_file = stage1_run_root / "camera_freq_target.npy"
        np.save(freqs_file, freq_target)

        force_points = cfg.force_points or self._default_force_points(cfg.plate_lx, cfg.plate_ly)
        for label in case_labels:
            if label not in force_points:
                raise ValueError(
                    f"Case '{label}' not in force_points. Provide Stage1Config.force_points with matching labels."
                )

        run_dirs: Dict[str, Path] = {}
        for label in case_labels:
            x, y = force_points[label]
            run_dirs[label] = self._run_solver_case(
                solver_script=solver_script,
                run_root=stage1_run_root,
                case_label=label,
                freq_file=freqs_file,
                nx=cols,
                ny=rows,
                plate_lx=cfg.plate_lx,
                plate_ly=cfg.plate_ly,
                force_xy=(float(x), float(y)),
            )

        ref_dir = run_dirs[case_labels[0]]
        freqs = np.asarray(np.load(ref_dir / "freqs.npy"), dtype=float)
        node_coords = np.asarray(np.load(ref_dir / "node_coords.npy"), dtype=float)

        nfreq = len(freqs)
        nnodes = node_coords.shape[0]
        y_all = np.zeros((nfreq, nnodes * 4, len(case_labels)), dtype=np.complex128)
        for k, label in enumerate(case_labels):
            stress_tensor = np.asarray(np.load(run_dirs[label] / "stress_tensor_frf.npy"), dtype=np.complex128)
            if stress_tensor.shape != (nfreq, nnodes, 4):
                raise ValueError(
                    f"Unexpected stress tensor shape for case '{label}': {stress_tensor.shape}, expected {(nfreq, nnodes, 4)}"
                )
            y_all[:, :, k] = stress_tensor.reshape(nfreq, nnodes * 4)

        cam_to_stress = self._camera_to_stress_factor_pa_per_k()
        component_overlay = 3

        if cfg.use_all_cases:
            y_final2 = y_all.copy()
            y_camera = y_cam_all.copy()
            df_imp_use = self._build_df_imp(case_labels, force_points)
        else:
            i_case = int(np.clip(cfg.case_idx, 0, y_all.shape[2] - 1))
            y_final2 = y_all[:, :, i_case:i_case + 1].copy()
            y_camera = y_cam_all[:, :, i_case:i_case + 1].copy()
            df_imp_use = self._build_df_imp([case_labels[i_case]], force_points)

        if y_camera.shape[0] != y_final2.shape[0]:
            y_cam_res = np.zeros((len(freqs), y_camera.shape[1], y_camera.shape[2]), dtype=np.complex128)
            for i in range(y_camera.shape[2]):
                y_cam_res[:, :, i] = self.interp_complex_to_target(freq_target, y_camera[:, :, i], freqs)
            y_camera = y_cam_res

        y_camera = cam_to_stress * y_camera

        df_node_coords, node_coords_repeated = self._build_df_node_coords(node_coords)
        df_camera, camera_positions, diag_cam = self.build_camera_df_from_grid(
            node_coords,
            rows=rows,
            cols=cols,
            flip_x=cfg.camera_flip_x,
            flip_y=cfg.camera_flip_y,
        )

        overlay_pack = self._build_overlay_interface(
            y_parent=y_final2,
            y_camera=y_camera,
            node_coords=node_coords,
            camera_positions=camera_positions,
            df_node_coords=df_node_coords,
            component_overlay=component_overlay,
        )

        y_semm = self._run_chunked_semm(
            y_parent=y_final2,
            y_overlay=overlay_pack["Y_overlay_num"],
            df_parent=df_node_coords,
            df_imp_parent=df_imp_use,
            df_overlay=overlay_pack["df_overlay"],
            df_imp_overlay=df_imp_use,
            freq_axis=freqs,
            scfg=scfg,
            label="Stage-1 SEMM",
        )

        out = {
            "freqs": freqs,
            "node_coords": node_coords,
            "node_coords_repeated": node_coords_repeated,
            "Y_all": y_all,
            "Y_final2": y_final2,
            "Y_camera": y_camera,
            "Y_SEMM": y_semm,
            "df_node_coords": df_node_coords,
            "df_imp_use": df_imp_use,
            "df_camera": df_camera,
            "camera_positions": camera_positions,
            "diag_cam": diag_cam,
            "component_overlay": component_overlay,
            "cam_to_stress_factor": cam_to_stress,
            "run_dirs": {k: str(v) for k, v in run_dirs.items()},
            "mapping_metadata": {
                "stage": "stage1",
                "component_overlay": int(component_overlay),
                "rows": int(rows),
                "cols": int(cols),
                "case_labels": list(df_imp_use["Case"].astype(str).tolist()),
                "diag_cam": diag_cam,
                "mean_cam_to_num_dist": float(overlay_pack["map_dist_mean"]),
                "max_cam_to_num_dist": float(overlay_pack["map_dist_max"]),
                "n_interface_dofs": int(len(overlay_pack["b_dof_idx"])),
            },
            "b_nodes": overlay_pack["b_nodes"],
            "b_dof_idx": overlay_pack["b_dof_idx"],
            "Y_overlay_num": overlay_pack["Y_overlay_num"],
            "df_overlay": overlay_pack["df_overlay"],
            "cam_to_num_idx": overlay_pack["cam_to_num_idx"],
            "stage1_crop": data.get("stage1_crop", cfg.crop),
        }
        self.state["stage1"].update(out)
        return out
    def load_base_measurement(self, stage2_config: Optional[Stage2Config] = None) -> Dict[str, Any]:
        cfg = stage2_config or self.stage2_cfg

        camera_npy = self._resolve_path(self.paths.stage2_camera_npy) if self.paths.stage2_camera_npy else None
        camera_hcc = self._resolve_path(self.paths.stage2_camera_hcc)
        camera_cache_saved = False

        if camera_npy is not None and camera_npy.exists():
            images = np.asarray(np.load(camera_npy), dtype=float)
            fs_ir = float(cfg.fs_ir_fallback)
            camera_source = str(camera_npy)
        else:
            images, fs_ir = self._load_hcc_images(
                camera_hcc,
                fs_ir_fallback=cfg.fs_ir_fallback,
                flip_vertical=cfg.flip_vertical,
                flip_horizontal=cfg.flip_horizontal,
            )
            camera_source = str(camera_hcc)
            if camera_npy is not None and bool(cfg.cache_stage2_camera_npy):
                camera_npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(camera_npy, np.asarray(images, dtype=float))
                camera_cache_saved = True

        acc, fs_acc, acc_time, acc_meta = self.load_signal_from_pkl(
            self._resolve_path(self.paths.stage2_accel_pkl),
            task_name=cfg.acc_task_name,
            channel_name=cfg.acc_channel_name,
            remove_dc=cfg.remove_dc,
        )

        scale = 1.0
        if cfg.acc_in_g:
            scale = float(cfg.acc_g0)
            acc = acc * scale

        out = {
            "images": images,
            "fs_ir": float(fs_ir),
            "camera_source": camera_source,
            "acc_ref": acc,
            "fs_acc": float(fs_acc),
            "acc_time": acc_time,
            "acc_meta": acc_meta,
            "acc_scale_applied": float(scale),
            "acc_units": "m/s^2",
            "camera_cache_path": str(camera_npy) if camera_npy is not None else None,
            "camera_cache_saved": bool(camera_cache_saved),
        }
        self.state["stage2"].update(out)
        return out

    def cache_stage2_camera_to_npy(
        self,
        overwrite: bool = False,
        stage2_config: Optional[Stage2Config] = None,
    ) -> Path:
        cfg = stage2_config or self.stage2_cfg
        if not self.paths.stage2_camera_npy:
            raise ValueError("PathsConfig.stage2_camera_npy is required for caching stage-2 camera data.")

        npy_path = self._resolve_path(self.paths.stage2_camera_npy)
        hcc_path = self._resolve_path(self.paths.stage2_camera_hcc)
        if npy_path.exists() and not overwrite:
            return npy_path

        images, _ = self._load_hcc_images(
            hcc_path,
            fs_ir_fallback=cfg.fs_ir_fallback,
            flip_vertical=cfg.flip_vertical,
            flip_horizontal=cfg.flip_horizontal,
        )
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, np.asarray(images, dtype=float))
        return npy_path

    def resolve_stage2_crop(
        self,
        stage1_crop: Optional[Dict[str, int]] = None,
        stage2_crop: Optional[Dict[str, int]] = None,
        mode: str = "inherit_stage1",
    ) -> Dict[str, int]:
        crop = stage1_crop or self.stage1_cfg.crop
        if crop is None:
            raise ValueError("No crop set. Use pick_crop() or load_crop().")
        return self._normalize_crop(crop)

    def build_base_overlay_stress_acc(
        self,
        base_data: Optional[Dict[str, Any]] = None,
        stage1_data: Optional[Dict[str, Any]] = None,
        stage2_config: Optional[Stage2Config] = None,
    ) -> Dict[str, Any]:
        cfg2 = stage2_config or self.stage2_cfg
        st2 = base_data or self.state.get("stage2", {})
        st1 = stage1_data or self.state.get("stage1", {})

        req2 = ["images", "fs_ir", "acc_ref", "fs_acc"]
        req1 = ["freqs", "node_coords", "df_node_coords", "stage1_crop", "Y_SEMM"]
        miss2 = [k for k in req2 if k not in st2]
        miss1 = [k for k in req1 if k not in st1]
        if miss2 or miss1:
            raise ValueError(f"Missing inputs. stage2 missing={miss2}, stage1 missing={miss1}")

        crop_used = self.resolve_stage2_crop(stage1_crop=st1.get("stage1_crop"))

        images_roi = self.crop_video(np.asarray(st2["images"]), crop_used)
        reduce_factor = cfg2.reduce_factor if cfg2.reduce_factor is not None else self.stage1_cfg.reduce_factor
        freq_base, h_base, _ = self.frf_ir_h1(
            np.asarray(st2["acc_ref"]),
            float(st2["fs_acc"]),
            images_roi,
            float(st2["fs_ir"]),
            int(cfg2.n_segments),
        )
        h_base = self.reduce_resolution(h_base, reduce_factor, mode="crop")

        rows2, cols2 = h_base.shape[1:]
        n_pix2 = int(rows2 * cols2)

        h_flat = h_base.reshape(len(freq_base), n_pix2)
        freq_stage1 = np.asarray(st1["freqs"], dtype=float)
        h_tgt = self.interp_complex_to_target(freq_base, h_flat, freq_stage1)

        cam_to_stress = float(st1.get("cam_to_stress_factor", self._camera_to_stress_factor_pa_per_k()))
        y_overlay_base = (cam_to_stress * h_tgt)[:, :, None]

        df_camera2, camera_positions2, diag_cam2 = self.build_camera_df_from_grid(
            np.asarray(st1["node_coords"]),
            rows=rows2,
            cols=cols2,
            flip_x=self.stage1_cfg.camera_flip_x,
            flip_y=self.stage1_cfg.camera_flip_y,
        )

        overlay_pack = self._build_overlay_interface(
            y_parent=np.asarray(st1["Y_SEMM"]),
            y_camera=y_overlay_base,
            node_coords=np.asarray(st1["node_coords"]),
            camera_positions=camera_positions2,
            df_node_coords=st1["df_node_coords"],
            component_overlay=int(cfg2.overlay_component),
        )

        out = {
            "Y_overlay_base": y_overlay_base,
            "Y_overlay_base_num": overlay_pack["Y_overlay_num"],
            "df_overlay_base": overlay_pack["df_overlay"],
            "df_camera_stage2": df_camera2,
            "camera_positions_stage2": camera_positions2,
            "diag_cam_stage2": diag_cam2,
            "b_nodes_stage2": overlay_pack["b_nodes"],
            "b_dof_idx_stage2": overlay_pack["b_dof_idx"],
            "cam_to_num_idx_stage2": overlay_pack["cam_to_num_idx"],
            "stage2_crop_used": crop_used,
        }
        self.state["stage2"].update(out)
        return out

    def identify_equivalent_inputs(
        self,
        stage1_data: Optional[Dict[str, Any]] = None,
        stage2_overlay: Optional[Dict[str, Any]] = None,
        stage2_config: Optional[Stage2Config] = None,
    ) -> np.ndarray:
        cfg2 = stage2_config or self.stage2_cfg
        st1 = stage1_data or self.state.get("stage1", {})
        st2 = stage2_overlay or self.state.get("stage2", {})

        req1 = ["Y_SEMM", "Y_final2", "df_imp_use"]
        req2 = ["Y_overlay_base_num", "b_dof_idx_stage2"]
        miss1 = [k for k in req1 if k not in st1]
        miss2 = [k for k in req2 if k not in st2]
        if miss1 or miss2:
            raise ValueError(f"Missing inputs for equivalent input solve. stage1={miss1}, stage2={miss2}")

        basis_name = str(cfg2.parent_basis).strip()
        if basis_name not in ("Y_SEMM", "Y_final2"):
            raise ValueError("Stage2Config.parent_basis must be 'Y_SEMM' or 'Y_final2'")

        y_basis = np.asarray(st1[basis_name])
        y_overlay_i = np.asarray(st2["Y_overlay_base_num"])
        b_dof = np.asarray(st2["b_dof_idx_stage2"], dtype=int)
        y_basis_i = y_basis[:, b_dof, :]

        n_freq, _, n_inputs = y_basis_i.shape
        q = np.zeros((n_freq, n_inputs), dtype=np.complex128)
        residual = np.full(n_freq, np.nan, dtype=float)
        residual_all = np.full(n_freq, np.nan, dtype=float)
        cond = np.full(n_freq, np.nan, dtype=float)

        lam = float(cfg2.equiv_regularization)
        eye = np.eye(n_inputs, dtype=np.complex128)
        fit_step = max(1, int(cfg2.equiv_fit_every_n_dof))
        fit_mask = np.zeros(len(b_dof), dtype=bool)
        fit_mask[::fit_step] = True

        for k in range(n_freq):
            b = y_basis_i[k, :, :]
            y = y_overlay_i[k, :, 0]

            valid_all = np.isfinite(y)
            valid_all &= np.all(np.isfinite(b), axis=1)
            if np.count_nonzero(valid_all) < n_inputs:
                continue

            valid_fit = valid_all & fit_mask
            if np.count_nonzero(valid_fit) < n_inputs:
                # Fallback to all valid rows if the selected subset is too small.
                valid_fit = valid_all

            b_v = b[valid_fit, :]
            y_v = y[valid_fit]

            try:
                q[k, :] = np.linalg.solve(b_v.conj().T @ b_v + lam * eye, b_v.conj().T @ y_v)
            except np.linalg.LinAlgError:
                q[k, :] = np.linalg.lstsq(b_v, y_v, rcond=None)[0]

            fit = b_v @ q[k, :]
            residual[k] = float(np.linalg.norm(fit - y_v) / (np.linalg.norm(y_v) + 1e-30))
            fit_all = b[valid_all, :] @ q[k, :]
            y_all = y[valid_all]
            residual_all[k] = float(np.linalg.norm(fit_all - y_all) / (np.linalg.norm(y_all) + 1e-30))
            try:
                cond[k] = float(np.linalg.cond(b_v))
            except np.linalg.LinAlgError:
                cond[k] = np.inf

        self.state["stage2"]["q_equiv"] = q
        self.state["stage2"]["fit_residual"] = residual
        self.state["stage2"]["fit_residual_all_dofs"] = residual_all
        self.state["stage2"]["fit_condition"] = cond
        self.state["stage2"]["fit_dof_step"] = int(fit_step)
        self.state["stage2"]["fit_dof_count_total"] = int(len(b_dof))
        self.state["stage2"]["fit_dof_count_used"] = int(np.count_nonzero(fit_mask))
        self.state["stage2"]["q_equiv_source"] = "overlay_fit"
        return q

    def identify_equivalent_inputs_virtual_base(
        self,
        stage1_data: Optional[Dict[str, Any]] = None,
        stage2_config: Optional[Stage2Config] = None,
    ) -> np.ndarray:
        cfg2 = stage2_config or self.stage2_cfg
        st1 = stage1_data or self.state.get("stage1", {})

        req = ["freqs", "df_imp_use", "node_coords", "Y_SEMM", "Y_final2"]
        miss = [k for k in req if k not in st1]
        if miss:
            raise ValueError(f"Missing stage-1 data for virtual-base q solve: {miss}")

        basis_name = str(cfg2.parent_basis).strip()
        if basis_name not in ("Y_SEMM", "Y_final2"):
            raise ValueError("Stage2Config.parent_basis must be 'Y_SEMM' or 'Y_final2'")
        y_basis = np.asarray(st1[basis_name])
        n_freq, _, n_inputs = y_basis.shape

        df_imp = st1["df_imp_use"].reset_index(drop=True)
        if len(df_imp) != n_inputs:
            raise ValueError(
                f"Input metadata mismatch: len(df_imp_use)={len(df_imp)} vs n_inputs={n_inputs}"
            )

        x_in = np.asarray(df_imp["Position_1"], dtype=float)
        y_in = np.asarray(df_imp["Position_2"], dtype=float)
        labels = [str(v) for v in df_imp["Case"].tolist()]

        node_coords = np.asarray(st1["node_coords"], dtype=float)
        x_mid = float(0.5 * (np.min(node_coords[:, 0]) + np.max(node_coords[:, 0])))
        y_bottom = float(np.min(node_coords[:, 1]))

        explicit = cfg2.virtual_base_input_weights or {}
        if explicit:
            w = np.zeros(n_inputs, dtype=float)
            for i, lab in enumerate(labels):
                w[i] = float(explicit.get(lab, 0.0))
        else:
            d = np.sqrt((x_in - x_mid) ** 2 + (y_in - y_bottom) ** 2)
            sigma = cfg2.virtual_base_sigma_m
            if sigma is None:
                lx = float(np.max(node_coords[:, 0]) - np.min(node_coords[:, 0]))
                ly = float(np.max(node_coords[:, 1]) - np.min(node_coords[:, 1]))
                sigma = 0.25 * max(lx, ly)
            sigma = max(float(sigma), 1e-9)
            w = np.exp(-0.5 * (d / sigma) ** 2)

        w = np.maximum(w, 0.0)
        sw = float(np.sum(w))
        if not np.isfinite(sw) or sw <= 0.0:
            raise ValueError("Virtual-base input weights collapsed to zero. Check virtual_base_input_weights/sigma.")
        w = w / sw

        mass_eq = float(cfg2.virtual_base_mass_kg)
        q0 = mass_eq * w
        q = np.tile(q0[None, :], (n_freq, 1)).astype(np.complex128)

        self.state["stage2"]["q_equiv"] = q
        self.state["stage2"]["fit_residual"] = np.full(n_freq, np.nan, dtype=float)
        self.state["stage2"]["fit_residual_all_dofs"] = np.full(n_freq, np.nan, dtype=float)
        self.state["stage2"]["fit_condition"] = np.full(n_freq, np.nan, dtype=float)
        self.state["stage2"]["fit_dof_step"] = None
        self.state["stage2"]["fit_dof_count_total"] = None
        self.state["stage2"]["fit_dof_count_used"] = None
        self.state["stage2"]["q_equiv_source"] = "virtual_base_stage1_only"
        self.state["stage2"]["virtual_base_reference"] = {
            "x_mid": x_mid,
            "y_bottom": y_bottom,
            "mass_eq_kg": mass_eq,
            "sigma_m": float(cfg2.virtual_base_sigma_m) if cfg2.virtual_base_sigma_m is not None else None,
            "input_weights_normalized": {labels[i]: float(w[i]) for i in range(n_inputs)},
        }
        return q

    def build_stage2_parent_acc(
        self,
        q_equiv: Optional[np.ndarray] = None,
        stage1_data: Optional[Dict[str, Any]] = None,
        stage2_overlay: Optional[Dict[str, Any]] = None,
        stage2_config: Optional[Stage2Config] = None,
    ) -> np.ndarray:
        cfg2 = stage2_config or self.stage2_cfg
        q = q_equiv

        if q is None:
            mode = str(cfg2.parent_q_mode).strip().lower()
            if mode == "virtual_base":
                q = self.identify_equivalent_inputs_virtual_base(
                    stage1_data=stage1_data,
                    stage2_config=cfg2,
                )
            elif mode == "fit_overlay":
                q = self.identify_equivalent_inputs(
                    stage1_data=stage1_data,
                    stage2_overlay=stage2_overlay,
                    stage2_config=cfg2,
                )
            else:
                raise ValueError("Stage2Config.parent_q_mode must be 'virtual_base' or 'fit_overlay'")

        return self.build_acceleration_parent_from_stage1(
            q_equiv=q,
            stage1_data=stage1_data,
            stage2_config=cfg2,
        )

    def build_acceleration_parent_from_stage1(
        self,
        q_equiv: Optional[np.ndarray] = None,
        stage1_data: Optional[Dict[str, Any]] = None,
        stage2_config: Optional[Stage2Config] = None,
    ) -> np.ndarray:
        cfg2 = stage2_config or self.stage2_cfg
        st1 = stage1_data or self.state.get("stage1", {})
        q = np.asarray(q_equiv if q_equiv is not None else self.state.get("stage2", {}).get("q_equiv"))

        if q.ndim != 2:
            raise ValueError("Equivalent inputs q_equiv are missing. Run identify_equivalent_inputs first.")

        basis_name = str(cfg2.parent_basis).strip()
        if basis_name not in ("Y_SEMM", "Y_final2"):
            raise ValueError("Stage2Config.parent_basis must be 'Y_SEMM' or 'Y_final2'")
        if basis_name not in st1:
            raise ValueError(f"Stage-1 basis '{basis_name}' not found")

        y_basis = np.asarray(st1[basis_name])
        if y_basis.shape[0] != q.shape[0] or y_basis.shape[2] != q.shape[1]:
            raise ValueError(f"Shape mismatch: basis={y_basis.shape}, q={q.shape}")

        y_parent_acc = np.einsum("fdi,fi->fd", y_basis, q)[:, :, None]
        self.state["stage2"]["Y_parent_acc"] = y_parent_acc
        return y_parent_acc

    def run_stage2_semm(
        self,
        parent_acc: Optional[np.ndarray] = None,
        stage2_overlay: Optional[Dict[str, Any]] = None,
        semm_config: Optional[SEMMConfig] = None,
    ) -> Dict[str, Any]:
        st1 = self.state.get("stage1", {})
        st2 = stage2_overlay or self.state.get("stage2", {})
        scfg = semm_config or self.semm_cfg

        y_parent = np.asarray(parent_acc if parent_acc is not None else st2.get("Y_parent_acc"))
        if y_parent.ndim != 3:
            raise ValueError("Stage-2 parent tensor missing. Run build_acceleration_parent_from_stage1 first.")

        req = ["Y_overlay_base_num", "df_overlay_base", "stage2_crop_used"]
        miss = [k for k in req if k not in st2]
        if miss:
            raise ValueError(f"Missing stage-2 overlay keys: {miss}")

        if "df_node_coords" not in st1 or "node_coords" not in st1 or "freqs" not in st1:
            raise ValueError("Missing stage-1 data required for stage-2 SEMM")

        node_coords = np.asarray(st1["node_coords"], dtype=float)
        x_mid = float(0.5 * (np.min(node_coords[:, 0]) + np.max(node_coords[:, 0])))
        y_bottom = float(np.min(node_coords[:, 1]))
        d1, d2, d3 = self.stage2_cfg.virtual_input_direction
        df_imp_base = pd.DataFrame([
            {
                "Position_1": x_mid,
                "Position_2": y_bottom,
                "Position_3": 0.0,
                "Direction_1": float(d1),
                "Direction_2": float(d2),
                "Direction_3": float(d3),
                "Case": self.stage2_cfg.virtual_input_case_name,
            }
        ])

        y_semm2 = self._run_chunked_semm(
            y_parent=y_parent,
            y_overlay=np.asarray(st2["Y_overlay_base_num"]),
            df_parent=st1["df_node_coords"],
            df_imp_parent=df_imp_base,
            df_overlay=st2["df_overlay_base"],
            df_imp_overlay=df_imp_base,
            freq_axis=np.asarray(st1["freqs"], dtype=float),
            scfg=scfg,
            label="Stage-2 SEMM",
        )

        out = {
            "Y_SEMM_stage2": y_semm2,
            "df_imp_base": df_imp_base,
            "stage2_crop_used": st2["stage2_crop_used"],
        }
        self.state["stage2"].update(out)
        return out

    def save_outputs(self, output_root: Optional[str] = None, run_name: Optional[str] = None) -> None:
        out_root = self._resolve_path(output_root or self.paths.output_root)
        run_id = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out_root / run_id
        stage1_dir = run_dir / "stage1"
        stage2_dir = run_dir / "stage2"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        stage2_dir.mkdir(parents=True, exist_ok=True)

        st1 = self.state.get("stage1", {})
        st2 = self.state.get("stage2", {})

        required = [
            (stage1_dir / "Y_SEMM_stage1.npy", st1, "Y_SEMM"),
            (stage1_dir / "freq_axis.npy", st1, "freqs"),
            (stage2_dir / "Y_parent_acc.npy", st2, "Y_parent_acc"),
            (stage2_dir / "Y_overlay_base.npy", st2, "Y_overlay_base"),
            (stage2_dir / "Y_SEMM_stage2.npy", st2, "Y_SEMM_stage2"),
            (stage2_dir / "q_equiv.npy", st2, "q_equiv"),
            (stage2_dir / "fit_residual.npy", st2, "fit_residual"),
        ]
        for p, d, k in required:
            if k not in d:
                raise ValueError(f"Missing key '{k}' required for saving outputs")
            np.save(p, np.asarray(d[k]))

        self._write_json(stage1_dir / "mapping_metadata.json", st1.get("mapping_metadata", {}))

        if "stage2_crop_used" in st2:
            self._write_json(stage2_dir / "stage2_crop_used.json", st2["stage2_crop_used"])

        run_meta = {
            "run_id": run_id,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "paths": asdict(self.paths),
            "stage1_config": asdict(self.stage1_cfg),
            "stage2_config": asdict(self.stage2_cfg),
            "semm_config": asdict(self.semm_cfg),
            "stage1_keys": sorted(list(st1.keys())),
            "stage2_keys": sorted(list(st2.keys())),
            "stage2_crop_mode": self.stage2_cfg.crop_mode,
            "stage2_crop_source": "manual" if self.stage2_cfg.crop_mode == "manual" else "inherit_stage1",
        }
        self._write_json(run_dir / "run_metadata.json", run_meta)
        self.state["meta"]["last_output_dir"] = str(run_dir)

    def save_stage1(self, output_root: Optional[str] = None, run_name: Optional[str] = None) -> Path:
        """Save Stage 1 results only (no Stage 2 required)."""
        out_root = self._resolve_path(output_root or self.paths.output_root)
        run_id = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out_root / run_id / "stage1"
        run_dir.mkdir(parents=True, exist_ok=True)

        st1 = self.state["stage1"]
        np.save(run_dir / "Y_SEMM_stage1.npy", st1["Y_SEMM"])
        np.save(run_dir / "freq_axis.npy", st1["freqs"])
        np.save(run_dir / "node_coords.npy", st1["node_coords"])
        np.save(run_dir / "Y_camera.npy", st1["Y_camera"])
        np.save(run_dir / "Y_cam_all.npy", st1["Y_cam_all"])
        np.save(run_dir / "Y_final2.npy", st1["Y_final2"])
        self._write_json(run_dir / "mapping_metadata.json", st1.get("mapping_metadata", {}))
        self._write_json(run_dir / "sync_info.json", st1.get("sync_info", {}))
        self._write_json(run_dir.parent / "run_metadata.json", {
            "run_id": run_id,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "paths": asdict(self.paths),
            "stage1_config": asdict(self.stage1_cfg),
            "stage1_keys": sorted(list(st1.keys())),
        })
        self.state["meta"]["last_stage1_dir"] = str(run_dir)
        return run_dir

    # -------------------- unified crop API --------------------
    def load_crop(self, path: Any = None) -> Dict[str, int]:
        """Load crop from JSON (default: paths.crop_json), set stage1_cfg.crop."""
        p = self._resolve_path(path or self.paths.crop_json)
        with p.open("r", encoding="utf-8") as f:
            crop = json.load(f)
        crop = self._normalize_crop(crop)
        self.stage1_cfg.crop = dict(crop)
        return crop

    def save_crop(self, crop: Optional[Dict[str, int]] = None, path: Any = None) -> Path:
        """Validate and write crop to JSON (default: paths.crop_json)."""
        crop = self._normalize_crop(crop or self.stage1_cfg.crop)
        self.stage1_cfg.crop = dict(crop)
        p = self._resolve_path(path or self.paths.crop_json)
        self._write_json(p, crop)
        return p

    def pick_crop(
        self,
        images: np.ndarray,
        frame_index: int = 0,
        save: bool = True,
        p_low: float = 1.0,
        p_high: float = 99.0,
    ) -> Dict[str, int]:
        """Interactive ginput crop picker. Sets stage1_cfg.crop, optionally saves to crop_json."""
        import matplotlib.pyplot as plt

        frame = np.asarray(images[frame_index])
        vmin, vmax = np.percentile(frame, [p_low, p_high])

        plt.figure(figsize=(8, 6))
        plt.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
        plt.title("Click TOP-LEFT then BOTTOM-RIGHT of ROI. Close the figure after the 2 clicks.")
        plt.xlabel("col (x)")
        plt.ylabel("row (y)")
        plt.tight_layout()

        pts = plt.ginput(2, timeout=0)
        plt.close()
        if len(pts) != 2:
            raise RuntimeError("Did not get 2 clicks. Try again.")

        (x1, y1), (x2, y2) = pts
        crop = {
            "col_start": max(0, int(np.floor(min(x1, x2)))),
            "col_end": int(np.ceil(max(x1, x2))),
            "row_start": max(0, int(np.floor(min(y1, y2)))),
            "row_end": int(np.ceil(max(y1, y2))),
        }
        crop = self._normalize_crop(crop)
        self.stage1_cfg.crop = dict(crop)
        if save:
            self.save_crop(crop)
        return crop

    # -------------------- legacy crop API (delegates) --------------------
    def pick_crop_two_clicks(
        self,
        images: np.ndarray,
        frame_index: int = 0,
        p_low: float = 1.0,
        p_high: float = 99.0,
    ) -> Dict[str, int]:
        """Legacy wrapper — delegates to pick_crop(save=False)."""
        crop = self.pick_crop(images, frame_index=frame_index, save=False,
                              p_low=p_low, p_high=p_high)
        self.stage2_cfg.stage2_crop = dict(crop)
        return crop

    def set_stage2_crop(self, crop_dict: dict) -> None:
        """Legacy wrapper — sets unified stage1_cfg.crop."""
        crop = self._normalize_crop(crop_dict)
        self.stage1_cfg.crop = dict(crop)
        self.stage2_cfg.stage2_crop = dict(crop)

    def save_stage2_crop(self, path: Any) -> None:
        """Legacy wrapper — delegates to save_crop."""
        self.save_crop(self.stage1_cfg.crop, path)

    def load_stage2_crop(self, path: Any) -> dict:
        """Legacy wrapper — delegates to load_crop."""
        crop = self.load_crop(path)
        self.stage2_cfg.stage2_crop = dict(crop)
        return crop
