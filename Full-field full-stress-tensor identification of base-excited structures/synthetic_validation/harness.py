"""End-to-end synthetic-validation harness.

Two entry points
----------------
run_case_from_modal(truth_md, prior_md, ...)
    Analytic path: modal data in, no MAPDL / SEMM required.
    Used by the oracle correctness gate.

run_case(truth_cfg, parent_cfg, ...)
    Full path: generate_fe → run_semm_stage1 → extract_mode_shapes.
    fe_models and semm_adapter are imported lazily so the analytic path
    (and its test) works without pyFBS / ANSYS installed.

Return dict (both functions)
-----------------------------
{
    "recovered":        {comp: (nfreq, nnodes)},   # reconstructed stress PSD
    "truth":            {comp: (nfreq, nnodes)},   # analytic ground truth PSD
    "metrics": {
        "nrmse":        {comp: float},
        "mac":          {comp: float},
    },
    "freqs":            (nfreq,),                  # Welch frequency axis [Hz]
    "condition_number": float,                     # cond(Psi_cam)
}
"""
from __future__ import annotations

import numpy as np
from scipy.signal import csd, welch

from synthetic_validation.forward_model import synthesize_base_excitation, truth_component_psd
from synthetic_validation.noise import inject_camera_noise
from synthetic_validation.expansion import (
    normalize_mode_shapes,
    modal_decompose,
    expand_components,
    stress_psd,
    virtual_reference_denoise,
    nnls_psd_decompose,
    expand_psd,
)
from synthetic_validation.metrics import nrmse, mac, condition_number, rel_rms_error, peak_ratio

# Physical conversion factor Pa/K.  Cancels in the oracle case (zero noise,
# perfect mode shapes) — value only affects real-data absolute scaling.
_CAMERA_TO_STRESS_FACTOR: float = -3.6e8


# ---------------------------------------------------------------------------
# Analytic (oracle) path
# ---------------------------------------------------------------------------

def _spatial_mac_at_peak(rec_c: np.ndarray, truth_c: np.ndarray) -> float:
    """Spatial MAC at the peak-response frequency bin.

    Parameters
    ----------
    rec_c, truth_c
        Arrays of shape (nfreq, nnodes): PSD for one stress component.

    Returns
    -------
    MAC value (0–1) computed from the spatial vectors at the frequency bin
    where the truth spatial energy is maximum.
    """
    spatial_energy = np.sum(np.abs(truth_c), axis=1)   # (nfreq,)
    fpk = int(np.argmax(spatial_energy))
    return mac(rec_c[fpk, :], truth_c[fpk, :])


def run_case_from_modal(
    truth_md: dict,
    prior_md: dict,
    *,
    saa_level: float,
    fs: float,
    n_frames: int,
    seed: int,
    noise_sigma_T: float = 0.0,
    n_modes: int | None = None,
    nperseg: int | None = None,
    window: str = "boxcar",
    n_segments: int | None = None,
) -> dict:
    """Oracle / analytic harness — no MAPDL or SEMM required.

    Parameters
    ----------
    truth_md, prior_md
        Modal-data dicts (keys: modal_freqs, modal_omega, zeta, node_coords,
        modal_sx, modal_sy, modal_sxy, gamma_base, modal_mass).
        Pass the same dict for both to get an exact oracle (perfect prior).
    saa_level
        Target base-acceleration PSD [(m/s²)²/Hz].
    fs
        Sampling frequency [Hz].
    n_frames
        Total number of time samples to synthesise.
    seed
        numpy RNG seed for reproducibility.
    noise_sigma_T
        Camera thermal noise standard deviation [K].  0.0 → noise-free.
    n_modes
        Truncate mode shapes to the first *n_modes* columns.  None → keep all.
    nperseg
        Welch segment length [samples].  Highest-priority override.
    window
        Welch window function (default ``"boxcar"`` — no spectral distortion).
    n_segments
        Number of Welch segments (50 % overlap formula from §2.3:
        ``seg = 2*n_frames / (n_segments + 1)``).  Ignored when *nperseg*
        is set.  None → single full-length segment (oracle default).

    Returns
    -------
    dict with keys: recovered, truth, metrics, freqs, condition_number.
    """
    rng = np.random.default_rng(seed)
    factor = _CAMERA_TO_STRESS_FACTOR
    nnodes = int(truth_md["node_coords"].shape[0])
    # synthesize_base_excitation expects a 2-D grid shape; use (nnodes, 1)
    grid_shape = (nnodes, 1)

    # ------------------------------------------------------------------
    # (a) Synthesise noise-free camera frames + accelerometer signal
    # ------------------------------------------------------------------
    out = synthesize_base_excitation(
        truth_md, fs, n_frames, saa_level, rng, factor, grid_shape
    )
    cam_frames = out["cam_frames"]   # (n_frames, nnodes, 1)
    accel = out["accel"]             # (n_frames,)

    if noise_sigma_T > 0.0:
        cam_frames = inject_camera_noise(cam_frames, noise_sigma_T, rng)

    # Convert temperature frames → stress (SX+SY), shape (n_frames, nnodes)
    sigma_cam = cam_frames.reshape(n_frames, nnodes) * factor

    # ------------------------------------------------------------------
    # (b) H1 transmissibility: T_cam = CSD(sigma, accel) / PSD(accel)
    #
    # Segment-length priority (§2.3):
    #   1. nperseg set explicitly → use as-is
    #   2. n_segments set → 50 % overlap formula: seg = 2*n_frames/(n_segments+1)
    #   3. default → single full-length segment (oracle: no averaging variance,
    #      recovers T_cam[f,i] = conj(H_true[f,i]) exactly at every Welch bin)
    # ------------------------------------------------------------------
    if nperseg is not None:
        seg = int(nperseg)
    elif n_segments is not None:
        seg = int(2 * n_frames / (n_segments + 1))
    else:
        seg = n_frames          # single segment → periodogram (no averaging variance)

    welch_kw = dict(fs=fs, nperseg=seg, window=window, detrend=False)

    freqs, S_aa = welch(accel, **welch_kw)   # (nfreq,), (nfreq,)
    nfreq = len(freqs)

    T_cam = np.empty((nfreq, nnodes), dtype=complex)
    for i in range(nnodes):
        # scipy csd(x,y)=conj(FFT x)*FFT y -> this is conj(H); fine for PSD since |conj(H)|^2==|H|^2
        _, G = csd(sigma_cam[:, i], accel, **welch_kw)
        T_cam[:, i] = G / S_aa

    # ------------------------------------------------------------------
    # (c) Mode-shape matrices from prior_md, shape (nnodes, nmodes)
    #     modal_sx etc. are stored as (nmodes, nnodes) in the dict.
    # ------------------------------------------------------------------
    psi_by_comp: dict[str, np.ndarray] = {
        "SX":    prior_md["modal_sx"].T,
        "SY":    prior_md["modal_sy"].T,
        "SXY":   prior_md["modal_sxy"].T,
        "SX+SY": (prior_md["modal_sx"] + prior_md["modal_sy"]).T,
    }
    if n_modes is not None:
        psi_by_comp = {c: p[:, :n_modes] for c, p in psi_by_comp.items()}

    # Normalise so ||Psi_cam[:, r]|| = 1 (mode-by-mode, driven by SX+SY)
    psi_by_comp = normalize_mode_shapes(psi_by_comp)

    # ------------------------------------------------------------------
    # (d) Modal decomposition → expansion → recovered stress PSD
    # ------------------------------------------------------------------
    gamma = modal_decompose(psi_by_comp["SX+SY"], T_cam)   # (nfreq, nmodes)
    T_by_comp = expand_components(gamma, psi_by_comp)       # {comp: (nfreq, nnodes)}
    recovered = stress_psd(T_by_comp, S_aa)                 # {comp: (nfreq, nnodes)}

    # ------------------------------------------------------------------
    # (e) Analytic ground-truth PSD on the SAME frequency grid and S_aa
    # ------------------------------------------------------------------
    truth = truth_component_psd(truth_md, freqs, S_aa)      # {comp: (nfreq, nnodes)}

    # ------------------------------------------------------------------
    # (f) Metrics per component
    # ------------------------------------------------------------------
    met_nrmse = {c: nrmse(recovered[c], truth[c]) for c in recovered}
    met_mac   = {c: _spatial_mac_at_peak(recovered[c], truth[c]) for c in recovered}
    met_relrms = {c: rel_rms_error(recovered[c], truth[c]) for c in recovered}
    met_peak   = {c: peak_ratio(recovered[c], truth[c]) for c in recovered}

    return {
        "recovered":        recovered,
        "truth":            truth,
        # nrmse/mac hide proportional errors; rel_rms + peak_ratio expose them
        "metrics":          {"nrmse": met_nrmse, "mac": met_mac,
                             "rel_rms": met_relrms, "peak_ratio": met_peak},
        "freqs":            freqs,
        "condition_number": condition_number(psi_by_comp["SX+SY"]),
        # complex per-component transmissibilities + excitation PSD, so callers
        # can form the full 3x3 cross-spectral stress-PSD matrix (S_ij = T_i conj(T_j) S_aa)
        "T_by_comp":        T_by_comp,
        "S_aa":             S_aa,
    }


# ---------------------------------------------------------------------------
# Direct-PSD (accelerometer-free) path
# ---------------------------------------------------------------------------

def run_direct_psd_from_modal(
    truth_md: dict,
    prior_md: dict,
    *,
    saa_level: float,
    fs: float,
    n_frames: int,
    seed: int,
    noise_sigma_T: float = 0.0,
    n_modes: int | None = None,
    nperseg: int | None = None,
    window: str = "boxcar",
    n_segments: int | None = None,
) -> dict:
    """Direct-PSD (accelerometer-free) expansion — analytic/oracle path.

    Synthesises the camera time history from *truth_md*, computes the stress
    PSD directly (no accelerometer), denoises it via a virtual spatial-average
    reference, decomposes it into modal contributions via NNLS, and expands
    to all stress components using *prior_md* mode shapes.

    Parameters mirror :func:`run_case_from_modal` exactly (same signature,
    same return-dict shape), except there is no accelerometer involved and
    ``condition_number`` is the condition of the squared-mode-shape basis
    ``|Psi_cam|^2`` rather than ``Psi_cam`` itself.

    Key limitation: the direct-PSD method recovers **auto-PSDs only**.
    Cross-spectral densities between nodes or between stress components cannot
    be recovered because phase information is discarded by the NNLS step.
    The transmissibility method (run_case_from_modal) does not have this
    limitation.

    Returns
    -------
    dict with keys: recovered, truth, metrics{nrmse, mac, rel_rms, peak_ratio},
    freqs, condition_number.
    """
    rng = np.random.default_rng(seed)
    factor = _CAMERA_TO_STRESS_FACTOR
    nnodes = int(truth_md["node_coords"].shape[0])
    grid_shape = (nnodes, 1)

    # ------------------------------------------------------------------
    # (a) Synthesise noise-free camera frames + accelerometer signal
    # ------------------------------------------------------------------
    out = synthesize_base_excitation(
        truth_md, fs, n_frames, saa_level, rng, factor, grid_shape
    )
    cam_frames = out["cam_frames"]   # (n_frames, nnodes, 1)
    accel      = out["accel"]        # (n_frames,) — kept for truth S_aa only

    if noise_sigma_T > 0.0:
        cam_frames = inject_camera_noise(cam_frames, noise_sigma_T, rng)

    # Convert temperature frames → stress (SX+SY), shape (n_frames, nnodes)
    sigma_cam = cam_frames.reshape(n_frames, nnodes) * factor

    # ------------------------------------------------------------------
    # (b) Welch segment length (same priority logic as run_case_from_modal)
    # ------------------------------------------------------------------
    if nperseg is not None:
        seg = int(nperseg)
    elif n_segments is not None:
        seg = int(2 * n_frames / (n_segments + 1))
    else:
        seg = n_frames      # single full-length segment (oracle default)

    welch_kw = dict(fs=fs, nperseg=seg, window=window, detrend=False)

    # ------------------------------------------------------------------
    # (c) Virtual-reference denoising: remove spatially-uncorrelated noise
    # ------------------------------------------------------------------
    freqs, S_clean = virtual_reference_denoise(sigma_cam, welch_kw)

    # ------------------------------------------------------------------
    # (d) Mode-shape matrices from prior_md, shape (nnodes, nmodes)
    # ------------------------------------------------------------------
    psi_by_comp: dict[str, np.ndarray] = {
        "SX":    prior_md["modal_sx"].T,
        "SY":    prior_md["modal_sy"].T,
        "SXY":   prior_md["modal_sxy"].T,
        "SX+SY": (prior_md["modal_sx"] + prior_md["modal_sy"]).T,
    }
    if n_modes is not None:
        psi_by_comp = {c: p[:, :n_modes] for c, p in psi_by_comp.items()}

    # Normalise so ||Psi_cam[:, r]|| = 1  (driven by SX+SY, same as transmissibility path)
    psi_by_comp = normalize_mode_shapes(psi_by_comp)

    # ------------------------------------------------------------------
    # (e) NNLS decomposition into modal participation PSDs
    # ------------------------------------------------------------------
    Psi_cam = psi_by_comp["SX+SY"]                           # (nnodes, nmodes)
    beta    = nnls_psd_decompose(S_clean, Psi_cam)           # (nfreq, nmodes)

    # ------------------------------------------------------------------
    # (f) Expand to all stress components
    # ------------------------------------------------------------------
    recovered = expand_psd(beta, psi_by_comp)                # {comp: (nfreq, nnodes)}

    # ------------------------------------------------------------------
    # (g) Ground-truth PSD on the same frequency grid.
    #     Use Welch of the synthesised acceleration so that the comparison
    #     is fair (same finite-sample S_aa as the transmissibility path).
    # ------------------------------------------------------------------
    _, S_aa = welch(accel, **welch_kw)                       # (nfreq,)
    truth = truth_component_psd(truth_md, freqs, S_aa)       # {comp: (nfreq, nnodes)}

    # ------------------------------------------------------------------
    # (h) Metrics per component
    # ------------------------------------------------------------------
    met_nrmse  = {c: nrmse(recovered[c], truth[c]) for c in recovered}
    met_mac    = {c: _spatial_mac_at_peak(recovered[c], truth[c]) for c in recovered}
    met_relrms = {c: rel_rms_error(recovered[c], truth[c]) for c in recovered}
    met_peak   = {c: peak_ratio(recovered[c], truth[c]) for c in recovered}

    return {
        "recovered":        recovered,
        "truth":            truth,
        "metrics":          {"nrmse": met_nrmse, "mac": met_mac,
                             "rel_rms": met_relrms, "peak_ratio": met_peak},
        "freqs":            freqs,
        # condition number on the squared-mode-shape basis (real, non-negative)
        "condition_number": condition_number(np.abs(Psi_cam) ** 2),
    }


# ---------------------------------------------------------------------------
# Full path (MAPDL + SEMM) — lazy imports so tests run without pyFBS/ANSYS
# ---------------------------------------------------------------------------

def run_case(
    truth_cfg,
    parent_cfg,
    *,
    noise: bool = True,
    n_modes: int = 2,
    saa_level: float,
    fs: float,
    n_frames: int,
    seed: int,
    noise_sigma_T: float | None = None,
    use_truth_as_prior: bool = False,
    nperseg: int | None = None,
    window: str = "boxcar",
    n_segments: int | None = None,
) -> dict:
    """Full path: generate_fe → run_semm_stage1 → extract_mode_shapes.

    Lazy-imports fe_models and semm_adapter so the analytic path and its test
    do not require pyFBS / ANSYS at import time.

    Parameters
    ----------
    truth_cfg, parent_cfg
        FEConfig objects (or dicts accepted by generate_fe) describing the
        truth and parent FE models respectively.
    noise
        If True, inject camera noise.  The noise level is set by
        ``noise_sigma_T`` when given, otherwise derived from
        ``truth_cfg["noise_sigma_T"]`` (defaults to 0.01 K if absent).
    n_modes
        Number of modes to extract from the SEMM FRFs.
    saa_level, fs, n_frames, seed
        Same as in run_case_from_modal.
    noise_sigma_T
        Camera thermal noise standard deviation [K].  When provided, overrides
        the value derived from *truth_cfg* (used by study_b SNR sweep).
        None → derive from truth_cfg (legacy behaviour).
    use_truth_as_prior
        If True, use the truth FE model's own modal data as the expansion
        prior (oracle / cheat run); SEMM output is still computed but not
        used for the prior.

    Returns
    -------
    dict with the same keys as run_case_from_modal.
    """
    # Lazy imports — not needed at module level
    from synthetic_validation.fe_models import generate_fe          # noqa: PLC0415
    from synthetic_validation.semm_adapter import run_semm_stage1   # noqa: PLC0415
    from synthetic_validation.expansion import extract_mode_shapes   # noqa: PLC0415

    # ------------------------------------------------------------------
    # 1. Generate FE models for each force case in truth_cfg
    # ------------------------------------------------------------------
    force_labels = list(truth_cfg.force_points.keys())
    truth_fes = [generate_fe(truth_cfg, fl) for fl in force_labels]

    # Stack per-case stress-tensor FRFs into (nfreq, nnodes*4, ncases)
    # truth_fes[k]["stress_tensor_frf"]: (nfreq, nnodes, 4) → reshape to (nfreq, nnodes*4)
    def _stack_frfs(fes):
        # Each FE dict has stress_tensor_frf: (nfreq, nnodes*4, ...) or (nfreq, nnodes, 4)
        parts = []
        for fe in fes:
            frf = fe["stress_tensor_frf"]          # (nfreq, nnodes*4) or (nfreq, nnodes, 4)
            if frf.ndim == 3:
                nfreq, nn, nc = frf.shape
                frf = frf.reshape(nfreq, nn * nc)
            parts.append(frf[:, :, np.newaxis])    # (nfreq, nnodes*4, 1)
        return np.concatenate(parts, axis=2)        # (nfreq, nnodes*4, ncases)

    parent_frf = _stack_frfs(truth_fes if use_truth_as_prior else
                              [generate_fe(parent_cfg, fl) for fl in force_labels])
    truth_frf_full = _stack_frfs(truth_fes)

    freq_axis = truth_fes[0]["freqs"]
    node_coords = truth_fes[0]["node_coords"]

    # SX+SY overlay (component index 3): (nfreq, nnodes, ncases)
    nfreq_fe = truth_frf_full.shape[0]
    nnodes_fe = truth_frf_full.shape[1] // 4
    ncases = truth_frf_full.shape[2]
    overlay_frf = truth_frf_full[:, 3::4, :]       # (nfreq, nnodes, ncases)

    # ------------------------------------------------------------------
    # 2. SEMM stage 1
    # ------------------------------------------------------------------
    Y_SEMM = run_semm_stage1(parent_frf, overlay_frf, node_coords, freq_axis)

    # ------------------------------------------------------------------
    # 3. Extract mode shapes from SEMM FRFs
    # ------------------------------------------------------------------
    mode_data = extract_mode_shapes(Y_SEMM, freq_axis, n_modes=n_modes)

    # ------------------------------------------------------------------
    # 4. Build modal_data dicts for truth and prior
    # ------------------------------------------------------------------
    truth_modal = truth_fes[0]["modal_data"]

    if use_truth_as_prior:
        prior_modal = truth_modal
    else:
        prior_modal = _mode_data_to_modal_dict(mode_data, node_coords, truth_modal)

    # ------------------------------------------------------------------
    # 5. Delegate to analytic path
    # ------------------------------------------------------------------
    if noise_sigma_T is not None:
        # Explicit override from caller (e.g., study_b SNR sweep).
        _noise_sigma = float(noise_sigma_T) if noise else 0.0
    else:
        if isinstance(truth_cfg, dict):
            sigma_T = truth_cfg.get("noise_sigma_T", 0.01)
        else:
            sigma_T = getattr(truth_cfg, "noise_sigma_T", 0.01)
        _noise_sigma = float(sigma_T) if noise else 0.0

    return run_case_from_modal(
        truth_modal, prior_modal,
        saa_level=saa_level, fs=fs, n_frames=n_frames, seed=seed,
        noise_sigma_T=_noise_sigma, n_modes=n_modes,
        nperseg=nperseg, window=window, n_segments=n_segments,
    )


def _mode_data_to_modal_dict(
    mode_data: dict,
    node_coords: np.ndarray,
    reference_modal: dict,
) -> dict:
    """Convert extract_mode_shapes output to a modal_data dict compatible with
    synthesize_base_excitation and base_excitation_frf.

    extract_mode_shapes returns:
        {"freqs": (nmodes,), "psi": {comp: (nnodes, nmodes)}}

    We borrow gamma_base, zeta, and modal_mass from *reference_modal*
    (truncated / padded to match the number of extracted modes).
    """
    psi = mode_data["psi"]
    freqs = np.asarray(mode_data["freqs"], float)
    nmodes = len(freqs)

    def _borrow(key, default):
        ref = reference_modal.get(key, None)
        if ref is None:
            return np.full(nmodes, default)
        ref = np.atleast_1d(ref)
        if len(ref) >= nmodes:
            return ref[:nmodes]
        # pad with last value
        return np.concatenate([ref, np.full(nmodes - len(ref), ref[-1])])

    return {
        "modal_freqs":  freqs,
        "modal_omega":  2.0 * np.pi * freqs,
        "zeta":         _borrow("zeta", 0.01),
        "node_coords":  node_coords,
        "modal_sx":     psi["SX"].T,            # (nmodes, nnodes)
        "modal_sy":     psi["SY"].T,
        "modal_sxy":    psi["SXY"].T,
        "gamma_base":   _borrow("gamma_base", 1.0),
        "modal_mass":   _borrow("modal_mass", 1.0),
    }
