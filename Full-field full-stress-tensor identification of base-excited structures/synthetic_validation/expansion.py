import numpy as np
from scipy.signal import find_peaks, welch as _welch, csd as _csd
from scipy.optimize import nnls as _nnls

COMPONENTS = ("SX", "SY", "SXY", "SX+SY")

def normalize_mode_shapes(psi_by_comp):
    cam = psi_by_comp["SX+SY"]                       # (nnodes, nmodes)
    scale = np.linalg.norm(cam, axis=0)              # per-mode ||psi_cam||
    scale = np.where(scale > 0, scale, 1.0)
    return {c: psi_by_comp[c] / scale[None, :] for c in psi_by_comp}

def modal_decompose(Psi_cam, T_cam_nodes):
    pinv = np.linalg.pinv(Psi_cam)                   # (nmodes, nnodes)
    return (pinv @ T_cam_nodes.T).T                  # (nfreq, nmodes)

def expand_components(gamma, psi_by_comp):
    return {c: gamma @ psi_by_comp[c].T for c in psi_by_comp}   # (nfreq, nnodes)

def stress_psd(T_by_comp, S_aa):
    S_aa = np.asarray(S_aa, float)
    return {c: (np.abs(T)**2) * S_aa[:, None] for c, T in T_by_comp.items()}

def _spatial_rms_over_cases(Y_comp):        # Y_comp (nfreq, nnodes, ncases)
    return np.sqrt(np.mean(np.abs(Y_comp)**2, axis=(1, 2)))

def virtual_reference_denoise(sigma_cam, welch_kw):
    """Denoise camera stress PSD via a virtual spatial-average reference.

    Computes the coherent PSD at each node relative to the spatial-mean
    reference r(t) = mean_x(sigma_cam[t, x]).  Spatially uncorrelated noise
    is suppressed because it is incoherent with r.

    Parameters
    ----------
    sigma_cam : (n_frames, nnodes) array
        Measured SX+SY stress from the camera (already converted to Pa).
    welch_kw : dict
        Keyword arguments forwarded to ``scipy.signal.welch`` and ``csd``
        (must include at least ``fs`` and ``nperseg``).

    Returns
    -------
    freqs : (nfreq,) ndarray — frequency axis [Hz]
    S_clean : (nfreq, nnodes) ndarray, real >=0 — denoised stress auto-PSD [Pa^2/Hz]

    Notes
    -----
    Formula (method doc §4b):
        S_clean[f, x] = |CSD(sigma_cam[:, x], r)|^2 / PSD(r)[f]
    For a single-mode zero-noise signal this equals the direct auto-PSD of
    sigma_cam[:, x], because the spatial-mean reference is perfectly correlated
    with every node.
    """
    sigma_cam = np.asarray(sigma_cam, dtype=float)
    r = sigma_cam.mean(axis=1)          # virtual reference (n_frames,)
    freqs, S_rr = _welch(r, **welch_kw)
    nfreq = len(freqs)
    nnodes = sigma_cam.shape[1]
    S_clean = np.zeros((nfreq, nnodes), dtype=float)
    safe_S_rr = np.where(S_rr > 0.0, S_rr, 1.0)   # avoid division by zero
    for i in range(nnodes):
        _, G_xr = _csd(sigma_cam[:, i], r, **welch_kw)
        S_clean[:, i] = np.where(S_rr > 0.0, np.abs(G_xr) ** 2 / safe_S_rr, 0.0)
    return freqs, S_clean


def nnls_psd_decompose(S_clean, Psi_cam):
    """Decompose a denoised camera PSD into non-negative modal participation PSDs.

    Solves, independently at each frequency bin:
        S_clean[f, :] ≈ |Psi_cam|^2 @ beta[f, :]    s.t. beta >= 0
    using scipy's NNLS (Active Set method).

    Parameters
    ----------
    S_clean : (nfreq, nnodes) array
        Denoised camera stress PSD (real, >= 0).
    Psi_cam : (nnodes, nmodes) array
        Camera-component (SX+SY) mode shapes, already normalised.

    Returns
    -------
    beta : (nfreq, nmodes) ndarray, real >=0 — modal participation PSD [Pa^2/Hz]
    """
    S_clean = np.asarray(S_clean, dtype=float)
    B = np.abs(np.asarray(Psi_cam)) ** 2        # (nnodes, nmodes), real >= 0
    nfreq, nnodes = S_clean.shape
    nmodes = B.shape[1]
    beta = np.zeros((nfreq, nmodes), dtype=float)
    for i_f in range(nfreq):
        beta[i_f, :], _ = _nnls(B, S_clean[i_f, :])
    return beta


def expand_psd(beta, psi_by_comp):
    """Expand modal participation PSDs to all stress components.

    For each component c:
        S_c[f, x] = sum_r  beta[f, r] * |psi_by_comp[c][x, r]|^2

    Parameters
    ----------
    beta : (nfreq, nmodes) real array — modal participation PSD from NNLS.
    psi_by_comp : dict[str, (nnodes, nmodes)] — normalised mode shapes per component.

    Returns
    -------
    dict[str, (nfreq, nnodes)] ndarray — auto-PSD per stress component (real >= 0).

    Notes
    -----
    The direct-PSD method recovers **auto-PSDs only** — phase information (and
    therefore cross-spectral densities between nodes or components) is lost.
    This is the fundamental contrast with the transmissibility method, which
    preserves complex transfer functions and can form the full 3×3 stress-PSD
    matrix S_ij = T_i * conj(T_j) * S_aa.
    """
    beta = np.asarray(beta, dtype=float)
    return {c: beta @ (np.abs(np.asarray(psi)) ** 2).T
            for c, psi in psi_by_comp.items()}


def _signed_shape_at_peak(H_by_off, fi, ref_off):
    """Signed real mode shape at frequency bin *fi* from complex FRFs.

    ``H_by_off`` maps a component offset (0=SX, 1=SY, 2=SXY, 3=SX+SY) to its
    complex FRF array ``(nfreq, nnodes, ncases)``.

    Near a resonance the response is dominated by a single (real, proportionally
    damped) mode, so for every impact case *k* the FRF equals the real mode shape
    times one complex per-case scalar: ``H_k(x) ≈ phi(x) * a_k``.  We recover the
    signed ``phi`` by rotating each case's phase so it aligns with a common
    reference (the SX+SY invariant of the highest-energy case), taking the real
    part, and averaging over cases.  The SAME per-case rotation is applied to
    every stress component, so the relative sign between components (e.g. tau_xy
    vs the invariant) is preserved.  This mirrors what LSCF / PolyMAX deliver via
    complex modal residues — unlike a magnitude-only peak-pick, which discards
    the sign of shear (tau_xy) reversals.
    """
    ref = H_by_off[ref_off][fi]                      # (nnodes, ncases) complex invariant
    ncases = ref.shape[1]
    # Anchor = case with the largest modal response; make its shape maximally real.
    anchor = ref[:, int(np.argmax(np.sum(np.abs(ref) ** 2, axis=0)))]
    theta0 = 0.5 * np.angle(np.sum(anchor ** 2))
    # Per-case phase that aligns each case's invariant to the anchor.
    phase = np.empty(ncases, dtype=complex)
    for k in range(ncases):
        proj = np.vdot(anchor, ref[:, k])            # conj(anchor) . ref_k
        phase[k] = np.exp(-1j * (theta0 + np.angle(proj))) if np.abs(proj) > 0 else np.exp(-1j * theta0)
    return {off: np.real(H[fi] * phase[None, :]).mean(axis=1) for off, H in H_by_off.items()}


def extract_mode_shapes(Y_SEMM, freq_axis, n_modes=2, prominence=None):
    # Shapes are extracted as SIGNED real mode shapes: each impact case is phase-
    # aligned to a common reference (the SX+SY invariant) before averaging, so the
    # spatial sign of sign-varying components (SXY) is preserved — as LSCF/PolyMAX
    # do via complex modal residues. See _signed_shape_at_peak.
    cam = Y_SEMM[:, 3::4, :]                              # SX+SY (nfreq, nnodes, ncases)
    rms = _spatial_rms_over_cases(cam)
    peaks, props = find_peaks(rms, prominence=(prominence or np.ptp(rms) * 0.05))
    order = np.argsort(rms[peaks])[::-1][:n_modes]
    peak_idx = np.sort(peaks[order])
    comp_slices = {"SX": 0, "SY": 1, "SXY": 2, "SX+SY": 3}
    nnodes = Y_SEMM.shape[1] // 4
    H_by_off = {off: Y_SEMM[:, off::4, :] for off in comp_slices.values()}
    psi = {c: [] for c in comp_slices}
    for fi in peak_idx:
        signed = _signed_shape_at_peak(H_by_off, fi, ref_off=comp_slices["SX+SY"])
        for c, off in comp_slices.items():
            psi[c].append(signed[off])
    psi = {c: (np.column_stack(v) if v else np.zeros((nnodes, 0))) for c, v in psi.items()}
    return {"freqs": freq_axis[peak_idx], "psi": psi}
