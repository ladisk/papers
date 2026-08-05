import numpy as np

def nrmse(recovered, truth, axis=None):
    recovered = np.asarray(recovered); truth = np.asarray(truth)
    rms = np.sqrt(np.mean((recovered - truth) ** 2, axis=axis))
    rng = np.max(truth, axis=axis) - np.min(truth, axis=axis)
    if axis is None:
        return float(rms / rng) if rng > 0 else 0.0
    safe_rng = np.where(rng > 0, rng, 1.0)
    return np.where(rng > 0, rms / safe_rng, 0.0)

def mac(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    num = np.abs(np.vdot(a, b)) ** 2
    den = np.vdot(a, a).real * np.vdot(b, b).real
    return float(num / den) if den > 0 else 0.0

def point_error(recovered, truth, node_indices):
    r = np.asarray(recovered); t = np.asarray(truth)
    idx = np.asarray(node_indices, int)
    denom = np.where(np.abs(t[..., idx]) > 0, np.abs(t[..., idx]), np.nan)
    return np.abs(r[..., idx] - t[..., idx]) / denom

def rel_rms_error(recovered, truth):
    """Relative RMS error normalized by the TRUTH norm (not its range).

    Unlike nrmse (range-normalized) and mac (scale-invariant), this is sensitive
    to a *proportional* error: recovered = k*truth gives exactly |k-1|.  Use it to
    expose a global under/over-estimate (e.g. Welch peak bias) that nrmse/mac hide.
    """
    r = np.asarray(recovered); t = np.asarray(truth)
    den = np.linalg.norm(t)
    return float(np.linalg.norm(r - t) / den) if den > 0 else 0.0

def peak_ratio(recovered, truth):
    """Median over nodes of recovered/truth at the truth's peak-energy frequency.

    1.0 = the resonance amplitude is recovered; <1 = underestimate (e.g. the sharp
    resonance smeared by a segmented Welch estimate). recovered/truth are (nfreq,nnodes).
    """
    r = np.asarray(recovered); t = np.asarray(truth)
    fk = int(np.argmax(np.sum(np.abs(t), axis=1)))
    denom = t[fk]
    m = np.abs(denom) > 0
    return float(np.median(r[fk][m] / denom[m])) if np.any(m) else 0.0

def condition_number(Psi):
    return float(np.linalg.cond(np.asarray(Psi)))

def component_ratio_error(recovered_by_comp, truth_by_comp):
    def ratios(d):
        base = np.abs(d["SX"]) + 1e-30
        return np.stack([np.abs(d["SY"]) / base, np.abs(d["SXY"]) / base])
    rr, rt = ratios(recovered_by_comp), ratios(truth_by_comp)
    return float(np.sqrt(np.mean((rr - rt) ** 2)))
