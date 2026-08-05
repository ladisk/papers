"""Reviewer-study drivers A–E for the synthetic validation suite.

Each driver is a thin loop over the harness; no pipeline logic lives here.

Analytic vs full-path dispatch
-------------------------------
Every driver accepts ``analytic: bool = False``.

* ``analytic=True``  — positional args are **modal_data dicts**; dispatches to
  ``run_case_from_modal``.  Used by tests (no ANSYS/pyFBS required).
* ``analytic=False`` — positional args are **FEConfig objects**; dispatches to
  ``run_case`` (MAPDL + SEMM).

SNR → noise mapping (Study B)
------------------------------
``noise_sigma_T = _SNR_BASE_NOISE / snr`` where ``_SNR_BASE_NOISE = 0.1 K``.
Higher SNR → smaller thermal-noise std.

Studies
-------
A  recovery           — realistic (parent prior) + oracle (truth prior) side-by-side.
B  noise              — Monte-Carlo confidence bands over n_reps realisations per SNR.
C  fe_discrepancy     — NRMSE vs parent-model discrepancy for a list of variants.
D  modal_convergence  — NRMSE vs number of modes included in the expansion.
E  conditioning       — Condition number of the mode-shape matrix; optional Tikhonov note.
"""
from __future__ import annotations

import numpy as np

from synthetic_validation.harness import run_case_from_modal

# Noise standard deviation [K] used as the reference for SNR=1.
_SNR_BASE_NOISE: float = 0.1

# Stress component keys produced by the harness.
_COMPS = ("SX", "SY", "SXY", "SX+SY")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(truth, parent, *, analytic: bool, **kw) -> dict:
    """Dispatch one harness call."""
    if analytic:
        return run_case_from_modal(truth, parent, **kw)
    else:
        # Lazy import keeps the analytic path free from pyFBS/ANSYS at module load.
        from synthetic_validation.harness import run_case  # noqa: PLC0415
        return run_case(truth, parent, **kw)


def _run_oracle(truth, parent, *, analytic: bool, **kw) -> dict:
    """Oracle run: use truth modal data as the expansion prior."""
    if analytic:
        return run_case_from_modal(truth, truth, **kw)
    else:
        from synthetic_validation.harness import run_case  # noqa: PLC0415
        return run_case(truth, parent, use_truth_as_prior=True, **kw)


# ---------------------------------------------------------------------------
# Study A — Recovery (realistic vs oracle)
# ---------------------------------------------------------------------------

def study_a_recovery(truth, parent, *, analytic: bool = False, **kw) -> dict:
    """Run a realistic and an oracle recovery and return both.

    Parameters
    ----------
    truth, parent
        Modal-data dicts (``analytic=True``) or FEConfig objects (``analytic=False``).
    analytic
        Dispatch to the analytic path (no ANSYS).
    **kw
        Forwarded to the harness.  Must include at minimum:
        ``saa_level``, ``fs``, ``n_frames``, ``seed``.

    Returns
    -------
    dict with keys ``"realistic"`` and ``"oracle"``, each a sub-dict::

        {
            "metrics":   {"nrmse": {comp: float}, "mac": {comp: float}},
            "recovered": {comp: (nfreq, nnodes)},
            "truth":     {comp: (nfreq, nnodes)},
            "freqs":     (nfreq,),
            "condition_number": float,
        }
    """
    res_realistic = _run(truth, parent, analytic=analytic, **kw)
    res_oracle = _run_oracle(truth, parent, analytic=analytic, **kw)

    def _extract(res):
        return {
            "metrics":          res["metrics"],
            "recovered":        res["recovered"],
            "truth":            res["truth"],
            "freqs":            res["freqs"],
            "condition_number": res["condition_number"],
        }

    return {
        "realistic": _extract(res_realistic),
        "oracle":    _extract(res_oracle),
    }


# ---------------------------------------------------------------------------
# Study B — Noise confidence bands
# ---------------------------------------------------------------------------

def study_b_noise(
    truth,
    parent,
    *,
    n_reps: int,
    snr_levels: list[float],
    analytic: bool = False,
    **kw,
) -> dict:
    """Monte-Carlo confidence bands for the recovered PSD over SNR levels.

    For each SNR in *snr_levels*, runs the harness ``n_reps`` times (different
    seeds) with ``noise_sigma_T = _SNR_BASE_NOISE / snr``.  Returns per-
    component (per-SNR) mean ± std of the recovered PSD.

    Parameters
    ----------
    truth, parent
        Modal-data dicts or FEConfig objects.
    n_reps
        Number of Monte-Carlo realisations per SNR.
    snr_levels
        List of SNR values (>0).  Higher → smaller noise.
    analytic
        Dispatch to the analytic path (no ANSYS).
    **kw
        Forwarded to the harness.  ``seed`` (default 0) is used as a base
        seed; each rep offsets it by the rep index.  ``noise_sigma_T`` in
        *kw* is ignored (overridden by the SNR mapping).  When
        ``n_segments`` is in *kw* Welch averaging is active.

    Returns
    -------
    dict::

        {
            "confidence_bands": {
                comp: {
                    snr_val: {"mean": (nfreq, nnodes), "std": (nfreq, nnodes)}
                }
            },
            "n_reps":     int,
            "snr_levels": list[float],
        }
    """
    base_seed = kw.pop("seed", 0)
    kw.pop("noise_sigma_T", None)   # we set it from the SNR mapping

    confidence_bands: dict[str, dict] = {comp: {} for comp in _COMPS}

    for snr_idx, snr in enumerate(snr_levels):
        noise_sigma_T = _SNR_BASE_NOISE / float(snr)
        runs_by_comp: dict[str, list] = {comp: [] for comp in _COMPS}

        for rep in range(n_reps):
            seed = base_seed + snr_idx * n_reps + rep
            if analytic:
                res = run_case_from_modal(
                    truth, parent,
                    seed=seed,
                    noise_sigma_T=noise_sigma_T,
                    **kw,
                )
            else:
                # Full path: forward the SNR-derived noise level explicitly.
                from synthetic_validation.harness import run_case  # noqa: PLC0415
                res = run_case(truth, parent, seed=seed,
                               noise_sigma_T=noise_sigma_T, **kw)

            for comp in _COMPS:
                if comp in res["recovered"]:
                    runs_by_comp[comp].append(res["recovered"][comp])

        for comp in _COMPS:
            if runs_by_comp[comp]:
                stacked = np.stack(runs_by_comp[comp], axis=0)  # (n_reps, nfreq, nnodes)
                confidence_bands[comp][snr] = {
                    "mean": np.mean(stacked, axis=0),
                    "std":  np.std(stacked, axis=0),
                }

    return {
        "confidence_bands": confidence_bands,
        "n_reps":           n_reps,
        "snr_levels":       snr_levels,
    }


# ---------------------------------------------------------------------------
# Study C — FE-model discrepancy
# ---------------------------------------------------------------------------

def study_c_fe_discrepancy(
    truth,
    parent_variants,
    *,
    analytic: bool = False,
    nominal_idx: int = 0,
    **kw,
) -> dict:
    """NRMSE and MAC vs parent-model discrepancy for each variant.

    Also computes the realistic−oracle gap using ``parent_variants[nominal_idx]``
    as the nominal parent.

    Parameters
    ----------
    truth
        Truth modal-data dict or FEConfig.
    parent_variants
        List of modal-data dicts or FEConfig objects (one per perturbation).
    analytic
        Dispatch to the analytic path (no ANSYS).
    nominal_idx
        Index of the nominal/unperturbed variant in *parent_variants*.  Used
        to compute the realistic−oracle gap.  Default 0.  Pass the index of
        the dp=0 entry when variants are ordered as perturbations around zero
        (e.g. ``nominal_idx=3`` when perturbations are
        ``[-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]``).
    **kw
        Forwarded to the harness (must include saa_level, fs, n_frames, seed).

    Returns
    -------
    dict::

        {
            "variants": [
                {
                    "metrics": {"nrmse": {comp: float}, "mac": {comp: float}},
                    "recovered": {comp: (nfreq, nnodes)},
                    "freqs":     (nfreq,),
                    "condition_number": float,
                },
                ...
            ],
            "realistic_oracle_gap": {comp: float},   # NRMSE(realistic) - NRMSE(oracle)
        }
    """
    variant_results = []
    for variant in parent_variants:
        res = _run(truth, variant, analytic=analytic, **kw)
        variant_results.append({
            "metrics":          res["metrics"],
            "recovered":        res["recovered"],
            "freqs":            res["freqs"],
            "condition_number": res["condition_number"],
        })

    # Realistic−oracle gap using parent_variants[nominal_idx] as the nominal parent.
    oracle_gap: dict[str, float] = {}
    if parent_variants:
        res_oracle = _run_oracle(truth, parent_variants[nominal_idx], analytic=analytic, **kw)
        nrmse_real   = variant_results[nominal_idx]["metrics"]["nrmse"]
        nrmse_oracle = res_oracle["metrics"]["nrmse"]
        oracle_gap = {
            comp: float(nrmse_real[comp]) - float(nrmse_oracle[comp])
            for comp in nrmse_real
        }

    return {
        "variants":              variant_results,
        "realistic_oracle_gap":  oracle_gap,
    }


# ---------------------------------------------------------------------------
# Study D — Modal convergence
# ---------------------------------------------------------------------------

def study_d_modal_convergence(
    truth,
    parent,
    *,
    mode_counts: list[int],
    analytic: bool = False,
    **kw,
) -> dict:
    """NRMSE and MAC vs number of modes included in the expansion.

    Parameters
    ----------
    truth, parent
        Modal-data dicts or FEConfig objects.
    mode_counts
        List of integers; for each, the harness is called with
        ``n_modes=count``.  Values must not exceed the number of modes
        available in the modal data / FE model.
    analytic
        Dispatch to the analytic path (no ANSYS).
    **kw
        Forwarded to the harness.  Any ``n_modes`` already in *kw* is
        ignored (overridden by *mode_counts* sweep).

    Returns
    -------
    dict::

        {
            "mode_counts": [int, ...],
            "nrmse": {comp: [float, ...]},
            "mac":   {comp: [float, ...]},
            "condition_numbers": [float, ...],
        }
    """
    kw.pop("n_modes", None)     # sweep overrides any caller-supplied value

    nrmse_by_comp: dict[str, list[float]] = {comp: [] for comp in _COMPS}
    mac_by_comp:   dict[str, list[float]] = {comp: [] for comp in _COMPS}
    cond_numbers: list[float] = []

    for n_modes in mode_counts:
        res = _run(truth, parent, analytic=analytic, n_modes=n_modes, **kw)
        for comp in _COMPS:
            nrmse_by_comp[comp].append(float(res["metrics"]["nrmse"].get(comp, np.nan)))
            mac_by_comp[comp].append(float(res["metrics"]["mac"].get(comp, np.nan)))
        cond_numbers.append(float(res["condition_number"]))

    return {
        "mode_counts":       mode_counts,
        "nrmse":             nrmse_by_comp,
        "mac":               mac_by_comp,
        "condition_numbers": cond_numbers,
    }


# ---------------------------------------------------------------------------
# Study E — Conditioning
# ---------------------------------------------------------------------------

def study_e_conditioning(
    truth,
    parent,
    *,
    regularize: bool = False,
    analytic: bool = False,
    **kw,
) -> dict:
    """Report the condition number of the mode-shape matrix.

    Parameters
    ----------
    truth, parent
        Modal-data dicts or FEConfig objects.
    regularize
        If True, record a note that the effect of regularisation was
        requested; the harness itself does not yet support a Tikhonov
        variant — this flag is a placeholder for future extension.
    analytic
        Dispatch to the analytic path (no ANSYS).
    **kw
        Forwarded to the harness (must include saa_level, fs, n_frames, seed).

    Returns
    -------
    dict::

        {
            "condition_number":   float,
            "metrics":            {"nrmse": {comp: float}, "mac": {comp: float}},
            "regularize_requested": bool,
            "regularize_applied":   bool,   # always False — not yet wired
            "regularize_note":    str,      # present only when regularize_requested=True
        }
    """
    res = _run(truth, parent, analytic=analytic, **kw)

    out: dict = {
        "condition_number":     float(res["condition_number"]),
        "metrics":              res["metrics"],
        "regularize_requested": bool(regularize),
        "regularize_applied":   False,   # Tikhonov / truncated-SVD not yet wired
    }

    if regularize:
        out["regularize_note"] = (
            "Tikhonov / truncated-SVD regularisation is not yet wired into the "
            "harness.  Requesting regularize=True records the intent; compare "
            "condition_number with and without n_modes truncation to gauge "
            "sensitivity."
        )

    return out
