"""Tests for the direct-PSD (accelerometer-free) expansion method.

TDD approach: tests written before implementation, then implementation added
to make them pass.

Oracle scenario: single_mode_modal fixture used as both truth AND prior
(perfect mode shapes), zero camera noise.  For a single-mode system the NNLS
with the exact squared shape is algebraically near-exact, so tight tolerances
are appropriate.
"""

import numpy as np
import pytest

from synthetic_validation.harness import run_direct_psd_from_modal


# ---------------------------------------------------------------------------
# Oracle correctness test
# ---------------------------------------------------------------------------

def test_direct_psd_oracle_recovers_autopsd(single_mode_modal):
    """Zero-noise, perfect-prior, single-mode oracle: recovered == truth.

    The direct-PSD path (virtual-reference denoise → NNLS → expand_psd)
    should reconstruct the analytic ground-truth auto-PSD with negligible
    error when the mode shapes are exact and there is no camera noise.
    """
    res = run_direct_psd_from_modal(
        single_mode_modal, single_mode_modal,
        saa_level=1.0,
        fs=2000,
        n_frames=40000,
        seed=0,
        noise_sigma_T=0.0,
        n_modes=1,
    )

    m = res["metrics"]
    # NRMSE must be tight — oracle should achieve near-zero error
    assert m["nrmse"]["SX"] < 0.05, (
        f"NRMSE(SX) = {m['nrmse']['SX']:.4f} >= 0.05"
    )
    # Peak amplitude ratio must be close to 1.0 (no systematic under/over-estimate)
    pr = m["peak_ratio"]["SX"]
    assert 0.9 <= pr <= 1.1, (
        f"peak_ratio(SX) = {pr:.4f} outside [0.9, 1.1]"
    )


# ---------------------------------------------------------------------------
# Non-negativity test
# ---------------------------------------------------------------------------

def test_direct_psd_recovered_nonnegative(single_mode_modal):
    """All recovered auto-PSDs must be non-negative (physical requirement).

    NNLS guarantees non-negative beta; the expansion is beta @ |Psi|^2.T
    with non-negative |Psi|^2, so the result must be non-negative everywhere.
    """
    res = run_direct_psd_from_modal(
        single_mode_modal, single_mode_modal,
        saa_level=1.0,
        fs=2000,
        n_frames=8000,
        seed=42,
        noise_sigma_T=0.0,
        n_modes=1,
    )
    for comp, arr in res["recovered"].items():
        assert np.all(arr >= -1e-30), (
            f"recovered['{comp}'] has negative values (min={arr.min():.2e})"
        )


# ---------------------------------------------------------------------------
# Return-dict shape test
# ---------------------------------------------------------------------------

def test_direct_psd_return_dict_keys(single_mode_modal):
    """Return dict must contain all required keys with correct array shapes."""
    res = run_direct_psd_from_modal(
        single_mode_modal, single_mode_modal,
        saa_level=1.0,
        fs=2000,
        n_frames=8000,
        seed=7,
        noise_sigma_T=0.0,
        n_modes=1,
    )
    assert set(res.keys()) >= {"recovered", "truth", "metrics", "freqs", "condition_number"}
    nfreq = len(res["freqs"])
    nnodes = single_mode_modal["node_coords"].shape[0]
    for comp in ("SX", "SY", "SXY", "SX+SY"):
        assert res["recovered"][comp].shape == (nfreq, nnodes)
        assert res["truth"][comp].shape    == (nfreq, nnodes)
    assert isinstance(res["condition_number"], float)
