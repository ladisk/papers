"""Tests for synthetic_validation.studies — analytic path only (no ANSYS/pyFBS).

All tests use the `single_mode_modal` fixture from conftest.py so they run
without MAPDL or pyFBS installed.  Keep n_frames small to stay fast.
"""
import numpy as np
import pytest

from synthetic_validation.studies import (
    study_a_recovery,
    study_b_noise,
    study_c_fe_discrepancy,
    study_d_modal_convergence,
    study_e_conditioning,
)


# ---------------------------------------------------------------------------
# Study B — noise confidence bands
# ---------------------------------------------------------------------------

def test_study_b_returns_bands(single_mode_modal):
    res = study_b_noise(
        single_mode_modal, single_mode_modal,
        n_reps=5, snr_levels=[1.0],
        analytic=True, fs=2000, n_frames=8000,
        saa_level=1.0, n_modes=1,
    )
    assert "confidence_bands" in res
    assert "SX" in res["confidence_bands"]
    assert res["n_reps"] == 5


def test_study_b_bands_have_mean_and_std(single_mode_modal):
    """Each component/snr entry must expose 'mean' and 'std' arrays."""
    res = study_b_noise(
        single_mode_modal, single_mode_modal,
        n_reps=3, snr_levels=[2.0],
        analytic=True, fs=2000, n_frames=4000,
        saa_level=1.0, n_modes=1,
    )
    band = res["confidence_bands"]["SX"][2.0]
    assert "mean" in band and "std" in band
    assert band["mean"].shape == band["std"].shape
    assert band["mean"].ndim == 2           # (nfreq, nnodes)


# ---------------------------------------------------------------------------
# Study A — realistic vs oracle recovery
# ---------------------------------------------------------------------------

def test_study_a_returns_realistic_and_oracle(single_mode_modal):
    res = study_a_recovery(
        single_mode_modal, single_mode_modal,
        analytic=True, fs=2000, n_frames=4000,
        saa_level=1.0, seed=0, n_modes=1,
    )
    assert "realistic" in res and "oracle" in res
    for run_key in ("realistic", "oracle"):
        m = res[run_key]["metrics"]
        assert "nrmse" in m
        assert "SX" in m["nrmse"]
    # Both should have spatial maps
    assert "recovered" in res["realistic"]
    assert "SX" in res["realistic"]["recovered"]


# ---------------------------------------------------------------------------
# Study D — modal convergence
# ---------------------------------------------------------------------------

def test_study_d_returns_entry_per_mode_count(single_mode_modal):
    # The fixture has exactly 1 mode, so mode_counts=[1] is the only valid choice.
    res = study_d_modal_convergence(
        single_mode_modal, single_mode_modal,
        mode_counts=[1],
        analytic=True, fs=2000, n_frames=4000,
        saa_level=1.0, seed=0,
    )
    assert "mode_counts" in res
    assert res["mode_counts"] == [1]
    assert "nrmse" in res
    assert "SX" in res["nrmse"]
    assert len(res["nrmse"]["SX"]) == 1


# ---------------------------------------------------------------------------
# Study B — non-degenerate bands (noise actually varies across reps)
# ---------------------------------------------------------------------------

def test_study_b_bands_nondegenerate_with_noise(single_mode_modal):
    """With nonzero noise (snr=1.0 → sigma=0.1 K), different seeds must
    produce different realisations so std > 0 somewhere in the band.
    A same-seed bug would make all reps identical and std==0.
    """
    res = study_b_noise(
        single_mode_modal, single_mode_modal,
        n_reps=5, snr_levels=[1.0],
        analytic=True, fs=2000, n_frames=8000,
        saa_level=1.0, n_modes=1,
    )
    band = res["confidence_bands"]["SX"][1.0]
    assert np.max(band["std"]) > 0, (
        "All reps produced identical PSD — likely a seed-collision bug "
        "(std is zero everywhere)."
    )


# ---------------------------------------------------------------------------
# Study D — length matches mode_counts (including duplicates)
# ---------------------------------------------------------------------------

def test_study_d_length_matches_mode_counts(single_mode_modal):
    """Result arrays must have one entry per element of mode_counts,
    even when the same count appears more than once.
    """
    res = study_d_modal_convergence(
        single_mode_modal, single_mode_modal,
        mode_counts=[1, 1],
        analytic=True, fs=2000, n_frames=4000,
        saa_level=1.0, seed=0,
    )
    assert len(res["mode_counts"]) == 2
    assert len(res["nrmse"]["SX"]) == 2
    assert len(res["condition_numbers"]) == 2


# ---------------------------------------------------------------------------
# Study C — analytic smoke test
# ---------------------------------------------------------------------------

def test_study_c_analytic_smoke(single_mode_modal):
    """study_c returns per-variant results and a realistic_oracle_gap entry."""
    res = study_c_fe_discrepancy(
        single_mode_modal,
        [single_mode_modal, single_mode_modal],
        analytic=True, nominal_idx=0,
        fs=2000, n_frames=4000, saa_level=1.0, seed=0, n_modes=1,
    )
    assert "variants" in res
    assert len(res["variants"]) == 2
    assert "realistic_oracle_gap" in res
    assert "SX" in res["realistic_oracle_gap"]


# ---------------------------------------------------------------------------
# Study E — explicit regularize keys
# ---------------------------------------------------------------------------

def test_study_e_regularize_keys(single_mode_modal):
    """study_e must expose regularize_requested, regularize_applied (always
    False), and a finite condition_number.
    """
    res = study_e_conditioning(
        single_mode_modal, single_mode_modal,
        analytic=True, regularize=True,
        fs=2000, n_frames=4000, saa_level=1.0, seed=0, n_modes=1,
    )
    assert res["regularize_requested"] is True
    assert res["regularize_applied"] is False
    assert np.isfinite(res["condition_number"])
