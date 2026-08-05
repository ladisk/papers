import numpy as np
from synthetic_validation.harness import run_case_from_modal


def test_oracle_zero_noise_recovers_truth(single_mode_modal):
    # parent == truth modal data, no noise, mode shapes taken as exact:
    res = run_case_from_modal(single_mode_modal, single_mode_modal,
                              saa_level=1.0, fs=2000, n_frames=40000, seed=0,
                              noise_sigma_T=0.0, n_modes=1)
    # recovered SX PSD should match truth SX PSD at the resonance band within a few %
    m = res["metrics"]
    assert m["nrmse"]["SX"] < 0.05
    assert m["nrmse"]["SXY"] < 0.10
    assert m["mac"]["SX"] > 0.98


def test_n_segments_changes_frequency_resolution(single_mode_modal):
    from synthetic_validation.harness import run_case_from_modal
    base = run_case_from_modal(single_mode_modal, single_mode_modal, saa_level=1.0,
                               fs=2000, n_frames=20000, seed=0, n_modes=1)
    seg = run_case_from_modal(single_mode_modal, single_mode_modal, saa_level=1.0,
                              fs=2000, n_frames=20000, seed=0, n_modes=1, n_segments=8)
    # multi-segment averaging uses shorter segments -> coarser (shorter) frequency axis
    assert seg["freqs"].shape[0] < base["freqs"].shape[0]
