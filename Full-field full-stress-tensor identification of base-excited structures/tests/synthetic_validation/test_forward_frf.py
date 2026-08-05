import numpy as np
from synthetic_validation.forward_model import base_excitation_frf, truth_component_psd

def test_single_mode_frf_matches_closed_form(single_mode_modal):
    md = single_mode_modal
    freqs = np.linspace(40, 70, 200)
    T = base_excitation_frf(md, freqs)
    w = 2*np.pi*freqs; wr = md["modal_omega"][0]; z = md["zeta"][0]
    D = wr**2 - w**2 + 2j*z*wr*w
    expect_sx = -md["gamma_base"][0] * md["modal_sx"][0][None, :] / D[:, None]
    assert np.allclose(T["SX"], expect_sx, rtol=1e-10)
    # invariant is the sum
    assert np.allclose(T["SX+SY"], T["SX"] + T["SY"], rtol=1e-12)

def test_psd_is_abs2_times_saa(single_mode_modal):
    md = single_mode_modal
    freqs = np.linspace(40, 70, 50); S_aa = np.ones_like(freqs) * 3.0
    T = base_excitation_frf(md, freqs); S = truth_component_psd(md, freqs, S_aa)
    assert np.allclose(S["SXY"], np.abs(T["SXY"])**2 * S_aa[:, None])
