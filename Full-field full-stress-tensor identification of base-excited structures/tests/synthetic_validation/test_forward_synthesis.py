import numpy as np
from scipy.signal import welch
from synthetic_validation.forward_model import synthesize_base_excitation, base_excitation_frf

def test_synth_accel_psd_matches_level(single_mode_modal):
    rng = np.random.default_rng(0)
    out = synthesize_base_excitation(single_mode_modal, fs=2000, n_frames=20000,
                                     saa_level=2.0, rng=rng,
                                     camera_to_stress_factor=-3.6e8, grid_shape=(2,2))
    f, Saa = welch(out["accel"], fs=2000, nperseg=2048)
    band = (f > 45) & (f < 65)
    assert 1.0 < np.median(Saa[band]) < 4.0            # ~ target level 2.0

def test_cam_frames_shape(single_mode_modal):
    rng = np.random.default_rng(1)
    out = synthesize_base_excitation(single_mode_modal, fs=2000, n_frames=4096,
                                     saa_level=1.0, rng=rng,
                                     camera_to_stress_factor=-3.6e8, grid_shape=(2,2))
    assert out["cam_frames"].shape == (4096, 2, 2)
