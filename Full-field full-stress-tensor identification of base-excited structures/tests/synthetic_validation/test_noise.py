import numpy as np
from synthetic_validation.noise import inject_camera_noise, estimate_camera_noise_floor, estimate_accel_noise_floor

def test_injection_reproduces_level():
    rng = np.random.default_rng(0)
    frames = np.zeros((5000, 4, 4))
    noisy = inject_camera_noise(frames, sigma_T=0.002, rng=rng)
    assert abs(np.std(noisy) - 0.002) < 1e-4

def test_estimate_recovers_known_floor():
    rng = np.random.default_rng(2)
    frames = rng.standard_normal((5000, 6, 6)) * 0.003     # pure white noise floor
    est = estimate_camera_noise_floor(frames)
    assert 0.002 < est < 0.004

def test_estimate_accel_floor_recovers_known_std():
    rng = np.random.default_rng(3)
    accel = rng.standard_normal(50000) * 0.01     # white noise, std 0.01
    est = estimate_accel_noise_floor(accel, fs=2000, quiet_band=(1, 999))
    assert 0.009 < est < 0.011
