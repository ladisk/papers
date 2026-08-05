import numpy as np

def inject_camera_noise(frames, sigma_T, rng):
    return np.asarray(frames) + rng.standard_normal(np.shape(frames)) * sigma_T

def inject_accel_noise(accel, sigma_a, rng):
    return np.asarray(accel) + rng.standard_normal(np.shape(accel)) * sigma_a

def estimate_camera_noise_floor(frames, method="hf_temporal"):
    if method != "hf_temporal":
        raise ValueError(f"Unknown method: {method!r}")
    frames = np.asarray(frames)
    # temporal first difference isolates high-frequency (noise-dominated) content;
    # diff of white noise has variance 2*sigma^2 -> divide by sqrt(2)
    d = np.diff(frames, axis=0)
    per_pixel = np.std(d.reshape(d.shape[0], -1), axis=0) / np.sqrt(2.0)
    return float(np.median(per_pixel))

def estimate_accel_noise_floor(accel, fs, quiet_band=(700, 900)):
    from scipy.signal import welch
    f, S = welch(np.asarray(accel), fs=fs, nperseg=min(4096, len(accel)))
    band = (f >= quiet_band[0]) & (f <= quiet_band[1])
    if not band.any():
        raise ValueError(f"quiet_band {quiet_band} contains no Welch bins (fs={fs}); check band vs Nyquist")
    df = f[1] - f[0]
    return float(np.sqrt(np.sum(S[band]) * df))
