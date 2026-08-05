import numpy as np

def base_excitation_frf(modal_data, freqs_hz):
    w = 2*np.pi*np.asarray(freqs_hz, float)
    wr = modal_data["modal_omega"]; z = modal_data["zeta"]; g = modal_data["gamma_base"]
    D = wr[None, :]**2 - w[:, None]**2 + 2j*z[None, :]*wr[None, :]*w[:, None]  # (nfreq, nmodes)
    q = -g[None, :] / D                                                       # (nfreq, nmodes)
    out = {}
    for key, arr in (("SX", modal_data["modal_sx"]), ("SY", modal_data["modal_sy"]),
                     ("SXY", modal_data["modal_sxy"])):
        out[key] = q @ arr                                                    # (nfreq, nnodes)
    out["SX+SY"] = out["SX"] + out["SY"]
    return out

def truth_component_psd(modal_data, freqs_hz, S_aa):
    T = base_excitation_frf(modal_data, freqs_hz); S_aa = np.asarray(S_aa, float)
    return {k: (np.abs(v)**2) * S_aa[:, None] for k, v in T.items()}

def _realize_base_accel(fs, n_frames, saa_level, rng):
    # white acceleration whose one-sided (Welch) PSD ~ saa_level over the band
    # variance = saa_level * (fs/2); generate white noise scaled to that variance
    std = np.sqrt(saa_level * (fs / 2.0))
    return rng.standard_normal(n_frames) * std

def synthesize_base_excitation(modal_data, fs, n_frames, saa_level, rng,
                               camera_to_stress_factor, grid_shape):
    a = _realize_base_accel(fs, n_frames, saa_level, rng)
    freqs = np.fft.rfftfreq(n_frames, d=1.0/fs)
    A = np.fft.rfft(a)
    w = 2*np.pi*freqs
    wr = modal_data["modal_omega"]; z = modal_data["zeta"]; g = modal_data["gamma_base"]
    D = wr[None, :]**2 - w[:, None]**2 + 2j*z[None, :]*wr[None, :]*w[:, None]
    q = (-g[None, :] / D) * A[:, None]                       # modal coord spectra (nfreq, nmodes)
    inv = q @ modal_data["modal_sx"] + q @ modal_data["modal_sy"]   # SX+SY spectrum (nfreq, nnodes)
    sigma = np.fft.irfft(inv, n=n_frames, axis=0)           # (n_frames, nnodes) stress [Pa]
    cam = sigma / camera_to_stress_factor                   # temperature [K]
    nrows, ncols = grid_shape
    return {"cam_frames": cam.reshape(n_frames, nrows, ncols),
            "accel": a, "S_aa_target": saa_level}
