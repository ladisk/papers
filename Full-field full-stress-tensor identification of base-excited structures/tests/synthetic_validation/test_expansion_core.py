import numpy as np
from synthetic_validation.expansion import modal_decompose, expand_components, stress_psd

def test_decompose_inverts_expansion():
    rng = np.random.default_rng(0)
    nnodes, nmodes, nfreq = 20, 2, 30
    Psi_cam = rng.standard_normal((nnodes, nmodes)) + 1j*rng.standard_normal((nnodes, nmodes))
    gamma_true = rng.standard_normal((nfreq, nmodes)) + 1j*rng.standard_normal((nfreq, nmodes))
    T_cam_nodes = (Psi_cam @ gamma_true.T).T                 # (nfreq, nnodes)
    gamma_est = modal_decompose(Psi_cam, T_cam_nodes)
    assert np.allclose(gamma_est, gamma_true, atol=1e-8)

def test_expand_and_psd():
    gamma = np.ones((3, 1), complex)
    psi = {"SX": np.array([[2.0]]), "SY": np.array([[1.0]]),
           "SXY": np.array([[0.5]]), "SX+SY": np.array([[3.0]])}
    T = expand_components(gamma, psi)
    assert np.allclose(T["SX"], 2.0)
    S = stress_psd(T, S_aa=np.array([4.0, 4.0, 4.0]))
    assert np.allclose(S["SX"], 4.0 * 4.0)                   # |2|^2 * 4
