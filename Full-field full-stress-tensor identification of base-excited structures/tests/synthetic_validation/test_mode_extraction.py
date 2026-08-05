import numpy as np
from synthetic_validation.expansion import extract_mode_shapes

def test_extract_single_peak():
    nfreq, nnodes, ncases = 200, 4, 1
    freq = np.linspace(45, 200, nfreq)
    fr = 54.0; w = 2*np.pi*freq; wr = 2*np.pi*fr
    D = wr**2 - w**2 + 2j*0.005*wr*w
    shape_cam = np.array([1.0, 0.8, 0.6, 0.4])
    Y = np.zeros((nfreq, nnodes*4, ncases), complex)
    # component 3 = SX+SY driven by the resonance
    Y[:, 3::4, 0] = shape_cam[None, :] / D[:, None]
    out = extract_mode_shapes(Y, freq, n_modes=1)
    assert abs(out["freqs"][0] - 54.0) < 1.0
    # extracted cam shape is proportional to the true shape
    est = out["psi"]["SX+SY"][:, 0]
    assert abs(abs(np.vdot(est, shape_cam))**2 /
               (np.vdot(est, est).real * shape_cam @ shape_cam) - 1.0) < 1e-6
