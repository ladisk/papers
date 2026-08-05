import numpy as np
from synthetic_validation.metrics import nrmse, mac, condition_number, point_error, component_ratio_error, rel_rms_error, peak_ratio


def test_rel_rms_error_exposes_proportional_bias():
    t = np.array([[0.0, 1.0, 10.0, 1.0, 0.0]])   # peaked "PSD"
    # a 30% global underestimate -> rel_rms == 0.30 (nrmse-by-range would hide it)
    assert abs(rel_rms_error(0.7 * t, t) - 0.30) < 1e-9
    assert rel_rms_error(t, t) == 0.0


def test_peak_ratio_at_resonance():
    t = np.array([[0.0, 0.0], [10.0, 8.0], [0.0, 0.0]])   # (nfreq=3, nnodes=2), peak at row 1
    assert abs(peak_ratio(0.5 * t, t) - 0.5) < 1e-12
    assert abs(peak_ratio(t, t) - 1.0) < 1e-12

def test_nrmse_zero_for_identical():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert nrmse(x, x) == 0.0

def test_mac_bounds():
    a = np.array([1.0, 2.0, 3.0]); b = 2.5 * a
    assert abs(mac(a, b) - 1.0) < 1e-12          # colinear -> 1
    c = np.array([1.0, 0.0, 0.0]); d = np.array([0.0, 1.0, 0.0])
    assert abs(mac(c, d)) < 1e-12                # orthogonal -> 0

def test_condition_number_identity():
    assert abs(condition_number(np.eye(3)) - 1.0) < 1e-12

def test_point_error_zero_for_identical():
    t = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(point_error(t, t, [0, 2]), [0.0, 0.0])

def test_component_ratio_error_zero_for_identical():
    d = {"SX": np.ones(4), "SY": np.ones(4) * 0.5, "SXY": np.ones(4) * 0.25}
    assert component_ratio_error(d, d) == 0.0

def test_nrmse_axis_shape_and_zero():
    t = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    res = nrmse(t.copy(), t, axis=1)
    assert res.shape == (2,)
    np.testing.assert_allclose(res, [0.0, 0.0])

def test_nrmse_axis_value():
    # row range = 4; diff = [0,4] -> rms = sqrt(mean([0,16])) = sqrt(8); nrmse = sqrt(8)/4
    t = np.array([[0.0, 4.0]])
    r = np.array([[0.0, 0.0]])
    np.testing.assert_allclose(nrmse(r, t, axis=1), [np.sqrt(8) / 4])
