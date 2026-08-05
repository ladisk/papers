# tests/synthetic_validation/test_fe_models.py
import numpy as np
from synthetic_validation.fe_models import generate_fe, build_gamma_base
from synthetic_validation.config import load_config

def test_gamma_base_formula():
    modal_uz = np.array([[1.0, 1.0, 1.0, 1.0]])       # 1 mode, 4 nodes
    m = np.array([0.25, 0.25, 0.25, 0.25]); Mi = np.array([1.0])
    g = build_gamma_base(modal_uz, m, Mi)
    assert np.allclose(g, [1.0])                        # sum(m*uz)/Mi = 1.0

def test_generate_fe_uses_cache(tmp_path):
    calls = {"n": 0}
    def fake_runner(cfg, force_label, out_dir):
        calls["n"] += 1
        nn = 4
        return {"stress_tensor_frf": np.zeros((10, nn, 4), complex),
                "freqs": np.linspace(45, 200, 10), "node_coords": np.zeros((nn, 3)),
                "modal_data": {"modal_freqs": np.array([54.0]), "modal_omega": np.array([339.]),
                               "zeta": np.array([0.005]), "node_coords": np.zeros((nn,3)),
                               "modal_sx": np.zeros((1,nn)), "modal_sy": np.zeros((1,nn)),
                               "modal_sxy": np.zeros((1,nn)), "gamma_base": np.array([1.0]),
                               "modal_mass": np.array([1.0])}}
    cfg = load_config("synthetic_validation/configs/parent.json")
    a = generate_fe(cfg, "center_middle", runner=fake_runner, cache_dir=tmp_path)
    b = generate_fe(cfg, "center_middle", runner=fake_runner, cache_dir=tmp_path)
    assert calls["n"] == 1                              # second call hits cache
    assert a["stress_tensor_frf"].shape == b["stress_tensor_frf"].shape
    np.testing.assert_array_equal(a["stress_tensor_frf"], b["stress_tensor_frf"])


def test_build_solver_env_forwards_material_and_geometry():
    from synthetic_validation.fe_models import _build_solver_env
    cfg = load_config("synthetic_validation/configs/parent.json")
    env = _build_solver_env(cfg, "center_middle")
    assert float(env["MAT_E"]) == cfg.E          # 58e9 for parent, distinct from truth's 63e9
    assert float(env["MAT_NU"]) == cfg.nu
    assert float(env["MAT_RHO"]) == cfg.rho
    assert float(env["PLATE_THK"]) == cfg.thickness
    assert int(env["GRID_NX"]) == cfg.grid_nx
