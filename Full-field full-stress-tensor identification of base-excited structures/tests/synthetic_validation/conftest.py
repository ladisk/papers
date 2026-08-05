import numpy as np, pytest

def make_modal_data(freqs_hz, zetas, node_coords, psi_sx, psi_sy, psi_sxy, gamma_base, gen_mass):
    freqs_hz = np.asarray(freqs_hz, float)
    return {
        "modal_freqs": freqs_hz,
        "modal_omega": 2*np.pi*freqs_hz,
        "zeta": np.asarray(zetas, float),
        "node_coords": np.asarray(node_coords, float),
        "modal_sx": np.asarray(psi_sx, float),      # (nmodes, nnodes)
        "modal_sy": np.asarray(psi_sy, float),
        "modal_sxy": np.asarray(psi_sxy, float),
        "gamma_base": np.asarray(gamma_base, float), # (nmodes,)
        "modal_mass": np.asarray(gen_mass, float),   # (nmodes,)
    }

@pytest.fixture
def single_mode_modal():
    # one mode, 4 nodes, unit stress shapes, known participation
    nodes = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], float)
    return make_modal_data(
        freqs_hz=[54.0], zetas=[0.005], node_coords=nodes,
        psi_sx=[[1.0,0.5,0.5,0.2]], psi_sy=[[0.3,0.3,0.3,0.3]],
        psi_sxy=[[0.1,-0.1,0.1,-0.1]], gamma_base=[2.0], gen_mass=[1.0])
