"""Characterization test for the SEMM Stage-1 adapter.

Pins the adapter's interface against the real reference SEMM output. Feeding the
reference's own SX+SY invariant as the overlay and its full tensor as the parent
must return (approximately) the parent, because SEMM with an overlay equal to the
parent's own boundary invariant reduces to the identity (Y_rem - Y_ov == 0).

Band restriction (documented deviation from the brief's cfg=None snippet):
The full 369-line reference run is ~0.8 s/line (~5 min). The brief explicitly
allows restricting the SEMM band via cfg to a small window and asserting on that
sub-range. We restrict SEMM to a ~25-line window and assert shape over the full
tensor and the relative difference over the SEMM band.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from synthetic_validation.semm_adapter import run_semm_stage1

REPO_ROOT = Path(__file__).resolve().parents[2]
REF = REPO_ROOT / "dual_stage_base_pipeline/outputs/20260219_090812/stage1/Y_SEMM_stage1.npy"

# Import SEMMConfig from the production engine to drive the band restriction.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from dual_stage_base_pipeline.semm_thermoelastic_pipeline import SEMMConfig  # noqa: E402


@pytest.mark.skipif(not REF.exists(), reason="reference SEMM output not present")
def test_semm_shape_and_interface_consistency():
    Y_ref = np.load(REF)  # (369, 3364, 6)
    nfreq, ndof, ncases = Y_ref.shape
    nnodes = ndof // 4

    parent = Y_ref.copy()
    overlay = Y_ref[:, 3::4, :]  # SX+SY as pseudo-camera overlay, one per node
    freq_axis = np.linspace(45.0, 300.0, nfreq)

    # Restrict SEMM to a small band (~25 lines) to keep the test fast.
    lo = 0
    hi = 25
    cfg = SEMMConfig(
        f_semm_min=float(freq_axis[lo]) - 1e-6,
        f_semm_max=float(freq_axis[hi]) + 1e-6,
    )

    Y = run_semm_stage1(
        parent, overlay,
        node_coords=np.zeros((nnodes, 3)),
        freq_axis=freq_axis,
        cfg=cfg,
    )

    # Shape must match the reference full tensor.
    assert Y.shape == Y_ref.shape

    # Over the SEMM band, the hybrid must stay close to the parent.
    band = (freq_axis >= cfg.f_semm_min) & (freq_axis <= cfg.f_semm_max)
    assert band.sum() > 0
    rel = np.linalg.norm(Y[band] - parent[band]) / np.linalg.norm(parent[band])
    assert rel < 0.25, f"relative difference {rel:.3e} exceeds 0.25"


@pytest.mark.skipif(not REF.exists(), reason="reference SEMM output not present")
def test_semm_overlay_drives_correction_not_a_noop():
    """Non-identity check: a *perturbed* overlay must actually change the SEMM output.

    Motivation: the identity test (overlay == parent's own SX+SY) returns the parent
    exactly (rel-diff 0). A no-op adapter that simply echoed the parent and ignored the
    overlay would ALSO pass that test. Here we feed an overlay whose SX+SY invariant is
    the parent's own SX+SY scaled by a known factor (2.0) -- representing a "measurement"
    that disagrees with the parent FE -- and prove the correction (i) is non-trivial and
    (ii) moves the recovered invariant toward the overlay.

    Assertions (over the SEMM-computed in-band lines only):
      (a) Not a no-op:      ||Y_band - parent_band|| / ||parent_band|| > 0.01
      (b) Toward the overlay: at the interface SX+SY DOFs (``[:, 3::4, :]``), the recovered
          invariant is closer to the scaled overlay than the parent's own SX+SY is --
          ||Y_inv_band - overlay_band|| < ||parent_inv_band - overlay_band||.

    (b) is a robust *inequality*, not an exact-value check: SEMM's fully-extend-svd
    correction is not a clean scaling, so we only require the recovered interface invariant
    to be pulled toward the overlay relative to the (uncorrected) parent -- which is exactly
    what distinguishes a correct wiring from a swapped/ignored-overlay bug.
    """
    Y_ref = np.load(REF)  # (369, 3364, 6)
    nfreq, ndof, ncases = Y_ref.shape
    nnodes = ndof // 4

    parent = Y_ref.copy()
    # Perturbed overlay: parent's SX+SY invariant scaled by a known factor.
    scale = 2.0
    overlay_scaled = Y_ref[:, 3::4, :] * scale
    freq_axis = np.linspace(45.0, 300.0, nfreq)

    # Restrict SEMM to the same small band as the identity test to keep it fast.
    lo = 0
    hi = 25
    cfg = SEMMConfig(
        f_semm_min=float(freq_axis[lo]) - 1e-6,
        f_semm_max=float(freq_axis[hi]) + 1e-6,
    )

    Y = run_semm_stage1(
        parent, overlay_scaled,
        node_coords=np.zeros((nnodes, 3)),
        freq_axis=freq_axis,
        cfg=cfg,
    )

    assert Y.shape == Y_ref.shape

    band = (freq_axis >= cfg.f_semm_min) & (freq_axis <= cfg.f_semm_max)
    assert band.sum() > 0

    # (a) Not a no-op: the perturbed overlay must change the output vs the parent.
    rel = np.linalg.norm(Y[band] - parent[band]) / np.linalg.norm(parent[band])
    assert rel > 0.01, f"result barely differs from parent (rel={rel:.3e}); overlay ignored?"

    # (b) Correction moves the interface invariant toward the overlay.
    Y_inv = Y[:, 3::4, :]
    parent_inv = parent[:, 3::4, :]
    dist_result = np.linalg.norm(Y_inv[band] - overlay_scaled[band])
    dist_parent = np.linalg.norm(parent_inv[band] - overlay_scaled[band])
    assert dist_result < dist_parent, (
        f"recovered interface invariant not pulled toward overlay: "
        f"||Y_inv - overlay||={dist_result:.6e} !< ||parent_inv - overlay||={dist_parent:.6e}"
    )
