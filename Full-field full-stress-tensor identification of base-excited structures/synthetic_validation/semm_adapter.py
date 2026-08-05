"""SEMM Stage-1 adapter: run the *existing* pyFBS-based SEMM on in-memory FRFs.

This module is a thin, DRY wrapper around the production SEMM path in
``dual_stage_base_pipeline/semm_thermoelastic_pipeline.py``.  It lets the
synthetic-validation harness feed **in-memory** parent + overlay FRF arrays to
the same ``pyFBS.SEMM`` routine the real pipeline uses, without touching disk,
running the FE solver, or reprocessing camera video.

=========================================================================
Investigation findings (read from the existing code before writing this)
=========================================================================

DOF / array layout (parent "numerical" model)
---------------------------------------------
* ``parent_frf`` has shape ``(nfreq, nnodes*4, ncases)`` complex.
* Axis-1 is the full stress tensor interleaved **per node** as
  ``[SX, SY, SXY, SX+SY]``; node ``n`` component ``c`` lives at index ``4*n+c``.
  (See ``run_stage1_numerical_and_semm``: ``stress_tensor.reshape(nfreq, nnodes*4)``
  from a ``(nfreq, nnodes, 4)`` tensor.)
* Axis-2 is the impact/reference index (one column per hammer case).

The overlay ("experimental"/camera) model
------------------------------------------
* The camera measures a single scalar per pixel that maps to the **SX+SY**
  invariant, i.e. component ``c = 3`` (``component_overlay = 3`` in the engine).
* ``overlay_frf`` here is that SX+SY FRF already sampled at the interface nodes,
  shape ``(nfreq, npixels, ncases)``.  In the production engine
  ``_build_overlay_interface`` maps camera pixels to nearest FE nodes and averages
  them onto the component-3 DOF of each interface node; this adapter assumes the
  overlay is *already* node-aligned (one channel per interface node), so
  ``npixels == nnodes`` and the interface is every node's SX+SY DOF.

The four bookkeeping DataFrames (what pyFBS.SEMM matches on)
-----------------------------------------------------------
``pyFBS.SEMM(Y_num, Y_exp, df_chn_num, df_imp_num, df_chn_exp, df_imp_exp, ...)``
uses the DataFrames *only* to partition DOFs into the boundary (B) set: it matches
rows of the num vs exp frames by exact equality of the six columns
``[Position_1..3, Direction_1..3]`` (rounded to 6 decimals) via
``find_locations_in_data_frames``.  The physical coordinate *values* never enter
the SEMM algebra -- only the resulting index partition does.  Therefore the
adapter only needs positions/directions that are (a) **unique per DOF** and
(b) **identical between the parent frame and the overlay subset** so each overlay
channel matches exactly one parent DOF.

* ``df_parent`` (df_chn_num): built by ``_build_df_node_coords`` -- each node
  repeated 4x with the direction pattern
  ``[[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0]]`` -> ``nnodes*4`` rows, one per parent DOF.
* ``df_overlay`` (df_chn_exp): the component-3 subset ``df_parent.iloc[3::4]``
  with directions forced to ``[0.5,0.5,0]`` -> ``nnodes`` rows, matching the
  overlay channels one-to-one.
* ``df_imp_parent`` == ``df_imp_overlay``: one row per case, distinct positions,
  direction ``[0,0,1]`` -> ``ncases`` rows.  Parent and overlay share the same
  impact frame so every impact is a boundary impact (matched one-to-one).

DEVIATION from the engine (documented):
The characterization contract passes ``node_coords`` that may be degenerate
(e.g. all-zeros in the reference test).  Degenerate positions collapse the
per-DOF matching (many overlay rows would match many parent rows and
``find_locations_in_data_frames`` would raise "Not all locations ..."), so this
adapter synthesizes **index-unique** synthetic node positions whenever the
supplied ``node_coords`` are not all-distinct.  This is safe because, as noted
above, SEMM ignores the coordinate values themselves.

SEMM invariant (why the characterization test holds)
-----------------------------------------------------
When the overlay equals the parent's own SX+SY invariant, the "removed" model
``Y_rem`` and the overlay ``Y_ov`` are identical, so ``(Y_rem - Y_ov) == 0`` and
SEMM returns the parent exactly (rel-diff 0).  This pins the interface.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Optional

import numpy as np

# The overlay component in the [SX, SY, SXY, SX+SY] interleave is SX+SY.
_COMPONENT_OVERLAY = 3


def _import_engine():
    """Import the production pipeline module, adding the repo root to sys.path
    if it is not already importable (tests run from the repo root)."""
    try:
        from dual_stage_base_pipeline.semm_thermoelastic_pipeline import (  # type: ignore
            SEMMConfig,
            SEMMThermoelasticPipeline,
        )
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from dual_stage_base_pipeline.semm_thermoelastic_pipeline import (  # type: ignore
            SEMMConfig,
            SEMMThermoelasticPipeline,
        )
    return SEMMThermoelasticPipeline, SEMMConfig


def _resolve_cfg(cfg: Any, freq_axis: np.ndarray, SEMMConfig):
    """Return a SEMMConfig. ``None`` -> a config whose band covers all of
    ``freq_axis``. A SEMMConfig is used as-is. A dict is splatted into SEMMConfig."""
    if cfg is None:
        pad = 1e-6
        return SEMMConfig(
            f_semm_min=float(np.min(freq_axis)) - pad,
            f_semm_max=float(np.max(freq_axis)) + pad,
        )
    if isinstance(cfg, SEMMConfig):
        return cfg
    if isinstance(cfg, dict):
        return SEMMConfig(**cfg)
    # Duck-typed object exposing the needed attributes.
    if all(hasattr(cfg, a) for a in ("f_semm_min", "f_semm_max", "semm_type")):
        return cfg
    raise TypeError(
        f"cfg must be None, a SEMMConfig, a dict, or expose SEMM band attrs; got {type(cfg)!r}"
    )


def run_semm_stage1(
    parent_frf: np.ndarray,
    overlay_frf: np.ndarray,
    node_coords: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Optional[Any] = None,
    *,
    label: str = "Stage-1 SEMM (synthetic)",
    verbose: bool = False,
) -> np.ndarray:
    """Run the existing pyFBS SEMM on in-memory synthetic FRFs.

    Parameters
    ----------
    parent_frf : (nfreq, nnodes*4, ncases) complex
        Parent (FE) full stress-tensor FRF, node-interleaved ``[SX,SY,SXY,SX+SY]``.
    overlay_frf : (nfreq, nnodes, ncases) complex
        Overlay SX+SY FRF sampled at the interface nodes (one channel per node).
        ``npixels`` must equal ``nnodes`` -- the overlay is assumed pre-mapped to
        nodes (this adapter does not do camera-pixel -> node nearest-neighbour
        mapping; the production engine's ``_build_overlay_interface`` does that).
    node_coords : (nnodes, 3)
        Node coordinates. Only used to derive the DOF-matching frames; the values
        are irrelevant to the SEMM algebra, so degenerate coords are replaced by
        index-unique synthetic positions (see module docstring).
    freq_axis : (nfreq,)
        Frequency of each line, in Hz.
    cfg : None | SEMMConfig | dict
        SEMM configuration. ``None`` runs SEMM over the entire ``freq_axis``.
        Pass a SEMMConfig with a narrow ``[f_semm_min, f_semm_max]`` to restrict
        the band (lines outside the band are returned as zeros).

    Returns
    -------
    Y_SEMM : (nfreq, nnodes*4, ncases) complex
        Hybrid model. Lines outside the SEMM band are zero.
    """
    SEMMThermoelasticPipeline, SEMMConfig = _import_engine()

    parent_frf = np.asarray(parent_frf)
    overlay_frf = np.asarray(overlay_frf)
    freq_axis = np.asarray(freq_axis, dtype=float).reshape(-1)

    if parent_frf.ndim != 3:
        raise ValueError(f"parent_frf must be 3D (nfreq, nnodes*4, ncases); got {parent_frf.shape}")
    if overlay_frf.ndim != 3:
        raise ValueError(f"overlay_frf must be 3D (nfreq, nnodes, ncases); got {overlay_frf.shape}")

    nfreq, ndof, ncases = parent_frf.shape
    if ndof % 4 != 0:
        raise ValueError(f"parent_frf axis-1 ({ndof}) must be a multiple of 4 (4 components/node)")
    nnodes = ndof // 4

    if overlay_frf.shape[0] != nfreq:
        raise ValueError(
            f"Frequency mismatch: parent={nfreq}, overlay={overlay_frf.shape[0]}"
        )
    if overlay_frf.shape[2] != ncases:
        raise ValueError(
            f"Case mismatch: parent={ncases}, overlay={overlay_frf.shape[2]}"
        )
    if overlay_frf.shape[1] != nnodes:
        raise ValueError(
            "Overlay must be node-aligned: expected npixels == nnodes "
            f"({nnodes}); got {overlay_frf.shape[1]}. Pre-map camera pixels to "
            "nodes before calling run_semm_stage1."
        )
    if len(freq_axis) != nfreq:
        raise ValueError(f"freq_axis length {len(freq_axis)} != nfreq {nfreq}")

    # --- Build DOF-matching frames (reusing the engine's static builders). ---
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.shape != (nnodes, 3) or np.unique(node_coords, axis=0).shape[0] != nnodes:
        # Degenerate / mismatched coords -> synthesize index-unique positions.
        # Values are irrelevant to SEMM; uniqueness makes DOF matching bijective.
        node_coords = np.column_stack(
            [np.arange(nnodes, dtype=float) * 1.0e-3,
             np.zeros(nnodes), np.zeros(nnodes)]
        )

    pipe = SEMMThermoelasticPipeline()

    df_parent, _ = pipe._build_df_node_coords(node_coords)
    b_dof_idx = 4 * np.arange(nnodes) + _COMPONENT_OVERLAY
    df_overlay = df_parent.iloc[b_dof_idx].copy().reset_index(drop=True)
    df_overlay[["Direction_1", "Direction_2", "Direction_3"]] = [0.5, 0.5, 0.0]

    case_labels = [f"case_{i}" for i in range(ncases)]
    force_points = {lab: (float(i), 0.0) for i, lab in enumerate(case_labels)}
    df_imp = pipe._build_df_imp(case_labels, force_points)

    scfg = _resolve_cfg(cfg, freq_axis, SEMMConfig)

    # --- Run the exact production SEMM loop on the in-memory arrays. ---
    if verbose:
        return pipe._run_chunked_semm(
            parent_frf, overlay_frf, df_parent, df_imp, df_overlay, df_imp,
            freq_axis, scfg, label,
        )
    with redirect_stdout(io.StringIO()):
        return pipe._run_chunked_semm(
            parent_frf, overlay_frf, df_parent, df_imp, df_overlay, df_imp,
            freq_axis, scfg, label,
        )
