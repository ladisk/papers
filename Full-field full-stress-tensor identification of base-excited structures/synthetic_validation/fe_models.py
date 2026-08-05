"""
fe_models.py — config->MAPDL wrapper with caching + base-participation factor

Public API
----------
build_gamma_base(modal_uz, lumped_mass, gen_mass) -> np.ndarray
    Γ_r = Σ_n m_n · φ_{r,n,z} / M_r

generate_fe(cfg, force_label, runner=default_mapdl_runner, cache_dir=...) -> dict
    Check cache; call runner if miss; write cache; return dict.

default_mapdl_runner(cfg, force_label, out_dir) -> dict
    Set env vars from cfg, invoke solver via subprocess, load outputs.

Return-dict contract (cross-task keys)
---------------------------------------
{
    "stress_tensor_frf": (nfreq, nnodes, 4) complex  [Pa/N],
    "freqs":             (nfreq,),
    "node_coords":       (nnodes, 3),
    "modal_data": {
        "modal_freqs":  (nmodes,),
        "modal_omega":  (nmodes,),
        "zeta":         (nmodes,),
        "node_coords":  (nnodes, 3),
        "modal_sx":     (nmodes, nnodes),
        "modal_sy":     (nmodes, nnodes),
        "modal_sxy":    (nmodes, nnodes),
        "gamma_base":   (nmodes,),
        "modal_mass":   (nmodes,),
    }
}
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from synthetic_validation.config import FEConfig, config_hash

# Default cache directory: synthetic_validation/cache/ (already .gitignored)
_DEFAULT_CACHE = Path(__file__).parent / "cache"

# Solver script location (repo-root-relative)
_SOLVER_SCRIPT = (
    Path(__file__).parent.parent
    / "stage1_solver"
    / "plate_stress_modal_superposition_force_pointmass_v2_camfreq_20260206_1239.py"
)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def build_gamma_base(
    modal_uz: np.ndarray,
    lumped_mass: np.ndarray,
    gen_mass: np.ndarray,
) -> np.ndarray:
    """Base-participation factor for each mode.

    Γ_r = Σ_n (m_n · φ_{r,n,z}) / M_r

    Parameters
    ----------
    modal_uz : (nmodes, nnodes) – z-displacement mode shapes (mass-normalised)
    lumped_mass : (nnodes,) – per-node lumped mass [kg]
    gen_mass : (nmodes,) – generalized (modal) mass M_r [kg]

    Returns
    -------
    gamma : (nmodes,)
    """
    modal_uz = np.asarray(modal_uz, float)   # (nmodes, nnodes)
    lumped_mass = np.asarray(lumped_mass, float)  # (nnodes,)
    gen_mass = np.asarray(gen_mass, float)   # (nmodes,)
    # broadcast lumped_mass over modes: (nmodes, nnodes) * (1, nnodes)
    numerator = (modal_uz * lumped_mass[np.newaxis, :]).sum(axis=1)  # (nmodes,)
    return numerator / gen_mass


# ---------------------------------------------------------------------------
# Cache helpers (flat npz with "md__" prefix for modal_data sub-keys)
# ---------------------------------------------------------------------------

_MD_PREFIX = "md__"
_MD_PREFIX_LEN = len(_MD_PREFIX)


def _cache_path(cache_dir: Path, cfg: FEConfig, force_label: str) -> Path:
    key = config_hash(cfg) + "_" + force_label
    return Path(cache_dir) / (key + ".npz")


def _save_cache(path: Path, d: dict) -> None:
    """Flatten the return dict and write to *path*.npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    md = d["modal_data"]
    np.savez(
        str(path),
        # top-level arrays
        stress_tensor_frf=d["stress_tensor_frf"],
        freqs=d["freqs"],
        node_coords=d["node_coords"],
        # modal_data sub-keys prefixed with "md__"
        md__modal_freqs=md["modal_freqs"],
        md__modal_omega=md["modal_omega"],
        md__zeta=md["zeta"],
        md__node_coords=md["node_coords"],
        md__modal_sx=md["modal_sx"],
        md__modal_sy=md["modal_sy"],
        md__modal_sxy=md["modal_sxy"],
        md__gamma_base=md["gamma_base"],
        md__modal_mass=md["modal_mass"],
    )


def _load_cache(path: Path) -> dict:
    """Reconstruct the return dict from a cached .npz file."""
    with np.load(str(path)) as raw:
        modal_data = {
            k[_MD_PREFIX_LEN:]: raw[k]
            for k in raw.files
            if k.startswith(_MD_PREFIX)
        }
        return {
            "stress_tensor_frf": raw["stress_tensor_frf"],
            "freqs": raw["freqs"],
            "node_coords": raw["node_coords"],
            "modal_data": modal_data,
        }


# ---------------------------------------------------------------------------
# MAPDL runner
# ---------------------------------------------------------------------------

def _build_solver_env(cfg: FEConfig, force_label: str) -> dict:
    """Return a dict of str->str env vars for the solver subprocess.

    Includes geometry/mesh, point-mass, force location, and material properties.
    Intended to be merged over ``os.environ`` by the caller.
    """
    fx, fy = cfg.force_points[force_label]
    return {
        "FORCE_X": str(fx),
        "FORCE_Y": str(fy),
        "GRID_NX": str(cfg.grid_nx),
        "GRID_NY": str(cfg.grid_ny),
        "PLATE_LX": str(cfg.plate_lx),
        "PLATE_LY": str(cfg.plate_ly),
        "POINT_MASS": str(cfg.point_mass),
        "POINT_MASS_X": str(cfg.point_mass_xy[0]),
        "POINT_MASS_Y": str(cfg.point_mass_xy[1]),
        "MAT_E": str(cfg.E),
        "MAT_NU": str(cfg.nu),
        "MAT_RHO": str(cfg.rho),
        "PLATE_THK": str(cfg.thickness),
        "BASE_BC": str(cfg.base_bc),
    }


def default_mapdl_runner(cfg: FEConfig, force_label: str, out_dir) -> dict:
    """Invoke the MAPDL solver via subprocess and return the cross-task dict.

    Environment variables set from *cfg*:
        FORCE_X, FORCE_Y          – from cfg.force_points[force_label]
        GRID_NX, GRID_NY          – mesh density
        PLATE_LX, PLATE_LY        – plate dimensions
        PLATE_THK                 – plate thickness [m]
        POINT_MASS                – added point mass [kg]
        POINT_MASS_X, POINT_MASS_Y – point mass location
        MAT_E, MAT_NU, MAT_RHO   – material properties
    Any FREQS_FILE already set in the parent environment is forwarded as-is.

    After the solver exits, loads:
        stress_tensor_frf.npy, freqs.npy, node_coords.npy, modal_data.npz

    Computes gamma_base from modal_data.npz arrays (modal_uz, lumped_mass,
    modal_mass) that the solver now saves (Step 3 of Task 9).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Force UTF-8 stdout in the child so the solver's unicode prints (e.g. its
    # final "done" banner) don't crash on Windows' cp1252 console encoding.
    env = {**os.environ, **_build_solver_env(cfg, force_label), "PYTHONIOENCODING": "utf-8"}

    subprocess.run(
        [sys.executable, str(_SOLVER_SCRIPT)],
        env=env,
        cwd=str(out_dir),
        check=True,
    )

    # Load solver outputs
    stress_tensor_frf = np.load(str(out_dir / "stress_tensor_frf.npy"))
    freqs = np.load(str(out_dir / "freqs.npy"))
    node_coords = np.load(str(out_dir / "node_coords.npy"))
    md_raw = np.load(str(out_dir / "modal_data.npz"))

    # Compute base-participation factor (requires arrays added in Step 3)
    gamma_base = build_gamma_base(
        md_raw["modal_uz"],
        md_raw["lumped_mass"],
        md_raw["modal_mass"],
    )

    # zeta stored as scalar in npz; broadcast to (nmodes,)
    nmodes = int(md_raw["modal_freqs"].shape[0])
    zeta_arr = np.full(nmodes, float(md_raw["zeta"]))

    modal_data = {
        "modal_freqs": md_raw["modal_freqs"],
        "modal_omega": md_raw["modal_omega"],
        "zeta": zeta_arr,
        "node_coords": node_coords,
        "modal_sx": md_raw["modal_sx"],
        "modal_sy": md_raw["modal_sy"],
        "modal_sxy": md_raw["modal_sxy"],
        "gamma_base": gamma_base,
        "modal_mass": md_raw["modal_mass"],
    }

    return {
        "stress_tensor_frf": stress_tensor_frf,
        "freqs": freqs,
        "node_coords": node_coords,
        "modal_data": modal_data,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_fe(
    cfg: FEConfig,
    force_label: str,
    runner=None,
    cache_dir=_DEFAULT_CACHE,
) -> dict:
    """Return FE model outputs for *cfg* / *force_label*, using cache if available.

    On a cache miss, *runner* is called with ``(cfg, force_label, tmp_dir)``
    where *tmp_dir* is a fresh temporary directory that the runner may use
    as its working directory.  The returned dict is written to the cache before
    being returned.

    Parameters
    ----------
    cfg : FEConfig
    force_label : str
        Key in ``cfg.force_points``.
    runner : callable | None
        ``runner(cfg, force_label, out_dir) -> dict``.
        Defaults to :func:`default_mapdl_runner`.
    cache_dir : path-like
        Directory for ``.npz`` cache files.
        Default: ``synthetic_validation/cache/`` (already .gitignored).

    Returns
    -------
    dict
        Keys: ``stress_tensor_frf``, ``freqs``, ``node_coords``, ``modal_data``.
    """
    if runner is None:
        runner = default_mapdl_runner

    cp = _cache_path(Path(cache_dir), cfg, force_label)

    if cp.exists():
        return _load_cache(cp)

    with tempfile.TemporaryDirectory() as tmp:
        d = runner(cfg, force_label, tmp)

    _save_cache(cp, d)
    return d
