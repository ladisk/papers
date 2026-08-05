from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json, hashlib

@dataclass(frozen=True)
class FEConfig:
    E: float; nu: float; rho: float; thickness: float
    plate_lx: float; plate_ly: float; grid_nx: int; grid_ny: int
    point_mass: float; point_mass_xy: tuple[float, float]; base_edge: str
    zeta: float; nmodes: int; fmin: float; fmax: float; nfreq: int
    force_points: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Boundary condition on the base edge:
    #   "clamped" - all 6 DOF fixed (the nominal, and the default everywhere)
    #   "pinned"  - translations fixed, rotations released (a simply-supported edge)
    # "pinned" exists so the prior's CLAMPING STIFFNESS can be degraded on purpose:
    # it is a far more severe boundary error than any real fixture compliance, so it
    # bounds the boundary-condition sensitivity the reviewers asked about.
    # NOTE: config_hash() uses asdict(), so this field participates in the FE cache key.
    base_bc: str = "clamped"

def load_config(path) -> FEConfig:
    d = json.loads(Path(path).read_text())
    d["point_mass_xy"] = tuple(d["point_mass_xy"])
    d["force_points"] = {k: tuple(v) for k, v in d.get("force_points", {}).items()}
    return FEConfig(**d)

def config_hash(cfg: FEConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True)
    return hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()[:16]
