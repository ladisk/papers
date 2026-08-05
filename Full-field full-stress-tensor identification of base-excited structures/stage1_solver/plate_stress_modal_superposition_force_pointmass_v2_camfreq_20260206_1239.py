"""
plate_stress_frf_modal_force.py

Compute stress FRFs (Stress / Force) using modal superposition
for a plate clamped on one edge and excited by a point force.

ANSYS MAPDL:
  - Modal analysis only (fixed-base modes)

Python:
  - FRF synthesis using classical force-driven modal superposition

Outputs:
  stress_tensor_frf : (nfreq, nnodes, 4) complex  [Pa / N]
  freqs             : (nfreq,)
  node_coords       : (nnodes, 3)
"""

import os
from ansys.mapdl.core import launch_mapdl
import numpy as np


# --- helpers ---
def _env_int(name, default):
    val = os.environ.get(name)
    if val is None:
        return int(default)
    return int(val)


def _env_float(name, default):
    val = os.environ.get(name)
    if val is None:
        return float(default)
    return float(val)


# --- defaults ---
force_node_id = None
force_x = None
force_y = None

# --- point mass defaults ---
# If point_mass is None or <=0, no point mass is added.
point_mass = 0.1   # [kg]
pm_x = 0.075         # [m]
pm_y = 0.14         # [m]
pm_patch_nx = _env_int("POINT_MASS_PATCH_NX", 30)
pm_patch_ny = _env_int("POINT_MASS_PATCH_NY", 5)
pm_node_center_id = None
pm_node_ids = np.array([], dtype=int)

# --- read env ---
if "FORCE_NODE_ID" in os.environ:
    force_node_id = int(os.environ["FORCE_NODE_ID"])

if "FORCE_X" in os.environ and "FORCE_Y" in os.environ:
    force_x = float(os.environ["FORCE_X"])
    force_y = float(os.environ["FORCE_Y"])

# --- read point mass env (optional) ---
# POINT_MASS: mass in kg
# POINT_MASS_X / POINT_MASS_Y: location in meters
if "POINT_MASS" in os.environ:
    point_mass = float(os.environ["POINT_MASS"])

if "POINT_MASS_X" in os.environ and "POINT_MASS_Y" in os.environ:
    pm_x = float(os.environ["POINT_MASS_X"])
    pm_y = float(os.environ["POINT_MASS_Y"])


# ============================================================
# USER INPUTS
# ============================================================

# --- Geometry / mesh ---
# Grid nodes in X/Y (can be overridden from env, e.g. GRID_NX=29, GRID_NY=29).
nx = _env_int("GRID_NX", 29)
ny = _env_int("GRID_NY", 29)

# Plate size in meters (can be overridden from env).
LX = _env_float("PLATE_LX", 0.15)
LY = _env_float("PLATE_LY", 0.15)
THK = _env_float("PLATE_THK", 0.003)

if nx < 2 or ny < 2:
    raise ValueError(f"Invalid grid size nx={nx}, ny={ny}. Both must be >= 2.")
if pm_patch_nx < 1 or pm_patch_ny < 1:
    raise ValueError(
        f"Invalid point-mass patch size POINT_MASS_PATCH_NX={pm_patch_nx}, "
        f"POINT_MASS_PATCH_NY={pm_patch_ny}. Both must be >= 1."
    )

# Default point-mass location: top-left corner of the plate
# (x=0, y=LY) in the coordinate system used for the plate area.
if point_mass is not None and point_mass > 0 and (pm_x is None or pm_y is None):
    pm_x = 0.0
    pm_y = float(LY)

# --- Material ---
E   = _env_float("MAT_E", 63e9)
NU  = _env_float("MAT_NU", 0.33)
RHO = _env_float("MAT_RHO", 2700)

# --- Frequency sweep ---
fmin  = 45.0
fmax  = 200.0
nfreq = 200
# Optional explicit frequency vector file (npy) passed from notebook.
freqs_file = os.environ.get("FREQS_FILE", "").strip()

# --- Modal parameters ---
nmodes = 10
zeta   = 0.005

# --- Boundary condition ---
base_edge = "bottom"   # 'left' | 'right' | 'bottom' | 'top'

# --- Force excitation ---
#force_node_id = None   # set after mesh creation if None
force_dir = "Z"        # 'X', 'Y', 'Z'
force_amp = 1.0        # [N] → unit-force FRF

# --- Output ---
out_dir = os.getcwd()
stress_tensor_outfile = os.path.join(out_dir, "stress_tensor_frf.npy")
freq_outfile          = os.path.join(out_dir, "freqs.npy")
coords_outfile        = os.path.join(out_dir, "node_coords.npy")
modal_data_outfile    = os.path.join(out_dir, "modal_data.npz")

# ============================================================
# LAUNCH MAPDL
# ============================================================

mapdl = launch_mapdl(run_location=out_dir, override=True, loglevel="ERROR")
mapdl.clear()
mapdl.prep7()

# ============================================================
# MATERIAL / ELEMENT
# ============================================================

mapdl.mp("EX",   1, E)
mapdl.mp("PRXY", 1, NU)
mapdl.mp("DENS", 1, RHO)

mapdl.et(1, "SHELL181")
mapdl.r(1, THK)

# ============================================================
# GEOMETRY + MESH
# ============================================================

k1 = mapdl.k("", 0,  0,  0)
k2 = mapdl.k("", LX, 0,  0)
k3 = mapdl.k("", LX, LY, 0)
k4 = mapdl.k("", 0,  LY, 0)
mapdl.a(k1, k2, k3, k4)

ndiv_x = nx - 1
ndiv_y = ny - 1

mapdl.lsel("S", "LINE", "", 1)
mapdl.lesize("ALL", "", "", ndiv_x)
mapdl.lsel("S", "LINE", "", 3)
mapdl.lesize("ALL", "", "", ndiv_x)

mapdl.lsel("S", "LINE", "", 2)
mapdl.lesize("ALL", "", "", ndiv_y)
mapdl.lsel("S", "LINE", "", 4)
mapdl.lesize("ALL", "", "", ndiv_y)

mapdl.allsel()
mapdl.amesh("ALL")

# ============================================================
# OPTIONAL POINT MASS (translational)
# ============================================================

if point_mass is not None and point_mass > 0:
    def _window_indices(center_idx, n_total, n_pick):
        n_pick = int(max(1, min(int(n_pick), int(n_total))))
        start = int(center_idx) - n_pick // 2
        start = max(0, min(start, int(n_total) - n_pick))
        return np.arange(start, start + n_pick, dtype=int)

    # Find nearest mesh node to (pm_x, pm_y)
    ncoords_m = mapdl.mesh.nodes  # (nnodes, 3)
    nnums_m = mapdl.mesh.nnum     # (nnodes,)

    dist = (ncoords_m[:, 0] - pm_x) ** 2 + (ncoords_m[:, 1] - pm_y) ** 2
    pm_center_idx = int(np.argmin(dist))
    pm_node_center_id = int(nnums_m[pm_center_idx])

    # Distribute total added mass over a local node patch.
    dx = LX / (nx - 1)
    dy = LY / (ny - 1)
    ix_all = np.clip(np.rint(ncoords_m[:, 0] / dx).astype(int), 0, nx - 1)
    iy_all = np.clip(np.rint(ncoords_m[:, 1] / dy).astype(int), 0, ny - 1)
    ix0 = int(np.clip(np.rint(pm_x / dx), 0, nx - 1))
    iy0 = int(np.clip(np.rint(pm_y / dy), 0, ny - 1))

    sel_ix = _window_indices(ix0, nx, pm_patch_nx)
    sel_iy = _window_indices(iy0, ny, pm_patch_ny)
    patch_mask = np.isin(ix_all, sel_ix) & np.isin(iy_all, sel_iy)
    pm_node_idxs = np.where(patch_mask)[0]

    if pm_node_idxs.size == 0:
        # Fallback to single nearest node if patch selection fails unexpectedly.
        pm_node_idxs = np.array([pm_center_idx], dtype=int)

    pm_node_ids = nnums_m[pm_node_idxs].astype(int)
    point_mass_per_node = float(point_mass) / float(pm_node_ids.size)

    # Add distributed point masses using MASS21 at selected nodes.
    # MASS21 real constants: MX, MY, MZ, IXX, IYY, IZZ
    mapdl.et(2, "MASS21")
    mapdl.r(2, point_mass_per_node, point_mass_per_node, point_mass_per_node, 0, 0, 0)
    mapdl.type(2)
    mapdl.real(2)
    for _nid in pm_node_ids:
        mapdl.e(int(_nid))
    mapdl.allsel()

    print(
        f"Added distributed point mass: total={point_mass:.6g} kg, "
        f"per_node={point_mass_per_node:.6g} kg, "
        f"patch={len(sel_ix)}x{len(sel_iy)} ({pm_node_ids.size} nodes), "
        f"target=(x={pm_x:.6g}, y={pm_y:.6g}), center_node={pm_node_center_id}"
    )

# ============================================================
# BOUNDARY CONDITIONS
# ============================================================

# The plate's own nodes, captured BEFORE any spring ground-nodes are created.
# A compliant clamp (BASE_BC=spring:<k>) adds one grounded node per base node; those
# are not part of the plate and must not reach node_coords / the stress arrays, which
# downstream code maps onto the nx*ny camera grid.
mapdl.allsel()
plate_nnum = mapdl.mesh.nnum.copy()

if base_edge.lower() == "left":
    sel = "LOC, X, 0"
elif base_edge.lower() == "right":
    sel = f"LOC, X, {LX}"
elif base_edge.lower() == "bottom":
    sel = "LOC, Y, 0"
elif base_edge.lower() == "top":
    sel = f"LOC, Y, {LY}"
else:
    raise ValueError("Invalid base_edge")

mapdl.nsel("S", sel)
mapdl.cm("BASE_NODES", "NODE")
mapdl.allsel()

mapdl.cmsel("S", "BASE_NODES")
# BASE_BC: "clamped" (default) = all 6 DOF fixed -> a perfectly rigid clamp.
# "spring:<k>"                 = a COMPLIANT clamp: translations and the in-plane
#   rotations stay fixed, but the bending rotation about the clamp line (ROTX, the
#   edge runs along X) is restrained by a torsional spring of <k> N*m/rad per node
#   instead of being rigid. k -> inf recovers the perfect clamp; finite k models
#   real fixture compliance. This is what "clamping stiffness" means for a cantilever.
#
# NOTE: a fully PINNED edge (rotations simply released) is NOT a valid degraded clamp
# here -- a cantilever whose only support cannot carry moment is a hinge, i.e. a
# mechanism, and the model returns a 0 Hz rigid-body mode. Use spring:<k> instead.
base_bc = os.environ.get("BASE_BC", "clamped").strip().lower()

if base_bc == "clamped":
    print("Base BC: clamped (rigid) -> all 6 DOF")
    for dof in ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"):
        mapdl.d("ALL", dof, 0)
    mapdl.allsel()

elif base_bc.startswith("spring:"):
    k_rot = float(base_bc.split(":", 1)[1])
    if k_rot <= 0:
        raise ValueError(f"spring stiffness must be > 0, got {k_rot}")
    base_ids = mapdl.mesh.nnum.tolist()          # currently-selected = BASE_NODES
    print(f"Base BC: compliant clamp, k_rot = {k_rot:g} N*m/rad per node "
          f"on ROTX, over {len(base_ids)} base nodes")
    # everything except the hinge rotation stays rigid
    for dof in ("UX", "UY", "UZ", "ROTY", "ROTZ"):
        mapdl.d("ALL", dof, 0)
    mapdl.allsel()

    # torsional spring to ground on ROTX: COMBIN14 with KEYOPT(2)=4 (1-D, ROTX)
    mapdl.prep7()
    mapdl.et(2, "COMBIN14")
    mapdl.keyopt(2, 2, 4)      # DOF = ROTX
    mapdl.keyopt(2, 3, 0)      # 1-D behaviour (selected by KEYOPT(2))
    mapdl.r(2, k_rot)
    mapdl.type(2)
    mapdl.real(2)
    for nid in base_ids:
        x, y, z = mapdl.mesh.nodes[mapdl.mesh.nnum.tolist().index(nid)]
        g = mapdl.n("", x, y, z)               # coincident ground node
        for dof in ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"):
            mapdl.d(g, dof, 0)
        mapdl.e(nid, g)
    mapdl.allsel()

else:
    raise ValueError(f"Invalid BASE_BC: {base_bc!r} "
                     f"(expected 'clamped' or 'spring:<k_rot>')")

# ============================================================
# MODAL ANALYSIS
# ============================================================

mapdl.finish()
mapdl.run("/SOLU")
mapdl.antype("MODAL")
mapdl.modopt("LANB", nmodes)
mapdl.mxpand(nmodes, "", "", "YES")
mapdl.solve()
mapdl.finish()

# ============================================================
# POSTPROCESSING – MODAL DATA
# ============================================================

mapdl.post1()
result = mapdl.result

node_coords = result.mesh.nodes
node_nums   = result.mesh.nnum

# drop the spring ground-nodes (if any): keep only the plate mesh
keep_nodes  = np.isin(node_nums, plate_nnum)
if not keep_nodes.all():
    print(f"Excluding {int((~keep_nodes).sum())} spring ground-nodes "
          f"({len(node_nums)} -> {int(keep_nodes.sum())} plate nodes)")
node_coords = node_coords[keep_nodes]
node_nums   = node_nums[keep_nodes]
nnodes      = len(node_nums)

# Choose force node
if force_node_id is not None:
    # exact node ID specified
    idxs = np.where(node_nums == force_node_id)[0]
    if len(idxs) == 0:
        raise ValueError(f"FORCE_NODE_ID={force_node_id} not found in mesh.")
    force_node_idx = int(idxs[0])

elif force_x is not None and force_y is not None:
    # nearest node to (force_x, force_y)
    dist = (node_coords[:, 0] - force_x)**2 + (node_coords[:, 1] - force_y)**2
    force_node_idx = int(np.argmin(dist))
    force_node_id  = int(node_nums[force_node_idx])

else:
    # fallback: center of plate
    cx, cy = LX / 2, LY / 2
    dist = (node_coords[:, 0] - cx)**2 + (node_coords[:, 1] - cy)**2
    force_node_idx = int(np.argmin(dist))
    force_node_id  = int(node_nums[force_node_idx])


dir_map = {"X": 0, "Y": 1, "Z": 2}
idir = dir_map[force_dir.upper()]

modal_freqs  = np.zeros(nmodes)
modal_omega  = np.zeros(nmodes)
# Force participation uses only the displacement component along the
# excitation direction (UX/UY/UZ). However, modal mass (generalized mass)
# must be computed from the full translational displacement vector
# (UX, UY, UZ) to remain physically consistent and to avoid inflating
# specific modes when their motion is not aligned with the chosen direction.
modal_phi    = np.zeros((nmodes, nnodes))
modal_ux     = np.zeros((nmodes, nnodes))
modal_uy     = np.zeros((nmodes, nnodes))
modal_uz     = np.zeros((nmodes, nnodes))
modal_sx     = np.zeros((nmodes, nnodes))
modal_sy     = np.zeros((nmodes, nnodes))
modal_sxy    = np.zeros((nmodes, nnodes))

for i in range(nmodes):
    modal_freqs[i] = result.time_values[i]
    modal_omega[i] = 2 * np.pi * modal_freqs[i]

    _, disp = result.nodal_displacement(i)
    disp = disp[keep_nodes]          # plate nodes only (see keep_nodes above)
    # disp columns are [UX, UY, UZ, ROTX, ROTY, ROTZ]
    modal_ux[i, :] = disp[:, 0]
    modal_uy[i, :] = disp[:, 1]
    modal_uz[i, :] = disp[:, 2]
    modal_phi[i, :] = disp[:, idir]

    _, stress = result.nodal_stress(i)
    stress = stress[keep_nodes]
    modal_sx[i, :]  = stress[:, 0]
    modal_sy[i, :]  = stress[:, 1]
    modal_sxy[i, :] = stress[:, 3]

# ============================================================
# MODAL MASSES (lumped)
# ============================================================

total_mass = LX * LY * THK * RHO
m = np.full(nnodes, total_mass / nnodes)

# Keep the modal-mass estimate consistent with the MAPDL point mass
# (lumped add-on at all selected patch nodes).
if point_mass is not None and point_mass > 0 and pm_node_ids.size > 0:
    point_mass_per_node = float(point_mass) / float(pm_node_ids.size)
    node_idx_map = {int(nid): i for i, nid in enumerate(node_nums)}
    missing_nodes = []
    for _nid in pm_node_ids:
        _idx = node_idx_map.get(int(_nid), None)
        if _idx is None:
            missing_nodes.append(int(_nid))
        else:
            m[int(_idx)] += point_mass_per_node
    if missing_nodes:
        # Should not happen, but keep the script robust.
        print(
            "Warning: some point-mass nodes not found in POST1 node list; "
            f"modal mass not adjusted for {len(missing_nodes)} nodes."
        )

Mi = np.zeros(nmodes)
for i in range(nmodes):
    # Generalized (modal) mass based on translational kinetic energy
    # Mi = sum_n m_n * (ux^2 + uy^2 + uz^2)
    Mi[i] = np.sum(m * (modal_ux[i]**2 + modal_uy[i]**2 + modal_uz[i]**2))

# ============================================================
# FRF SYNTHESIS (STRESS / FORCE)
# ============================================================

if freqs_file:
    if not os.path.exists(freqs_file):
        raise FileNotFoundError(f"FREQS_FILE does not exist: {freqs_file}")
    freqs = np.asarray(np.load(freqs_file), dtype=float).reshape(-1)
    if freqs.size < 2:
        raise ValueError(f"FREQS_FILE must contain at least 2 frequencies, got {freqs.size}")
    if np.any(np.diff(freqs) <= 0):
        raise ValueError("FREQS_FILE frequencies must be strictly increasing.")
    nfreq = int(freqs.size)
    fmin = float(freqs[0])
    fmax = float(freqs[-1])
    print(f"Using explicit frequency vector from {freqs_file} (n={nfreq}, {fmin:.3f}..{fmax:.3f} Hz)")
else:
    freqs = np.linspace(fmin, fmax, nfreq)
    print(f"Using linspace frequency vector (n={nfreq}, {fmin:.3f}..{fmax:.3f} Hz)")

omegas = 2 * np.pi * freqs

sx_frf  = np.zeros((nfreq, nnodes), dtype=np.complex128)
sy_frf  = np.zeros((nfreq, nnodes), dtype=np.complex128)
sxy_frf = np.zeros((nfreq, nnodes), dtype=np.complex128)

for j, omega in enumerate(omegas):
    for i in range(nmodes):
        wi = modal_omega[i]
        D  = wi**2 - omega**2 + 2j*zeta*wi*omega

        if abs(D) < 1e-30 or Mi[i] < 1e-30:
            continue

        phi_f = modal_phi[i, force_node_idx]
        Hi = (phi_f / (Mi[i] * D)) * force_amp

        sx_frf[j]  += Hi * modal_sx[i]
        sy_frf[j]  += Hi * modal_sy[i]
        sxy_frf[j] += Hi * modal_sxy[i]

sx_sy_frf = sx_frf + sy_frf

stress_tensor_frf = np.stack(
    [sx_frf, sy_frf, sxy_frf, sx_sy_frf],
    axis=-1
)

# ============================================================
# SAVE
# ============================================================

np.save(stress_tensor_outfile, stress_tensor_frf)
np.save(freq_outfile, freqs)
np.save(coords_outfile, node_coords)

np.savez(
    modal_data_outfile,
    modal_freqs=modal_freqs,
    modal_omega=modal_omega,
    modal_mass=Mi,
    force_node_id=force_node_id,
    point_mass=point_mass if point_mass is not None else 0.0,
    point_mass_x=pm_x if pm_x is not None else np.nan,
    point_mass_y=pm_y if pm_y is not None else np.nan,
    point_mass_patch_nx=int(pm_patch_nx),
    point_mass_patch_ny=int(pm_patch_ny),
    point_mass_num_nodes=int(pm_node_ids.size),
    point_mass_node_id=pm_node_center_id if pm_node_center_id is not None else -1,
    point_mass_node_ids=pm_node_ids.astype(np.int64),
    force_dir=force_dir,
    zeta=zeta,
    modal_sx=modal_sx,
    modal_sy=modal_sy,
    modal_sxy=modal_sxy,
    modal_uz=modal_uz,
    lumped_mass=m,
)

print("✅ DONE")
print(f"Force node: {force_node_id}")
if point_mass is not None and point_mass > 0 and pm_node_ids.size > 0:
    print(
        f"Point mass: total={point_mass:.6g} kg at (x={pm_x:.6g}, y={pm_y:.6g}), "
        f"distributed over {pm_node_ids.size} nodes (patch {pm_patch_nx}x{pm_patch_ny}), "
        f"center node {pm_node_center_id}"
    )
print(f"Stress FRF shape: {stress_tensor_frf.shape}  [Pa/N]")
print(f"Saved: {stress_tensor_outfile}")
print(f"Freqs saved: {freq_outfile}  [{freqs.min():.2f}..{freqs.max():.2f}] Hz")
print(f"Coords saved: {coords_outfile}  shape={node_coords.shape}")
print(f"Modal freqs (Hz): {modal_freqs[:min(10, nmodes)]}")
print(f"Grid: nx={nx}, ny={ny}, plate={LX:.6g} x {LY:.6g} m")

mapdl.exit()
