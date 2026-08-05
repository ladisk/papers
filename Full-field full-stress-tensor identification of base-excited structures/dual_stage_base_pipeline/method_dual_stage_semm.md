# Method: Dual-Stage SEMM for Base Excitation Stress FRFs

## Overview

This method computes full-field, multi-component stress FRFs under base excitation using two sequential applications of the System Equivalent Model Mixing (SEMM) technique:

1. **Stage 1 SEMM** — Combines a numerical FE model (parent, all 4 stress components at all nodes, for 6 hammer force cases) with experimental camera FRFs (overlay, SX+SY component only, same 6 cases) to produce a hybrid stress FRF that is accurate in all stress components: `Y_SEMM(f)`, shape `(n_freq, n_nodes*4, 6)`, units Pa/N.

2. **Equivalent input identification** — Converts the 6 hammer force inputs into a single virtual base acceleration input by computing frequency-dependent weights `q(f)` that combine the 6 hammer FRFs into a single base-acceleration FRF: `Y_parent_acc(f) = Y_SEMM(f) · q(f)`, shape `(n_freq, n_nodes*4, 1)`, units Pa/(m/s²).

3. **Stage 2 SEMM** — Combines the Stage 1-derived parent model under virtual base acceleration with a new camera overlay FRF measured during actual base excitation (using accelerometer as reference), producing the final hybrid stress FRF: `Y_SEMM_stage2(f)`, shape `(n_freq, n_nodes*4, 1)`, units Pa/(m/s²).

The final stress PSD is obtained by multiplying by the base acceleration PSD: `S_σσ(f,x) = |Y_SEMM_stage2(f,x)|² · S_aa(f)`.

---

## Notation

| Symbol | Description | Shape |
|--------|-------------|-------|
| `Y_num(f, j, k)` | Numerical FE stress FRF: stress at node j for force case k | `(n_freq, n_nodes*4, n_cases)` |
| `Y_cam_s1(f, i, k)` | Stage 1 camera stress FRF (SX+SY): pixel i for force case k | `(n_freq, n_pixels, n_cases)` |
| `Y_SEMM(f, j, k)` | Stage 1 SEMM hybrid stress FRF | `(n_freq, n_nodes*4, n_cases)` |
| `q(f)` | Equivalent input weights (force → acceleration) | `(n_freq, n_cases)` complex |
| `Y_parent_acc(f, j)` | Parent model under virtual base acceleration | `(n_freq, n_nodes*4, 1)` complex |
| `Y_cam_s2(f, i)` | Stage 2 camera stress FRF (SX+SY) vs accelerometer | `(n_freq, n_pixels, 1)` complex |
| `Y_SEMM_s2(f, j)` | Stage 2 SEMM final stress FRF | `(n_freq, n_nodes*4, 1)` complex |
| `S_aa(f)` | Base acceleration auto-PSD | `(n_freq,)` real |
| `S_σσ,c(f, x)` | Stress PSD for component c | `(n_freq, n_nodes)` real |
| `comp` | Stress component index: 0=SX, 1=SY, 2=SXY, 3=SX+SY | scalar |

---

## Data Sources

### Stage 1 (hammer excitation of clamped structure)

For each of the 6 hammer impact cases (center_middle, center_top, left_top, left_middle, right_middle, right_top):
- **Force signal**: `.pkl` file containing force time history and sampling rate.
- **Camera time history**: `.npy` file, shape `(n_frames, n_rows, n_cols)`, units in Kelvin. Requires preprocessing: impact-synchronized cropping, flip vertical, crop to ROI, spatial resolution reduction, temporal mean removal.

### Numerical FE model (parent for Stage 1)

- A Python solver script (`plate_stress_modal_superposition_force_pointmass_v2_camfreq_*.py`) is executed for each force case, producing:
  - `stress_tensor_frf.npy`: complex stress FRF, shape `(n_freq, n_nodes, 4)`, units Pa/N. The 4 components are SX, SY, SXY, SX+SY.
  - `freqs.npy`: frequency vector in Hz, shape `(n_freq,)`
  - `node_coords.npy`: FE node coordinates, shape `(n_nodes, 3)`

- The solver computes modal superposition stress FRFs using FE eigenmodes and eigenfrequencies. It uses the same spatial grid (nx, ny) and plate dimensions as the camera grid.

### Stage 2 (base excitation of same clamped structure)

- **Camera time history**: `.hcc` or `.npy` file, shape `(n_frames, n_rows, n_cols)`, units in Kelvin.
- **Accelerometer time history**: `.pkl` file containing acceleration signal and sampling rate. If in g-units, converted to m/s² by multiplying by 9.80665.

### Conversion

- Camera temperature to stress: `σ [Pa] = T [K] * camera_to_stress_factor` where `camera_to_stress_factor = 1 / thermoelastic_coefficient` and `thermoelastic_coefficient = -(α_T · T_ref) / (ρ · c_p)` from adiabatic thermoelastic theory. This converts the camera signal to `SX + SY` (first stress invariant).

---

## Step 1: Build Stage 1 Camera FRFs

### 1.1 Impact-synchronized cropping

For each of the 6 hammer cases, the force and camera signals must be synchronized:

1. Detect the impact in the force signal: find the peak, determine start and end of the transient based on RMS thresholds and relative peak thresholds.
2. Crop both force and camera signals to start just before the impact (configurable `pre_ms`).
3. Window the force: apply a cosine taper after the impact ends, then zero-pad beyond.

```python
force_proc, images_sync, sync_info, _ = sync_crop_and_window_force_camera(
    force_raw, fs_force, images_full, fs_ir,
    pre_ms=2.0,
    noise_ref_s=0.25,
    start_rms_mult=4.0,
    end_rms_mult=1.0,
    force_zero_post_ms=0.5,
    force_end_taper_ms=3.0,
)
```

### 1.2 Camera preprocessing

For each case:
1. Crop to ROI: `images_roi = images[:, row_start:row_end, col_start:col_end]`
2. Compute H1 FRF estimate using Welch method:

```python
freq_cam, H1_cam = frf_ir_h1(force_proc, fs_force, images_roi, fs_ir, n_segments=5)
```

The H1 estimator computes:
```
H1(f, pixel) = G_yx(f, pixel) / G_xx(f)
```
where `G_yx` is the cross-spectral density (camera × force*) and `G_xx` is the force auto-spectral density. Welch averaging with Hann window and 50% overlap.

3. Reduce spatial resolution by averaging blocks: `H1_reduced = reduce_resolution(H1_cam, factor=3, mode="crop")`
4. Interpolate to the target frequency axis (matching the numerical model).

### 1.3 Assemble multi-case camera FRF

Stack all 6 cases into a single tensor:

```python
Y_cam_all = np.zeros((n_freq, n_pixels, 6), dtype=complex)
for k, label in enumerate(case_labels):
    # ... load, sync, crop, H1, reduce, interpolate ...
    Y_cam_all[:, :, k] = H1_interpolated_flat
```

Apply thermoelastic conversion: `Y_cam_all *= camera_to_stress_factor`

**Output:** `Y_cam_all(f, i, k)` — camera stress FRF (SX+SY) at pixel i for force case k. Shape `(n_freq, n_pixels, 6)`, units Pa/N.

---

## Step 2: Build Stage 1 Numerical FRFs (Parent Model)

### 2.1 Run the FE solver for each hammer case

For each of the 6 hammer positions, the solver computes the stress FRF at all nodes using modal superposition:

```
σH_jk(ω) = Σ_r  ψ_σ,jr · φ_kr / (ω_r² - ω² + 2i ξ_r ω_r ω)
```

where `ψ_σ,jr` is the stress mode shape at node j for mode r, and `φ_kr` is the displacement mode shape at the force application point k.

The solver produces `stress_tensor_frf.npy` with 4 stress components per node (SX, SY, SXY, SX+SY) at all frequencies.

### 2.2 Assemble multi-case numerical FRF

```python
Y_num = np.zeros((n_freq, n_nodes * 4, 6), dtype=complex)
for k, label in enumerate(case_labels):
    stress_tensor = np.load(run_dirs[label] / "stress_tensor_frf.npy")  # (n_freq, n_nodes, 4)
    Y_num[:, :, k] = stress_tensor.reshape(n_freq, n_nodes * 4)
```

**Output:** `Y_num(f, j, k)` — numerical stress FRF for all 4 components interleaved, 6 cases. Shape `(n_freq, n_nodes*4, 6)`, units Pa/N.

The interleaving layout: `[SX_0, SY_0, SXY_0, (SX+SY)_0, SX_1, SY_1, ...]`. Extract component c with `Y_num[:, c::4, :]`.

---

## Step 3: Stage 1 SEMM

### 3.1 SEMM theory

SEMM (System Equivalent Model Mixing) is a frequency-based substructuring method that creates a hybrid model by combining:
- **Parent model**: the numerical FE model (full-field, all components, but with modeling errors)
- **Overlay model**: the experimental camera data (accurate amplitude at measured locations, but only SX+SY component and limited spatial resolution)

The overlay "corrects" the parent at the interface DOFs (camera pixel locations mapped to the nearest FE nodes, for the SX+SY component only).

Using pyFBS with `SEMM_type="fully-extend-svd"`:

```python
Y_SEMM = pyFBS.SEMM(
    Y_parent,           # (n_lines, n_parent_dofs, n_cases)
    Y_overlay,          # (n_lines, n_interface_dofs, n_cases)
    df_parent,          # DataFrame: Position_1,2,3 and Direction_1,2,3 for all parent DOFs
    df_imp_parent,      # DataFrame: Position/Direction/Case for all input cases
    df_overlay,         # DataFrame: Position/Direction for all interface DOFs
    df_imp_overlay,     # DataFrame: same input cases
    SEMM_type="fully-extend-svd",
)
```

### 3.2 Interface DOF mapping

Camera pixels are mapped to the nearest FE nodes using Euclidean distance:

```python
d2 = np.sum((camera_positions[:, None, :] - node_coords[None, :, :]) ** 2, axis=2)
cam_to_num_idx = np.argmin(d2, axis=1)  # nearest FE node for each pixel
```

Multiple camera pixels mapping to the same node are averaged. The interface DOFs are the unique node indices, using the SX+SY component (direction encoding `[0.5, 0.5, 0.0]`):

```python
b_nodes = np.unique(cam_to_num_idx)
b_dof_idx = 4 * b_nodes + 3    # component 3 = SX+SY
```

### 3.3 DOF coordinate encoding

SEMM requires coordinate DataFrames for spatial matching between parent and overlay. The stress components are encoded as directions:

| Component | Direction_1 | Direction_2 | Direction_3 | Interpretation |
|-----------|-------------|-------------|-------------|----------------|
| SX | 1.0 | 0.0 | 0.0 | Stress in X-direction |
| SY | 0.0 | 1.0 | 0.0 | Stress in Y-direction |
| SXY | 0.0 | 0.0 | 1.0 | Shear stress XY |
| SX+SY | 0.5 | 0.5 | 0.0 | First stress invariant |

Each FE node has 4 DOFs (one per stress component), so `n_parent_dofs = n_nodes * 4`. The parent DataFrame has one row per DOF with position (node x,y,z) and direction (component encoding).

### 3.4 Chunked frequency-by-frequency processing

SEMM is applied independently at each frequency line (or small chunks) to manage memory:

```python
for idx in frequency_chunks:
    Y_SEMM[idx] = pyFBS.SEMM(
        Y_parent[idx], Y_overlay[idx],
        df_parent, df_imp_parent,
        df_overlay, df_imp_overlay,
        SEMM_type="fully-extend-svd",
    )
```

### 3.5 What SEMM does mathematically

The "fully-extend-svd" SEMM variant uses the Lagrange multiplier formulation:

```
Y_SEMM = Y_parent + Y_parent_b · (Y_overlay_b^{-1} - Y_parent_b^{-1})^{-1} · (Y_overlay_b^{-1} · Y_overlay - Y_parent_b^{-1} · Y_parent_b_ext)
```

where subscript `b` denotes interface DOFs, and `b_ext` denotes the extension from interface to full field. The SVD variant uses pseudoinverse for robustness.

In effect:
- At interface DOFs (SX+SY at camera-mapped nodes): the result matches the experimental overlay
- At non-interface DOFs (SX, SY, SXY, and nodes without camera coverage): the result is the parent FE model, adjusted to be consistent with the overlay correction
- The "extension" propagates the experimental correction from SX+SY to all other components through the numerical model's component relationships

**Output:** `Y_SEMM(f, j, k)` — hybrid stress FRF, shape `(n_freq, n_nodes*4, 6)`, units Pa/N. All 4 stress components at all nodes, corrected by camera data.

---

## Step 4: Equivalent Input Identification

### 4.1 Goal

The Stage 1 SEMM result `Y_SEMM` has 6 hammer force inputs (Pa/N). To use it as the parent model for Stage 2 (base excitation), we need to express the base acceleration as a combination of the 6 hammer forces.

The equivalent input vector `q(f)` converts from force basis to acceleration basis:

```
Y_parent_acc(f) = Y_SEMM(f) · q(f)
```

where `Y_parent_acc` has shape `(n_freq, n_nodes*4, 1)` and units Pa/(m/s²).

### 4.2 Virtual base method (used in practice)

The virtual base method constructs `q` from the Stage 1 metadata alone — no fitting to Stage 2 data needed. It assumes the base excitation applies a spatially distributed force proportional to a Gaussian weighting of the 6 hammer positions:

1. Compute the center of the base region: `x_mid = (x_min + x_max) / 2`, `y_bottom = y_min`
2. Compute distance from each hammer point to the base center: `d_k = sqrt((x_k - x_mid)² + (y_k - y_bottom)²)`
3. Apply Gaussian weighting: `w_k = exp(-d_k² / (2σ²))` where `σ = 0.25 · max(L_x, L_y)`
4. Normalize: `w_k = w_k / Σ w_k`
5. Scale by equivalent mass: `q_k = m_eq · w_k`

```python
q = np.tile(q0[None, :], (n_freq, 1))    # (n_freq, 6), frequency-independent
```

**Key property:** The virtual base q is frequency-independent (real-valued, constant). This means `Y_parent_acc(f) = Y_SEMM(f) · q₀` is a simple linear combination of the 6 hammer FRFs.

### 4.3 Alternative: overlay fit method

An alternative approach fits `q(f)` at each frequency to minimize the mismatch between the predicted and measured camera stress at the interface DOFs:

```
minimize_q  ||Y_SEMM_interface(f) · q(f) - Y_cam_stage2(f)||²
```

This gives frequency-dependent complex-valued `q(f)` via regularized least squares:

```python
q[f] = (B^H · B + λ·I)^{-1} · B^H · y
```

where `B = Y_SEMM[f, interface_dofs, :]` and `y = Y_cam_stage2[f, :, 0]`.

### 4.4 Build the parent acceleration FRF

```python
Y_parent_acc = np.einsum("fdi,fi->fd", Y_SEMM, q)[:, :, None]
# shape: (n_freq, n_nodes*4, 1)
```

**Output:** `Y_parent_acc(f, j)` — parent stress FRF per unit base acceleration. Units: Pa/(m/s²). This serves as the parent model for Stage 2 SEMM.

---

## Step 5: Build Stage 2 Camera Overlay

### 5.1 Load base excitation measurement

1. Load camera time history (base excitation): `.hcc` or `.npy`, shape `(n_frames, n_rows, n_cols)`, Kelvin
2. Load accelerometer signal: `.pkl`, units m/s² (converted from g if needed)
3. Flip vertical, crop to same ROI as Stage 1

### 5.2 Compute camera stress FRF vs accelerometer

Compute the H1 FRF estimate using the accelerometer as the reference signal:

```python
freq_base, H1_base = frf_ir_h1(accel, fs_acc, images_roi, fs_ir, n_segments=5)
```

This gives the stress transmissibility (stress per unit acceleration):

```
Y_cam_s2(f, pixel) = G_σa(f, pixel) / G_aa(f)
```

Units: Pa/(m/s²) after thermoelastic conversion.

### 5.3 Preprocessing

1. Reduce spatial resolution: `H1_reduced = reduce_resolution(H1_base, factor, mode="crop")`
2. Reshape to flat pixel vector: `h_flat`, shape `(n_freq_base, n_pixels)`
3. Interpolate to Stage 1 frequency axis: `h_tgt`, shape `(n_freq, n_pixels)`
4. Apply thermoelastic conversion: `Y_overlay_base = cam_to_stress * h_tgt`
5. Add case dimension: `Y_overlay_base = Y_overlay_base[:, :, None]`, shape `(n_freq, n_pixels, 1)`

### 5.4 Build Stage 2 overlay interface

Same interface mapping as Stage 1 — map camera pixels to nearest FE nodes, average multiple pixels per node:

```python
b_nodes_s2 = np.unique(cam_to_num_idx)
b_dof_idx_s2 = 4 * b_nodes_s2 + 3    # SX+SY component
```

The overlay tensor is resampled onto the node grid at the interface DOFs.

**Output:** `Y_overlay_base_num`, shape `(n_freq, n_interface_dofs, 1)`, complex. Units: Pa/(m/s²).

---

## Step 6: Stage 2 SEMM

### 6.1 Input definition

For Stage 2, the excitation is a single virtual base input. It is described by:
- Position: center of the bottom edge (`x_mid`, `y_min`, 0)
- Direction: `(0, 0, 1)` (out-of-plane acceleration)
- Case label: `"base_excitation"`

```python
df_imp_base = pd.DataFrame([{
    "Position_1": x_mid, "Position_2": y_bottom, "Position_3": 0.0,
    "Direction_1": 0.0, "Direction_2": 0.0, "Direction_3": 1.0,
    "Case": "base_excitation",
}])
```

### 6.2 Run Stage 2 SEMM

```python
Y_SEMM_stage2 = pyFBS.SEMM(
    Y_parent_acc,              # (n_freq, n_nodes*4, 1)
    Y_overlay_base_num,        # (n_freq, n_interface_dofs, 1)
    df_node_coords,            # parent DOF coordinates
    df_imp_base,               # single base input
    df_overlay_base,           # overlay DOF coordinates
    df_imp_base,               # same single base input
    SEMM_type="fully-extend-svd",
)
```

This second SEMM application:
- **Corrects** the parent acceleration FRF (derived from Stage 1 + equivalent inputs) using the actual camera measurement during base excitation
- The parent already has the right mode shapes (from Stage 1); Stage 2 corrects the amplitudes and possibly phases to match the actual base excitation response
- Again, the correction is applied at SX+SY interface DOFs and propagated to all 4 stress components

**Output:** `Y_SEMM_stage2(f, j)` — final hybrid stress FRF under base excitation. Shape `(n_freq, n_nodes*4, 1)`, units Pa/(m/s²). Contains all 4 stress components at all nodes.

---

## Step 7: Full-Field Stress PSD

### 7.1 Base acceleration PSD

Compute the acceleration PSD from the Stage 2 accelerometer signal:

```python
from scipy.signal import welch
f_welch, S_aa = welch(accel, fs=fs, nperseg=seg_len)
```

### 7.2 Stress PSD for each component

Extract the FRF for component c and compute:

```python
Y_c = Y_SEMM_stage2[:, c::4, 0]      # (n_freq, n_nodes)
S_sigma_c = np.abs(Y_c)**2 * S_aa[:, None]    # (n_freq, n_nodes), Pa²/Hz
```

Explicitly:
```
S_σσ,SX(f, x)  = |Y_SEMM_stage2(f, x, SX)|²  · S_aa(f)
S_σσ,SY(f, x)  = |Y_SEMM_stage2(f, x, SY)|²  · S_aa(f)
S_σσ,SXY(f, x) = |Y_SEMM_stage2(f, x, SXY)|² · S_aa(f)
```

**Output:** `S_σσ,c(f, x)` — full-field stress PSD for each component at every node, at every frequency. Units: Pa²/Hz.

---

## Verification

1. **Stage 1 SEMM quality**: Compare `Y_SEMM[:, 3::4, :]` (SX+SY from SEMM) with `Y_cam_all` at interface nodes — should match closely
2. **Equivalent input residual**: For the overlay fit method, the relative fitting residual should be small (< 10-20%). For the virtual base method, compare `Y_SEMM · q` with `Y_cam_s2` at interface DOFs
3. **Stage 2 SEMM quality**: Compare `Y_SEMM_stage2[:, 3::4, 0]` with `Y_cam_s2` at interface nodes
4. **Stress PSD consistency**: `S_σσ,SX+SY` from expansion should match the directly computed camera stress PSD
5. **Spatial stress patterns**: Mode shapes visible in the stress PSD at resonance should match known structural modes

---

## Theoretical Derivation

### SEMM formulation

SEMM is based on Lagrange Multiplier Frequency Based Substructuring (LM-FBS). Consider a parent model `P` and an overlay model `O` that share interface DOFs `b`. The SEMM result replaces the parent's behavior at the interface with the overlay's, while preserving the parent's full-field extension:

```
Y_SEMM = Y_P + Y_P,bi · (Y_O,bb^{-1} - Y_P,bb^{-1})^{-1} · (Y_O,bb^{-1} · Y_O,bk - Y_P,bb^{-1} · Y_P,bk)
```

where:
- `Y_P,bi` = parent FRF from input to interface DOFs
- `Y_P,bb` = parent FRF at interface (interface-to-interface)
- `Y_O,bb` = overlay FRF at interface
- `Y_O,bk` = overlay FRF from input to interface
- `Y_P,bk` = parent FRF from input to interface

The "fully-extend-svd" variant uses SVD-based pseudoinverse for numerical stability.

### Equivalent input concept

Base excitation applies a distributed inertia force to all mass DOFs. For a rigid base with acceleration `a(t)`, each mass point `m_i` experiences force `f_i = -m_i · a`. If the hammer positions sample the plate surface, the base force can be approximated as:

```
f_base(ω) ≈ q · a(ω)
```

where `q = [q_1, q_2, ..., q_6]^T` are equivalent force weights at the 6 hammer positions. The stress response is then:

```
σ(ω) = Y_SEMM(ω) · q · a(ω) = Y_parent_acc(ω) · a(ω)
```

The Gaussian weighting approximates the spatial distribution of inertia forces, with mass `m_eq` controlling the overall scaling. For the virtual base method, `q` is real and frequency-independent because it represents a geometric/mass distribution property, not a dynamic one.

### Why two SEMMs?

- **Stage 1 SEMM** corrects the numerical model using experimental data from hammer excitation — the same excitation type as the FE solver. This gives accurate stress mode shapes and component ratios.
- **Stage 2 SEMM** corrects the equivalent-input conversion by using actual camera data from base excitation. This accounts for any errors in the equivalent input assumption (spatial force distribution, damping changes, boundary condition differences).

---

## Pipeline Call Sequence

```python
pipe = SEMMThermoelasticPipeline(paths=paths, stage1=s1_cfg, stage2=s2_cfg, semm=semm_cfg)

# Stage 1
s1 = pipe.build_hammer_stage1()           # Load + H1 for 6 hammer cases
s1 = pipe.run_stage1_numerical_and_semm() # FE solver + SEMM

# Stage 2
s2 = pipe.load_base_measurement()          # Load camera + accelerometer
s2 = pipe.build_base_overlay_stress_acc()  # H1 camera vs accel + interface mapping
q  = pipe.identify_equivalent_inputs_virtual_base()  # or identify_equivalent_inputs()
Yp = pipe.build_acceleration_parent_from_stage1()    # Y_SEMM · q
s2 = pipe.run_stage2_semm()                # Final SEMM
```

---

## Assumptions

1. **Same boundary conditions** in Stage 1 (hammer) and Stage 2 (base excitation) — same clamped configuration, same mode shapes
2. **Linear system** — stress is linearly proportional to excitation
3. **Equivalent input approximation** — the base excitation force distribution can be approximated by a weighted combination of the 6 hammer point forces. This is an approximation; the actual base force is distributed over the entire clamped boundary, while hammer points are discrete locations on the plate surface
4. **Rigid base** — all base DOFs move together with the same acceleration. If the base (shaker table + fixture) has its own dynamics, the transmissibility concept breaks down
5. **Camera measures SX+SY only** — the overlay correction is applied at one component; other components (SX, SY, SXY) are corrected indirectly through the numerical model's component relationships
6. **Thermoelastic linearity** — camera temperature fluctuation is proportional to SX+SY (adiabatic condition met at measurement frequencies)
7. **Spatial resolution sufficiency** — the reduced camera grid (typically 29x29 ≈ 841 pixels) has enough resolution to capture the stress field spatial features

---

## Advantages

- **Full SEMM framework** — established method from the frequency-based substructuring community (pyFBS library)
- **Two correction stages** — Stage 1 corrects mode shapes, Stage 2 corrects the excitation coupling
- **Multi-component output** — produces SX, SY, SXY, SX+SY at every FE node
- **Complex FRF output** — preserves phase information, enabling cross-spectral analysis if needed
- **Flexible equivalent input** — supports both virtual base (pre-determined weights) and overlay fit (data-driven weights)

## Limitations

- **Complexity** — two SEMM computations, each requiring coordinate DataFrames, interface mapping, and chunked frequency processing
- **Single excitation case in Stage 2** — SEMM with only 1 input case has limited rank; it can scale amplitudes but cannot independently update multiple mode shapes. The correction is essentially a scalar correction at each frequency at the interface DOFs
- **Equivalent input approximation** — the Gaussian weighting of 6 hammer points is a coarse approximation to the actual distributed base force. The virtual base mass `m_eq` is a tuning parameter
- **Requires accelerometer** — the base acceleration reference is needed for the H1 estimator in Stage 2
- **Numerical FE model dependence** — the parent model accuracy (especially component ratios SX/SY/SXY) directly affects the final result. FE modeling errors in material properties, boundary conditions, or mesh density propagate through

---

## Comparison with Alternative Methods

| Aspect | Dual-Stage SEMM | Transmissibility Expansion | Direct PSD Expansion |
|--------|----------------|---------------------------|---------------------|
| Accelerometer | Required | Required | Not needed |
| FE model | Required (parent) | Not required | Not required |
| Mode shape source | FE + camera (SEMM) | sdypy-EMA from Y_SEMM FRFs | sdypy-EMA from Y_SEMM FRFs |
| Equivalent inputs | Required (6→1 conversion) | Not required | Not required |
| Number of SEMMs | 2 | 0 | 0 |
| Phase preservation | Full (complex FRF) | Full (complex transmissibility) | Lost (PSD is real) |
| Prediction for new profiles | Reuse Y_SEMM_s2, plug in new S_aa | Reuse T, plug in new S_aa | Must re-measure camera |
| Implementation complexity | High | Medium | Low |
| Theoretical framework | LM-FBS / SEMM | H1 + modal expansion | PSD decomposition + NNLS |
| End result | Same: S_σσ,c(f, x) | Same: S_σσ,c(f, x) | Same: S_σσ,c(f, x) |
