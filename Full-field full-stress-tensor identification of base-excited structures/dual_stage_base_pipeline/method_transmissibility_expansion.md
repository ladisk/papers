# Method: Transmissibility-Based Multi-Component Stress Expansion

## Overview

This method computes full-field, multi-component stress PSD under base excitation by:
1. Extracting stress mode shapes from Stage 1 hammer FRFs (all components)
2. Computing stress transmissibility from Stage 2 camera + accelerometer
3. Decomposing the transmissibility into modal contributions
4. Expanding to all stress components using Stage 1 mode shapes
5. Computing stress PSD by multiplying by the base acceleration PSD

The accelerometer is used as a reference to compute a proper H1 transmissibility estimate, separating the structural transfer function from the excitation.

---

## Notation

| Symbol | Description | Shape |
|--------|-------------|-------|
| `Y_SEMM(f, j, k)` | Stage 1 SEMM stress FRF: stress at node j per unit force at excitation case k | `(n_freq, n_nodes*4, n_cases)` |
| `comp` | Stress component index: 0=SX, 1=SY, 2=SXY, 3=SX+SY | scalar |
| `ψ_r,c(x)` | Stress mode shape of mode r for component c at spatial point x | `(n_nodes,)` complex |
| `T_cam(f, x)` | Measured stress transmissibility (SX+SY) from camera+accelerometer | `(n_freq, n_pixels)` complex |
| `T_c(f, x)` | Expanded stress transmissibility for component c | `(n_freq, n_nodes)` complex |
| `γ_r(f)` | Modal participation function of mode r | `(n_freq,)` complex |
| `S_aa(f)` | Base acceleration auto-PSD | `(n_freq,)` real |
| `S_σσ,c(f, x)` | Stress PSD for component c | `(n_freq, n_nodes)` real |
| `G_σa(f, x)` | Cross-spectral density between camera stress and base acceleration | `(n_freq, n_pixels)` complex |
| `G_aa(f)` | Auto-spectral density of base acceleration | `(n_freq,)` real |

---

## Data Sources

### Stage 1 (hammer excitation of clamped structure)
- `Y_SEMM_stage1.npy`: complex stress FRFs, shape `(n_freq, n_nodes*4, 6)`. The second axis interleaves 4 stress components per node: `[SX_0, SY_0, SXY_0, (SX+SY)_0, SX_1, SY_1, ...]`. Extract component c with slice `[:, c::4, :]`.
- `freq_axis.npy`: frequency vector in Hz, shape `(n_freq,)`
- `node_coords.npy`: FE node coordinates, shape `(n_nodes, 3)`

### Stage 2 (base excitation of same clamped structure)
- Camera time history: `.npy` file, shape `(n_frames, n_rows, n_cols)`, units in Kelvin (temperature fluctuation). Requires preprocessing: flip vertical, crop to ROI, reduce resolution, remove temporal mean.
- Accelerometer time history: `.npy` file, shape `(n_frames,)` or `(n_frames, n_channels)`, units in m/s² or g.
- Sampling frequency `fs` from metadata.

### Conversion
- Camera temperature to stress: `σ [Pa] = T [K] * camera_to_stress_factor` where `camera_to_stress_factor = -E * α_T / (ρ * cp * (1 - 2ν))` from thermoelastic theory. This converts the thermoelastic signal to `SX + SY` (first stress invariant).

---

## Step 1: Extract Stress Mode Shapes from Stage 1

### 1.1 Modal identification using sdypy-EMA

The SEMM stress FRFs are treated as standard FRFs and fed into `sdypy-EMA` for proper modal identification. This uses the LSCF (Least Squares Complex Frequency) method — a polyreference frequency-domain technique that identifies all modes simultaneously.

The stress FRFs have a special structure: the "response" DOFs are stress values (not displacements), but the mathematical form of the FRF is identical (modal synthesis, Eq. 3 in Česnik et al.). The modal identification algorithm does not distinguish between displacement and stress FRFs — it extracts poles and residues regardless.

#### Preparing the FRF matrix

For each stress component `c`, extract the FRF sub-matrix:

```python
# Y_SEMM shape: (n_freq, n_nodes*4, n_cases)
# Extract component c for all nodes and all cases:
H_c = Y_SEMM[:, c::4, :]    # shape (n_freq, n_nodes, n_cases)
```

Reshape into the format expected by sdypy-EMA: `(n_freq, n_response, n_reference)` where n_response = n_nodes and n_reference = n_cases (6 hammer points).

#### Running the identification

```python
import sdypy as sp

# Create the EMA model
model = sp.EMA(
    freq=freq_axis,           # frequency vector [Hz]
    FRF=H_c,                  # FRF matrix (n_freq, n_nodes, n_cases)
    pol_order_high=20,         # max polynomial order (tune as needed)
)

# Compute stabilization diagram
model.get_participation_factors()

# Select stable poles (interactively or programmatically)
# This identifies natural frequencies, damping ratios, and residues
model.select_poles()
```

#### Extracting modal parameters

After pole selection, sdypy-EMA provides:

```python
nat_freq = model.nat_freq          # natural frequencies [Hz], shape (n_modes,)
damping = model.damping            # damping ratios [-], shape (n_modes,)
residues = model.residues          # complex residues, shape (n_modes, n_nodes, n_cases)
```

The residue for mode r, response node j, reference case k is:
```
A_r,jk = ψ_σ,jr · φ_kr
```

The stress mode shape `ψ_σ,r` is contained in the residue matrix. For a single reference k:
```python
psi_r_c = residues[r, :, k]       # mode shape from reference k (up to scalar φ_kr)
```

With multiple references, the mode shape is obtained more robustly (sdypy-EMA handles this internally via the polyreference formulation).

#### Repeat for each component

Run the identification for each stress component (c = 0,1,2,3) to obtain:
- `ψ_r,SX` — stress mode shape for SX
- `ψ_r,SY` — stress mode shape for SY
- `ψ_r,SXY` — stress mode shape for SXY
- `ψ_r,cam` — stress mode shape for SX+SY (camera component)

**Note:** The natural frequencies and damping ratios should be consistent across components (they are structural properties). Differences indicate identification issues. Use the most reliable component (typically SX+SY with highest SNR) for frequency/damping, and extract shapes from each component separately.

**Output for each mode r:**
- Natural frequency `f_r` [Hz]
- Damping ratio `ξ_r` [-]
- Four complex stress mode shape vectors: `ψ_r,SX`, `ψ_r,SY`, `ψ_r,SXY`, `ψ_r,cam`, each shape `(n_nodes,)`

### 1.2 Normalization

Normalize the camera-component mode shapes to unit Euclidean norm:

```python
psi_hat_r_cam = psi_r_cam / np.linalg.norm(psi_r_cam)
```

The other component mode shapes must be scaled consistently — normalize all by the same factor used for the camera component:

```python
scale = np.linalg.norm(psi_r_cam)
psi_hat_r_SX  = psi_r_SX / scale
psi_hat_r_SY  = psi_r_SY / scale
psi_hat_r_SXY = psi_r_SXY / scale
psi_hat_r_cam = psi_r_cam / scale   # this has unit norm
```

This ensures the expansion ratios between components are preserved.

### 1.3 Build the mode shape basis matrix

For the camera component, stack all mode shapes into a matrix:

```python
Psi_cam = np.column_stack([psi_hat_1_cam, psi_hat_2_cam, ...])  # (n_nodes, n_modes)
```

Compute the pseudoinverse once:

```python
Psi_cam_pinv = np.linalg.pinv(Psi_cam)   # (n_modes, n_nodes)
```

---

## Step 2: Compute Stress Transmissibility from Stage 2

### 2.1 Preprocess camera data

1. Load camera time history: `cam_raw`, shape `(n_frames, n_rows, n_cols)`
2. Flip vertical if needed (match Stage 1 convention)
3. Crop to ROI: `cam_crop = cam_raw[:, row_start:row_end, col_start:col_end]`
4. Reduce spatial resolution: downsample by factor (e.g., 3) to match Stage 1 grid
5. Remove temporal mean per pixel: `cam = cam_crop - mean(cam_crop, axis=0)`
6. Convert to stress: `sigma_cam = cam * camera_to_stress_factor`
7. Reshape to 2D: `sigma_cam_2d`, shape `(n_frames, n_pixels)`

### 2.2 Map camera pixels to FE nodes

Camera pixels and FE nodes may not be on the same grid. Build a mapping:

```python
# camera pixel positions from the grid metadata
xs_cam = np.linspace(x_min, x_max, n_cols_reduced)
ys_cam = np.linspace(y_min, y_max, n_rows_reduced)
xx, yy = np.meshgrid(xs_cam, ys_cam)
cam_positions = np.column_stack([xx.ravel(), yy.ravel()])

# nearest-neighbor mapping to FE nodes
from scipy.spatial import cKDTree
tree = cKDTree(node_coords[:, :2])
_, cam_to_node = tree.query(cam_positions)
```

Use this mapping to transfer camera data to the node grid, or to transfer mode shapes to the camera grid.

### 2.3 Compute cross-spectral density (H1 estimator)

Using Welch's method, compute the stress transmissibility at each pixel:

```python
from scipy.signal import csd, welch

seg_len = int(2 * n_frames / (n_segments + 1))    # match Stage 1 convention

# For each pixel i:
G_sigma_a = csd(sigma_cam_2d[:, i], accel, fs=fs, nperseg=seg_len, axis=0)  # cross-spectrum
G_aa      = welch(accel, fs=fs, nperseg=seg_len)                             # auto-spectrum

T_cam_i = G_sigma_a / G_aa                         # H1 estimator, complex
```

Vectorized: compute for all pixels simultaneously.

**Output:** `T_cam(f, x)`, shape `(n_freq_welch, n_pixels)`, complex. Units: Pa / (m/s²).

### 2.4 Coherence (quality check)

```python
coherence = |G_sigma_a|² / (G_sigma_sigma * G_aa)
```

Values close to 1.0 indicate a linear relationship (good). Low coherence at certain frequencies or pixels indicates noise or nonlinearity.

---

## Step 3: Modal Decomposition of Transmissibility

At each frequency `f`, project the measured transmissibility onto the mode shape basis:

```python
# T_cam_nodes: transmissibility mapped to FE node grid, shape (n_freq, n_nodes), complex
# Psi_cam_pinv: precomputed, shape (n_modes, n_nodes)

gamma = np.zeros((n_freq, n_modes), dtype=complex)
for i_f in range(n_freq):
    gamma[i_f, :] = Psi_cam_pinv @ T_cam_nodes[i_f, :]
```

**Output:** `γ_r(f)` for each mode r — the complex, frequency-dependent modal participation function. Units: Pa/(m/s²) (same as transmissibility, since mode shapes are unit-normalized).

### Interpretation

- `|γ_1(f)|` peaks sharply at mode 1 resonance (~54 Hz)
- `|γ_2(f)|` peaks sharply at mode 2 resonance (~170 Hz)
- Away from resonances, both are small
- The ratio `|γ_1(f_1)| / |γ_2(f_2)|` tells you which mode is more strongly excited by this base profile

---

## Step 4: Multi-Component Expansion

For any stress component `c`, reconstruct the transmissibility:

```python
# Psi_c: mode shape matrix for component c, shape (n_nodes, n_modes)
# gamma: modal participation, shape (n_freq, n_modes)

T_c = gamma @ Psi_c.T    # shape (n_freq, n_nodes), complex
```

Explicitly:
```
T_SX(f, x)  = γ₁(f) · ψ̂_1,SX(x)  + γ₂(f) · ψ̂_2,SX(x)
T_SY(f, x)  = γ₁(f) · ψ̂_1,SY(x)  + γ₂(f) · ψ̂_2,SY(x)
T_SXY(f, x) = γ₁(f) · ψ̂_1,SXY(x) + γ₂(f) · ψ̂_2,SXY(x)
```

---

## Step 5: Full-Field Stress PSD

Compute the base acceleration PSD:

```python
f_welch, S_aa = welch(accel, fs=fs, nperseg=seg_len)
```

Then for each component:

```python
S_sigma_c = np.abs(T_c)**2 * S_aa[np.newaxis, :]    # shape (n_freq, n_nodes)
```

Explicitly:
```
S_σσ,c(f, x) = |T_c(f, x)|² · S_aa(f)
```

**Output:** Full-field stress PSD for SX, SY, SXY at every node, at every frequency. Units: Pa²/Hz.

---

## Verification

1. **Stabilization diagram** from sdypy-EMA should show stable poles (consistent frequency, damping, and mode shape across polynomial orders)
2. **Coherence** between camera and accelerometer should be close to 1.0 at resonances
3. **Reconstruction check**: `T_cam_reconstructed = gamma @ Psi_cam.T` should match `T_cam` with low residual
4. **MAC** between Stage 1 mode shapes and Stage 2 transmissibility shapes at resonance should be close to 1.0
5. **S_σσ,cam reconstructed** vs **S_σσ,cam measured** — the SX+SY PSD from expansion should match the directly measured camera PSD

---

## Assumptions

1. **Same boundary conditions** in Stage 1 and Stage 2 (same clamped configuration) → mode shapes are identical
2. **Well-separated modes** — at each resonance, one mode dominates (cross-modal terms negligible)
3. **Linear system** — stress is linearly proportional to excitation (no geometric/material nonlinearity)
4. **Proportional or light damping** — mode shapes are approximately real
5. **Modal truncation** — only the identified modes are included; contribution from higher modes is assumed negligible in the frequency range of interest

---

## Advantages

- Proper H1 transmissibility estimate with coherence quality metric
- Transmissibility T is a structural property — once computed, reuse for any S_aa to predict stress PSD under different excitation profiles
- Full complex information preserved throughout (phase, sign)
- Standard theoretical framework (modal analysis, H1 estimator) well-established in literature

## Limitations

- Requires accelerometer at the base
- Assumes rigid base (all base DOFs move together) — if base is flexible, transmissibility concept breaks down
- Modal truncation error at frequencies above the highest identified mode
- Cross-modal terms neglected (valid for well-separated modes only)
