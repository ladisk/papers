# Method: Direct PSD Multi-Component Stress Expansion

## Overview

This method computes full-field, multi-component stress PSD under base excitation by:
1. Extracting stress mode shapes from Stage 1 hammer FRFs (all components)
2. Computing stress PSD directly from the Stage 2 camera time history (no accelerometer)
3. Decomposing the measured PSD into modal contributions via non-negative least squares
4. Expanding to all stress components using Stage 1 mode shapes

No accelerometer is needed. The camera time history already contains the full stress response including the effect of the excitation profile. The only role of Stage 1 is to provide the relationship between stress components via mode shapes.

---

## Notation

| Symbol | Description | Shape |
|--------|-------------|-------|
| `Y_SEMM(f, j, k)` | Stage 1 SEMM stress FRF: stress at node j per unit force at excitation case k | `(n_freq, n_nodes*4, n_cases)` |
| `comp` | Stress component index: 0=SX, 1=SY, 2=SXY, 3=SX+SY | scalar |
| `ψ_r,c(x)` | Stress mode shape of mode r for component c at spatial point x | `(n_nodes,)` complex |
| `S_cam(f, x)` | Measured stress PSD from camera (SX+SY) | `(n_freq, n_pixels)` real, ≥ 0 |
| `S_c(f, x)` | Expanded stress PSD for component c | `(n_freq, n_nodes)` real, ≥ 0 |
| `β_r(f)` | Modal participation PSD of mode r | `(n_freq,)` real, ≥ 0 |

---

## Data Sources

### Stage 1 (hammer excitation of clamped structure)
- `Y_SEMM_stage1.npy`: complex stress FRFs, shape `(n_freq, n_nodes*4, 6)`. The second axis interleaves 4 stress components per node: `[SX_0, SY_0, SXY_0, (SX+SY)_0, SX_1, SY_1, ...]`. Extract component c with slice `[:, c::4, :]`.
- `freq_axis.npy`: frequency vector in Hz, shape `(n_freq,)`
- `node_coords.npy`: FE node coordinates, shape `(n_nodes, 3)`

### Stage 2 (base excitation of same clamped structure)
- Camera time history: `.npy` file, shape `(n_frames, n_rows, n_cols)`, units in Kelvin (temperature fluctuation).
- Sampling frequency `fs` from metadata.
- **No accelerometer required.**

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

### 1.3 Build the squared mode shape basis

For the PSD decomposition, we need the squared magnitude of each mode shape. These represent the spatial power distribution of each mode.

```python
Phi_cam = np.column_stack([
    np.abs(psi_1_cam)**2,
    np.abs(psi_2_cam)**2,
    ...
])   # shape (n_nodes, n_modes), real, non-negative
```

Similarly for each component:
```python
Phi_SX  = np.column_stack([np.abs(psi_1_SX)**2,  np.abs(psi_2_SX)**2,  ...])
Phi_SY  = np.column_stack([np.abs(psi_1_SY)**2,  np.abs(psi_2_SY)**2,  ...])
Phi_SXY = np.column_stack([np.abs(psi_1_SXY)**2, np.abs(psi_2_SXY)**2, ...])
```

---

## Step 2: Compute Camera Stress PSD

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
from scipy.spatial import cKDTree
tree = cKDTree(node_coords[:, :2])
_, cam_to_node = tree.query(cam_pixel_positions)
```

Use this to transfer mode shapes from the node grid to the camera pixel grid, or to aggregate camera data onto the node grid.

### 2.3 Compute PSD via Welch

```python
from scipy.signal import welch

seg_len = int(2 * n_frames / (n_segments + 1))

f_welch, S_cam = welch(sigma_cam_2d, fs=fs, nperseg=seg_len, axis=0)
# S_cam shape: (n_freq_welch, n_pixels), real, non-negative
# Units: Pa²/Hz
```

Optionally band-limit to the frequency range of interest (e.g., 45–300 Hz).

**Output:** `S_cam(f, x)` — the stress PSD at each pixel for SX+SY. This is already the final result for the camera component. It includes the full effect of the base excitation profile.

---

## Step 3: Modal Decomposition of PSD

### 3.1 Theoretical basis

The stress at any point is a superposition of modal contributions:

```
σ(f, x) = Σ_r α_r(f) · ψ_r,cam(x)
```

where `α_r(f)` are complex random variables (for random excitation). The PSD is:

```
S_cam(f, x) = E[|σ(f, x)|²] / Δf
            = Σ_r Σ_s  E[α_r α_s*]/Δf  ·  ψ_r(x) · ψ_s*(x)
```

**Key assumption: well-separated modes.** When modes are well-separated in frequency (e.g., 54 Hz and 170 Hz), the cross-modal terms `E[α_r α_s*]` for `r ≠ s` are negligible. This simplifies to:

```
S_cam(f, x) ≈ Σ_r β_r(f) · |ψ_r,cam(x)|²
```

where `β_r(f) = E[|α_r(f)|²] / Δf ≥ 0` is the modal participation PSD — a real, non-negative scalar at each frequency that describes how much power mode r carries.

### 3.2 Non-negative least squares (NNLS)

At each frequency `f`, solve:

```
S_cam(f, :) ≈ Φ_cam · β(f)
```

where:
- `S_cam(f, :)` is the measured PSD vector, shape `(n_nodes,)`, real ≥ 0
- `Φ_cam = [|ψ_1,cam|², |ψ_2,cam|², ...]` is the basis matrix, shape `(n_nodes, n_modes)`, real ≥ 0
- `β(f) = [β_1(f), β_2(f), ...]` are the unknowns, shape `(n_modes,)`, real ≥ 0

This is a massively overdetermined system (~841 equations, 2 unknowns) with non-negativity constraint.

```python
from scipy.optimize import nnls

beta = np.zeros((n_freq, n_modes))
residual = np.zeros(n_freq)

for i_f in range(n_freq):
    beta[i_f, :], residual[i_f] = nnls(Phi_cam, S_cam_nodes[i_f, :])
```

**Alternative (faster, no constraint):** If negative values are not a concern (they shouldn't be at resonances where signal is strong), ordinary least squares works:

```python
Phi_cam_pinv = np.linalg.pinv(Phi_cam)                # (n_modes, n_nodes), computed once
beta = (Phi_cam_pinv @ S_cam_nodes.T).T                # (n_freq, n_modes)
beta = np.maximum(beta, 0)                             # clip negatives
```

### 3.3 Output

`β_r(f)` for each mode r — the modal participation PSD. Units: Pa²/Hz (same as stress PSD, since mode shapes have unit norm squared).

**Interpretation:**
- `β_1(f)` peaks sharply around 54 Hz (mode 1 resonance)
- `β_2(f)` peaks sharply around 170 Hz (mode 2 resonance)
- Away from resonances, both are near zero
- The integral `∫ β_r(f) df` gives the total power of mode r — comparing these across modes tells you which one dominates
- Different excitation profiles change the relative magnitudes of `β_1` and `β_2`

---

## Step 4: Multi-Component Expansion

For any stress component `c`, reconstruct the PSD by swapping the mode shapes:

```python
S_c = beta @ Phi_c.T    # shape (n_freq, n_nodes), real
```

Explicitly:
```
S_SX(f, x)  = β₁(f) · |ψ₁,SX(x)|²  + β₂(f) · |ψ₂,SX(x)|²
S_SY(f, x)  = β₁(f) · |ψ₁,SY(x)|²  + β₂(f) · |ψ₂,SY(x)|²
S_SXY(f, x) = β₁(f) · |ψ₁,SXY(x)|² + β₂(f) · |ψ₂,SXY(x)|²
```

**Why this works:** The modal participation `β_r(f)` depends only on the excitation and the structure's dynamics — NOT on which stress component we observe. It is the same for SX, SY, SXY, and SX+SY. The camera measurement determines `β_r(f)` via the SX+SY shapes, and Stage 1 provides the shapes for all other components.

**Output:** `S_σσ,c(f, x)` — full-field stress PSD for each component, at every node, at every frequency. Units: Pa²/Hz.

---

## Step 5: Validation and Quality Metrics

### 5.1 Stabilization diagram
The sdypy-EMA stabilization diagram should show stable poles (consistent frequency, damping, and mode shape across polynomial orders), confirming reliable modal identification.

### 5.2 Reconstruction residual
At each frequency, the NNLS residual quantifies how well the two-mode basis explains the measured PSD:

```python
S_cam_reconstructed = beta @ Phi_cam.T
relative_residual = np.linalg.norm(S_cam - S_cam_reconstructed, axis=1) / np.linalg.norm(S_cam, axis=1)
```

Should be small (< 10-20%) at resonances where signal is strong.

### 5.3 MAC (Modal Assurance Criterion)
Compare the measured PSD spatial pattern at resonance with the squared mode shape:

```python
def MAC(a, b):
    return np.abs(a @ b)**2 / ((a @ a) * (b @ b))

mac_r = MAC(S_cam[k_r, :], np.abs(psi_r_cam)**2)
```

Should be close to 1.0.

### 5.4 Self-consistency check
The SX+SY expansion should match the directly measured camera PSD:

```python
S_cam_expanded = beta @ Phi_cam.T
error = np.abs(S_cam_expanded - S_cam) / np.max(S_cam)
```

This should be small everywhere.

---

## Theoretical Derivation

### Why mode shapes are the same for force and base excitation

For a clamped structure, the eigenvalue problem is:

```
(K - ω² M) φ = 0
```

with boundary conditions `x_c = 0` at the clamped DOFs. The eigenvalues `ω_r` and eigenvectors `φ_r` depend only on the structure's mass, stiffness, and boundary conditions — NOT on the excitation type (Česnik et al., JSV 2013; Ewins, Modal Testing, 2000).

Stage 1 (hammer) and Stage 2 (base excitation) apply to the same clamped structure → same mode shapes.

### Stress FRF modal synthesis

The stress FRF (Česnik et al., Eq. 15) can be written using modal synthesis:

```
σH_jk(ω) = Σ_r  ψ_σ,jr · φ_kr / (ω_r² - ω² + 2i ξ_r ω_r ω)
```

where `ψ_σ,jr` is the stress mode shape value at node j for mode r, and `φ_kr` is the displacement mode shape at excitation point k.

At resonance `ω = ω_r`, mode r dominates:

```
σH_jk(ω_r) ≈ ψ_σ,jr · φ_kr / (2i ξ_r ω_r²)
```

For all nodes j and cases k, this is a rank-1 matrix `ψ_σ,r · φ_r^T / scalar`. The sdypy-EMA identification exploits this modal structure across the full frequency range to extract poles (frequencies, damping) and residues (mode shapes) simultaneously.

### PSD decomposition

Under random base excitation, the stress at point x is:

```
σ(t, x) = Σ_r α_r(t) · ψ_r(x)
```

The auto-PSD is:

```
S_σσ(f, x) = Σ_r Σ_s S_αα,rs(f) · ψ_r(x) · ψ_s*(x)
```

For well-separated modes, `S_αα,rs ≈ 0` when `r ≠ s`:

```
S_σσ(f, x) ≈ Σ_r β_r(f) · |ψ_r(x)|²
```

This is the equation solved by NNLS in Step 3.

---

## Assumptions

1. **Same boundary conditions** in Stage 1 and Stage 2 → identical mode shapes
2. **Well-separated modes** → cross-modal PSD terms negligible (valid for 54 Hz vs 170 Hz)
3. **Linear system** → stress linearly proportional to excitation
4. **Light/proportional damping** → mode shapes approximately real
5. **Modal truncation** → only identified modes included; higher modes assumed negligible
6. **Thermoelastic linearity** → camera temperature ∝ SX+SY (adiabatic condition met at measurement frequencies)

---

## Advantages

- **No accelerometer needed** — camera time history is sufficient
- Simpler implementation — no cross-spectral estimation, no H1 estimator
- Directly produces stress PSD (the end goal for fatigue analysis)
- NNLS guarantees physically meaningful (non-negative) PSD values
- Conceptually transparent: camera determines "how much of each mode", Stage 1 provides "what each mode looks like"

## Limitations

- Cannot separate structure from excitation — if you change the excitation profile, you must re-measure with the camera (cannot predict)
- Works only with PSD (magnitude-squared) — no phase information between pixels
- Cross-modal terms neglected — not suitable for closely-spaced modes
- The expansion assumes the measured stress field is well-represented by the identified modes (residual should be checked)
- Cannot verify with coherence (no reference signal)

---

## Comparison with Transmissibility Method

| Aspect | Direct PSD | Transmissibility |
|--------|-----------|-----------------|
| Accelerometer | Not needed | Required |
| Prediction for new profiles | Must re-measure | Reuse T, plug in new S_aa |
| Phase information | Lost (PSD is real) | Preserved (T is complex) |
| Coherence check | Not available | Available |
| Implementation | Simpler | More complex |
| Theoretical framework | PSD decomposition | H1 estimator + modal expansion |
| End result | Same: S_σσ,c(f, x) | Same: S_σσ,c(f, x) |
