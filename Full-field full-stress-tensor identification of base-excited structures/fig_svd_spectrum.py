"""
Fig (R2.4 / App. B.3) -- singular-value spectrum of the two-mode first-invariant
camera basis Psi_cam = [psi_1, psi_2] (SX+SY mode shapes at the camera points,
column-normalised to unit norm, as built by
synthetic_validation.expansion.normalize_mode_shapes and consumed by the
pseudoinverse in modal_decompose()).

Basis source: the SEMM-identified prior mode shapes (modal_prior_semm.npz) --
this is the basis actually available at Method Step 2b (the true mode shapes
are unknown in a real deployment; the identified prior is what the
pseudoinverse in expansion.modal_decompose() actually uses).

Reproduces (not hardcodes) the response-letter numbers:
    singular values   1.21, 0.72
    condition number  kappa = 1.7
    inter-mode correlation  0.48  (|<psi_1,psi_2>|, unit-norm columns)
"""
from pathlib import Path as _Path

# Repository root (this file's directory) - all paths below are relative to it.
_REPO = _Path(__file__).resolve().parent
(_REPO / 'synthetic_validation' / 'figures').mkdir(parents=True, exist_ok=True)

import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, str(_REPO))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from plot_style import apply_style, MM_TO_INCH
apply_style()

from synthetic_validation.expansion import normalize_mode_shapes

BASE = str(_REPO / 'synthetic_validation' / 'analysis' / 'broad_analysis')
FIG  = (_REPO / 'synthetic_validation' / 'figures')

# ---------------------------------------------------------------------------
# Build Psi_cam exactly as the harness does (harness.py run_case_from_modal):
#   psi_by_comp["SX+SY"] = (modal_sx + modal_sy).T   (nnodes, nmodes)
#   truncate to n_modes=2, then normalize_mode_shapes() -> unit-norm columns
# ---------------------------------------------------------------------------
prior = {k: v for k, v in np.load(f'{BASE}/modal_prior_semm.npz', allow_pickle=True).items()}
n_modes = 2

psi_by_comp = {
    "SX":    prior["modal_sx"].T,
    "SY":    prior["modal_sy"].T,
    "SXY":   prior["modal_sxy"].T,
    "SX+SY": (prior["modal_sx"] + prior["modal_sy"]).T,
}
psi_by_comp = {c: p[:, :n_modes] for c, p in psi_by_comp.items()}
psi_by_comp = normalize_mode_shapes(psi_by_comp)
Psi_cam = psi_by_comp["SX+SY"]              # (nnodes, 2), unit-norm columns

U, S, Vt = np.linalg.svd(Psi_cam, full_matrices=False)
kappa = float(S[0] / S[1])
corr = float(abs(Psi_cam[:, 0] @ Psi_cam[:, 1]))

print(f'singular values: {S[0]:.3f}, {S[1]:.3f}')
print(f'condition number kappa = {kappa:.3f}')
print(f'inter-mode correlation = {corr:.3f}')

assert abs(S[0] - 1.21) < 0.01, S[0]
assert abs(S[1] - 0.72) < 0.01, S[1]
assert abs(kappa - 1.7) < 0.05, kappa
assert abs(corr - 0.48) < 0.01, corr

# ---------------------------------------------------------------------------
# Small appendix panel: stem plot of the two singular values
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(68 * MM_TO_INCH, 52 * MM_TO_INCH))

idx = np.array([1, 2])
markerline, stemlines, baseline = ax.stem(idx, S, basefmt=' ')
plt.setp(markerline, color='#0072B2', markersize=6, zorder=3)
plt.setp(stemlines, color='#0072B2', linewidth=1.4)

for i, s in zip(idx, S):
    ax.annotate(r'$%.2f$' % s, xy=(i, s), xytext=(0, 5), textcoords='offset points',
                fontsize=7.5, ha='center', va='bottom', color='#0072B2')

ax.set_xlim(0.5, 2.5)
ax.set_ylim(0, max(S) * 1.35)
ax.set_xticks(idx)
ax.set_xticklabels([r'$\sigma_1$', r'$\sigma_2$'])
ax.set_ylabel(r'singular value', fontsize=9)

ax.text(0.97, 0.95, r'$\kappa(\Psi_{\mathrm{cam}})=%.1f$' % kappa,
        transform=ax.transAxes, fontsize=8, ha='right', va='top')

fig.tight_layout()
fig.savefig(FIG / 'fig_svd_spectrum.pdf')
fig.savefig(FIG / 'fig_svd_spectrum.png')
print('saved fig_svd_spectrum')
plt.close(fig)
