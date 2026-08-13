"""
Fig — Recovery error vs. stress level (the fatigue-relevance finding).

Key finding of the numerical verification: the recovery is most accurate exactly
at the HIGH-STRESS nodes, and degrades only at low-stress nodes. Because fatigue
damage scales with a high power of stress (Basquin/Wohler exponent), the
high-stress nodes are the ones that govern life prediction -- so the method is
accurate where it matters. The fatigue-damage-weighted error is ~1.3 %, an order
of magnitude below the unweighted full-field mean.

Left panel : median relative recovery error per component vs. true-stress
             percentile (nodes sorted low->high stress). Error collapses to ~1-4 %
             in the high-stress band (shaded).
Right panel: damage-weighted vs. unweighted first-invariant error for a range of
             Basquin exponents b -- weighting by stress^b (fatigue damage) pulls
             the effective error down to ~1.3 %.

Data: synthetic_validation/figures_data/validation_fields.npz
"""
from pathlib import Path as _Path

# Repository root (this file's directory) - all paths below are relative to it.
_REPO = _Path(__file__).resolve().parent
(_REPO / 'synthetic_validation' / 'figures').mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_style import apply_style, MM_TO_INCH

apply_style()

BASE     = _REPO
DATA_DIR = BASE / 'synthetic_validation' / 'figures_data'
FIG_DIR  = BASE / 'synthetic_validation' / 'figures'

def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{name}.pdf')
    fig.savefig(FIG_DIR / f'{name}.png')
    print(f'Saved: {name}.pdf, {name}.png')

# ── Load ──────────────────────────────────────────────────────────────────
d     = np.load(DATA_DIR / 'validation_fields.npz', allow_pickle=True)
freqs = d['freqs']
df    = float(freqs[1] - freqs[0])

def band_rms(tag, kind):
    """Per-node band-RMS stress amplitude = sqrt(integral of PSD df)."""
    P = np.abs(d[f'{kind}_{tag}'])          # (Nf, Nnodes)
    return np.sqrt(np.sum(P, axis=0) * df)  # (Nnodes,)

COLORS = ['#0072B2', '#D55E00', '#009E73', '0.25']
COMPS = [
    ('SX',    r'$\sigma_{xx}$'),
    ('SY',    r'$\sigma_{yy}$'),
    ('SXY',   r'$\tau_{xy}$'),
    ('SX+SY', r'$\sigma_{xx}+\sigma_{yy}$'),
]

# ── Panel (a): median error vs stress percentile ──────────────────────────
NBIN = 10
edges = np.linspace(0, 100, NBIN + 1)
centers = 0.5 * (edges[:-1] + edges[1:])

fig_w = 170 * MM_TO_INCH
fig_h = 70 * MM_TO_INCH
fig, (axA, axB) = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                               gridspec_kw={'width_ratios': [1.35, 1.0]})

for (tag, tex), color in zip(COMPS, COLORS):
    tru = band_rms(tag, 'tru'); rec = band_rms(tag, 'rec')
    relerr = np.abs(rec - tru) / tru * 100.0
    pct = 100.0 * (np.argsort(np.argsort(tru)) / (len(tru) - 1))  # percentile rank
    med = np.array([np.median(relerr[(pct >= edges[i]) & (pct < edges[i + 1])])
                    if np.any((pct >= edges[i]) & (pct < edges[i + 1])) else np.nan
                    for i in range(NBIN)])
    ls = '--' if tag == 'SX+SY' else '-'
    axA.plot(centers, med, ls, color=color, lw=1.1, marker='o', ms=3.0,
             label=tex, zorder=3)

# Shade fatigue-relevant high-stress band (top 25%)
axA.set_yscale('log')
axA.set_xlim(0, 100)
axA.axvspan(75, 100, color='0.85', alpha=0.6, zorder=0)
axA.set_xlabel(r'True-stress percentile of node [\%]', fontsize=8.5)
axA.set_ylabel(r'Median relative recovery error [\%]', fontsize=8.5)
axA.set_title('(a) accuracy concentrates at high stress', fontsize=8.5, loc='left', pad=3)
axA.tick_params(labelsize=7.5, width=0.4, length=2.5)
axA.tick_params(which='minor', width=0.3, length=1.5)
axA.legend(fontsize=7, loc='upper right', handlelength=1.5, ncol=2,
           columnspacing=1.0, borderpad=0.3)
axA.axhline(1.0, color='0.6', lw=0.5, ls=':', zorder=1)
# 'fatigue-relevant' label at the BOTTOM of the shaded band (clear of the legend)
ylo_a, _ = axA.get_ylim()
axA.text(87.5, ylo_a * 1.5, 'fatigue-\nrelevant', ha='center', va='bottom',
         fontsize=6.5, color='0.35')

# ── Panel (b): damage-weighted vs unweighted error ────────────────────────
tru = band_rms('SX+SY', 'tru'); rec = band_rms('SX+SY', 'rec')
relerr = np.abs(rec - tru) / tru
bvals = np.array([0, 2, 3, 4, 5, 6, 8, 10])
werr = []
for b in bvals:
    w = tru ** b
    werr.append(np.sum(w * relerr) / np.sum(w) * 100.0)
werr = np.array(werr)
uerr = np.mean(relerr) * 100.0

axB.plot(bvals, werr, '-o', color='#0072B2', lw=1.2, ms=3.5,
         label='damage-weighted', zorder=3)
axB.axhline(uerr, color='0.35', lw=1.0, ls='--',
            label=f'unweighted mean ({uerr:.0f}\\%)', zorder=2)
axB.set_xlabel(r'Basquin fatigue exponent $b$  (damage $\propto \sigma^{\,b}$)', fontsize=8.5)
axB.set_ylabel(r'First-invariant error [\%]', fontsize=8.5)
axB.set_title('(b) fatigue-weighted error is $\\sim$1\\%', fontsize=8.5, loc='left', pad=3)
axB.set_ylim(0, max(uerr * 1.15, 15))
axB.set_xlim(-0.3, 10.3)
axB.tick_params(labelsize=7.5, width=0.4, length=2.5)
axB.legend(fontsize=7, loc='center right', handlelength=1.5, borderpad=0.3)
# annotate the plateau value
axB.annotate(f'{werr[bvals.tolist().index(5)]:.1f}\\%', xy=(5, werr[bvals.tolist().index(5)]),
             xytext=(5, werr[bvals.tolist().index(5)] + uerr * 0.18),
             fontsize=7, color='#0072B2', ha='center')

fig.tight_layout(pad=0.6, w_pad=1.8)
save_fig(fig, 'fig_error_vs_stress')
plt.close(fig)

# ── Console summary ───────────────────────────────────────────────────────
print('\nDamage-weighted first-invariant error:')
for b, e in zip(bvals, werr):
    print(f'  b={b:2d}: {e:.2f}%')
print(f'  unweighted mean: {uerr:.2f}%')
print('Done.')
