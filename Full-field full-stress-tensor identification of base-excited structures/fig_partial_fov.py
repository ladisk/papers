"""
Fig — Partial-FoV / reduced-resolution robustness (R2.11), corner-mass truth, SEMM prior.

Mean per-component relative-RMS error vs number of camera points (log x), two series:
  - noise-free : flat at the prior-discrepancy floor (~11 %) from 841 down to 9 points
                 -> recovery is insensitive to camera resolution (2-mode basis over-determined).
  - 25 % per-point noise : fewer points -> weaker spatial averaging -> rising error.

Data: synthetic_validation/figures_data/straggler_data.npz
"""
import sys, matplotlib
matplotlib.use('Agg')
sys.path.insert(0, r'C:\Users\jasas\Work\Clanki\Clanek_2\Writing_article\article_bundle\scripts')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_style import apply_style, MM_TO_INCH
apply_style()

DATA = Path(r'C:\Users\jasas\Work\Clanki\Clanek_2\thermoelastic-stress-expansion\synthetic_validation\figures_data')
FIG  = Path(r'C:\Users\jasas\Work\Clanki\Clanek_2\thermoelastic-stress-expansion\synthetic_validation\figures')

def save_fig(fig, name):
    fig.savefig(FIG / f'{name}.pdf'); fig.savefig(FIG / f'{name}.png'); print('saved', name)

d = np.load(DATA / 'straggler_data.npz', allow_pickle=True)
idx = np.argsort(d['npoints'])
npts = d['npoints'][idx]
nf = d['relrms_noisefree'][idx] * 100.0
ny = d['relrms_noisy'][idx] * 100.0
noise_pct = int(round(float(d['noise_rel']) * 100))

C_NF, C_NY = '#0072B2', '#D55E00'
fig, ax = plt.subplots(figsize=(120 * MM_TO_INCH, 75 * MM_TO_INCH))
ax.semilogx(npts, nf, color=C_NF, ls='-',  marker='o', ms=4, lw=1.0, label='noise-free')
ax.semilogx(npts, ny, color=C_NY, ls='--', marker='^', ms=4, lw=1.0,
            label=rf'${noise_pct}\,\%$ per-point noise')

ax.axvline(x=9, color='gray', ls=':', lw=0.6)
ax.set_xlabel(r'Number of camera points')
ax.set_ylabel(r'Mean rel-RMS error [\%]')
ax.set_xticks(npts)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.minorticks_off()
ax.set_xlim(npts[0] * 0.7, npts[-1] * 1.6)
ax.set_ylim(0, max(ny.max() * 1.1, 20))
ax.legend(loc='upper right')
ax.text(9 * 1.15, ax.get_ylim()[1] * 0.92, r'$\leftarrow$ min (2-mode basis)',
        fontsize=6.5, color='gray', va='top', ha='left')

fig.tight_layout(pad=0.4)
save_fig(fig, 'fig_partial_fov')
plt.close(fig)
