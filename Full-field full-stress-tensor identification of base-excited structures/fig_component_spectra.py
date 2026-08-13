"""
Fig — per-component stress-PSD spectra, truth vs recovered, at the representative
highly-stressed multiaxial node (38, 54) mm (node 283). Four panels: sigma_xx,
sigma_yy, tau_xy, and the measured invariant sigma_xx+sigma_yy. Truth is a thick
translucent grey band, Recovered a thin blue line on top; per-node rel-RMS on the
PSD is annotated in each panel.

REGENERATED from data (the original one-off script was never committed). This is now
the committed generator for fig_component_spectra.

Mentor edits applied: (1) no super-title on top of the figure -- the node location
goes in the caption instead; (2) legend moved to the lower-left of the sigma_xx panel.

Data: synthetic_validation/figures_data/validation_fields.npz
  freqs                : (5001,) Hz
  x, y                 : (841,) m  (node coordinates)
  tru_SX/rec_SX ...     : (5001, 841) Pa^2/Hz  (also SY, SXY, and 'SX+SY' invariant)
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

BASE = _REPO
DATA = BASE / 'synthetic_validation' / 'figures_data'
FIG  = BASE / 'synthetic_validation' / 'figures'

d = np.load(DATA / 'validation_fields.npz', allow_pickle=True)
freqs = d['freqs']
x = d['x'] * 1000.0; y = d['y'] * 1000.0
cp = int(np.argmin((x - 38.0) ** 2 + (y - 54.0) ** 2))   # -> node 283 at ~(37.5, 53.6) mm
RES = [51.4, 187.6]                                       # the two base-active resonances [Hz]

TRUTH_C, TRUTH_LW, TRUTH_A = '0.60', 3.4, 0.55           # truth: thick translucent grey band
REC_C, REC_LW = '#0072B2', 0.9                            # recovered: thin blue line

COMPS = [('SX',    r'$\sigma_{xx}$'),
         ('SY',    r'$\sigma_{yy}$'),
         ('SXY',   r'$\tau_{xy}$'),
         ('SX+SY', r'$\sigma_{xx}+\sigma_{yy}$')]

fig, axes = plt.subplots(2, 2, figsize=(170 * MM_TO_INCH, 112 * MM_TO_INCH),
                         constrained_layout=True)

print(f'probe node cp={cp} at ({x[cp]:.1f}, {y[cp]:.1f}) mm')
for ax, (key, tex) in zip(axes.flat, COMPS):
    tru = d[f'tru_{key}'][:, cp]
    rec = d[f'rec_{key}'][:, cp]
    pos = tru[tru > 0]
    floor = 1e-3 * pos.min() if len(pos) else 1e-30
    for r in RES:
        ax.axvline(r, color='0.72', lw=0.6, ls=':', zorder=0)
    ax.semilogy(freqs, np.clip(tru, floor, None), '-', color=TRUTH_C, lw=TRUTH_LW,
                alpha=TRUTH_A, solid_capstyle='round', zorder=2, label='truth')
    ax.semilogy(freqs, np.clip(rec, floor, None), '-', color=REC_C, lw=REC_LW,
                zorder=3, label='recovered')
    ax.set_xlim(30, 250)
    ax.set_title(tex, fontsize=9)
    ax.tick_params(labelsize=7)
    rel = np.sqrt(np.sum((rec - tru) ** 2) / np.sum(tru ** 2)) * 100
    ax.text(0.97, 0.95, rf'rel-RMS ${rel:.1f}\%$', transform=ax.transAxes,
            va='top', ha='right', fontsize=7.5, color='0.40')
    print(f'  {key:6s} rel-RMS = {rel:.1f}%')

for ax in axes[:, 0]:
    ax.set_ylabel(r'PSD [Pa$^2$/Hz]', fontsize=8)
for ax in axes[1, :]:
    ax.set_xlabel(r'frequency [Hz]', fontsize=8)
for ax in axes[0, :]:
    ax.set_xticklabels([])

# legend in the sigma_xx panel, LOWER LEFT (mentor's request)
axes[0, 0].legend(loc='lower left', fontsize=7.5, frameon=False, ncol=1,
                  handlelength=1.5, handletextpad=0.5, borderaxespad=0.6)

fig.savefig(FIG / 'fig_component_spectra.pdf')
fig.savefig(FIG / 'fig_component_spectra.png')
print('saved fig_component_spectra')
plt.close(fig)
