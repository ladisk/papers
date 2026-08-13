from pathlib import Path as _Path

# Repository root (this file's directory) - all paths below are relative to it.
_REPO = _Path(__file__).resolve().parent
(_REPO / 'synthetic_validation' / 'figures').mkdir(parents=True, exist_ok=True)

import sys, matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(_REPO))
from plot_style import apply_style, MM_TO_INCH
apply_style()

import numpy as np, json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG = str(_REPO / 'synthetic_validation' / 'figures')
def save_fig(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf')
    fig.savefig(f'{FIG}/{name}.png')
    print('saved', name)

S = json.load(open(str(_REPO / 'synthetic_validation' / 'figures_data' / 'fig_data_broad.json')))

sig = np.array(S['noise_sigma'], float)          # [0, 0.005, 0.01, 0.05, 0.1] K
# plot the noise-free (0 K) point at a small floor on the log axis
FLOOR = 1e-3
x = sig.copy()
x[x == 0] = FLOOR

comps  = ['SX', 'SY', 'SXY']
labels = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\tau_{xy}$']
colors = ['#0072B2', '#CC3311', '#009E73']  # blue / red / green -- clearly distinct

fig, ax = plt.subplots(figsize=(120 * MM_TO_INCH, 82 * MM_TO_INCH))

for comp, c in zip(comps, colors):
    yt = np.array(S['noise_trans'][comp]) * 100.0    # transmissibility (with accel)
    yd = np.array(S['noise_direct'][comp]) * 100.0   # direct-PSD (accel-free)
    ax.plot(x, yt, '-',  color=c, lw=1.3, marker='o', ms=3.6, mfc=c, mec=c, zorder=3)
    ax.plot(x, yd, '--', color=c, lw=1.3, marker='s', ms=3.6, mfc='none', mec=c, zorder=3)

ax.set_xscale('log')
ax.set_ylim(0, 145)

# vertical line: per-node signal SNR~1
snr = 0.013
ax.axvline(snr, color='0.5', ls=':', lw=0.7, zorder=1)
ax.text(snr * 1.12, 62, r'per-node signal (SNR$\approx$1)',
        rotation=90, va='center', ha='left', fontsize=6.5, color='0.45')

ax.set_xlabel(r'camera noise $\sigma_T$ [K]')
ax.set_ylabel(r'rel-RMS error, stress amplitude [\%]')

# x ticks: label the floor point 'noise-free'
ticks = [FLOOR, 0.005, 0.01, 0.05, 0.1]
ax.set_xticks(ticks)
ax.set_xticklabels(['noise-free', '0.005', '0.01', '0.05', '0.1'])
ax.set_xlim(FLOOR * 0.7, 0.15)
ax.minorticks_off()

# component-colour legend
leg1_handles = [Line2D([0], [0], color=c, lw=1.4) for c in colors]
leg1 = ax.legend(leg1_handles, labels, loc='upper left', title='component',
                 fontsize=7, title_fontsize=7, handlelength=1.6,
                 borderaxespad=0.4, labelspacing=0.3)
ax.add_artist(leg1)

# method (line-style) legend
leg2_handles = [
    Line2D([0], [0], color='0.2', lw=1.2, ls='-',  marker='o', ms=3.2, mfc='0.2', mec='0.2'),
    Line2D([0], [0], color='0.2', lw=1.2, ls='--', marker='s', ms=3.2, mfc='none', mec='0.2'),
]
leg2_labels = ['transmissibility (with accelerometer)', 'direct-PSD (accelerometer-free)']
ax.legend(leg2_handles, leg2_labels, loc='upper center', fontsize=7,
          handlelength=2.4, borderaxespad=0.4, labelspacing=0.3,
          bbox_to_anchor=(0.55, 1.0))

fig.tight_layout()
save_fig(fig, 'fig_noise_robustness')
