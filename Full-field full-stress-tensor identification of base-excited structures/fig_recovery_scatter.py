"""
Fig — Recovered vs True stress-PSD agreement scatter at resonance.

Log-log scatter of True (x) vs Recovered (y) stress PSD across all 841 nodes,
for sigma_xx, sigma_yy, tau_xy. 1:1 line = perfect recovery. Legend shows
rel-RMS per component. Colorblind-safe Okabe-Ito palette.

NOTE on sorting: coordinates are rounded before lexsort to form correct grid,
but the scatter uses all 841 nodes directly (no reshaping needed).

Data: synthetic_validation/figures_data/validation_fields.npz
  rec_SX, tru_SX etc. : (5001, 841) Pa^2/Hz
  fk                  : int, resonance frequency index
  metrics_relrms       : (4,) rel-RMS values
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

# ── Paths ─────────────────────────────────────────────────────────────────
BASE     = _REPO
DATA_DIR = BASE / 'synthetic_validation' / 'figures_data'
FIG_DIR  = BASE / 'synthetic_validation' / 'figures'

def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{name}.pdf')
    fig.savefig(FIG_DIR / f'{name}.png')
    print(f'Saved: {name}.pdf, {name}.png')

# ── Load data ─────────────────────────────────────────────────────────────
d    = np.load(DATA_DIR / 'validation_fields.npz', allow_pickle=True)
fk   = int(d['fk'])

metrics_rr = d['metrics_relrms']  # [SX, SY, SXY, SX+SY]

# Okabe-Ito colorblind-safe palette: blue, vermilion, bluish-green
COLORS = ['#0072B2', '#D55E00', '#009E73']

COMPS = [
    ('SX',  r'$\sigma_{xx}$', 0),
    ('SY',  r'$\sigma_{yy}$', 1),
    ('SXY', r'$\tau_{xy}$',   2),
]

# Probe node highlighted in Fig V3 (off-axis, all three components active).
PROBE = 113

# ── Figure ─────────────────────────────────────────────────────────────────
fig_w = 83 * MM_TO_INCH   # Elsevier single column
fig_h = 80 * MM_TO_INCH

fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

all_vals = []   # collect all plotted values for axis limits

for (tag, tex, midx), color in zip(COMPS, COLORS):
    rec = d[f'rec_{tag}'][fk, :]   # (841,) Pa^2/Hz at resonance
    tru = d[f'tru_{tag}'][fk, :]

    # Filter: keep finite positive pairs AND true PSD > 0.1 % of component max
    # This removes symmetry-node zeros (where mode shape is zero by construction)
    # while preserving all physically meaningful nodes.
    tru_thresh = 1e-3 * np.nanmax(tru)
    mask = (np.isfinite(rec) & np.isfinite(tru)
            & (rec > 0) & (tru > tru_thresh))
    rec_m = rec[mask]
    tru_m = tru[mask]
    all_vals.extend([rec_m, tru_m])
    print(f'  {tag}: {mask.sum()} / 841 nodes shown (tru thresh={tru_thresh:.2e})')

    rr_pct = metrics_rr[midx] * 100
    label  = tex + rf', rel-RMS $= {rr_pct:.1f}\%$'

    ax.scatter(tru_m, rec_m, s=5, color=color, alpha=0.65,
               linewidths=0, label=label, rasterized=True)

    # Highlight the Fig V3 probe node in this component's colour
    ax.scatter(tru[PROBE], rec[PROBE], s=55, color=color,
               edgecolors='k', linewidths=0.9, marker='*', zorder=6)

print(f'Plotted {len(all_vals)//2} component datasets at resonance index {fk}')

# 1:1 reference line
all_v = np.concatenate(all_vals)
lo = 10 ** np.floor(np.log10(all_v.min()))
hi = 10 ** np.ceil (np.log10(all_v.max()))
ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='1:1 line', zorder=5)

# Legend proxy for the probe-node star (component-neutral, black-edged)
from matplotlib.lines import Line2D
probe_handle = Line2D([0], [0], marker='*', color='0.4', markeredgecolor='k',
                      markeredgewidth=0.9, markersize=8, linestyle='None',
                      label='Fig.\\,V3 probe node')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect('equal')

ax.set_xlabel(r'True stress PSD [Pa$^2$/Hz]',       fontsize=8)
ax.set_ylabel(r'Recovered stress PSD [Pa$^2$/Hz]',  fontsize=8)
ax.tick_params(labelsize=7, width=0.4, length=2.5)
ax.tick_params(which='minor', width=0.3, length=1.5)

handles, labels = ax.get_legend_handles_labels()
handles.append(probe_handle)
labels.append(probe_handle.get_label())
ax.legend(handles, labels, fontsize=6.5, loc='upper left',
          handlelength=1.2, handletextpad=0.5)

fig.tight_layout(pad=0.5)
fig.canvas.draw()
save_fig(fig, 'fig_recovery_scatter')
plt.close(fig)
print('Done.')
