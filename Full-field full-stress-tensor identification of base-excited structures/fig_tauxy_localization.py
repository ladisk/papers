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
from matplotlib.patches import Rectangle

FIG = str(_REPO / 'synthetic_validation' / 'figures')

def save_fig(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf')
    fig.savefig(f'{FIG}/{name}.png')
    print('saved', name)

d = np.load(str(_REPO / 'synthetic_validation' / 'figures_data' / 'fig_data_broad.npz'), allow_pickle=True)

x = d['x']; y = d['y']
tru_SXY = d['tru_SXY']
sxy_relerr = d['sxy_relerr']
top10 = np.array(d['top10_inv'])
mass_box = d['mass_box']  # [x0, x1, y0, y1]


def to_grid(vals, x, y):
    xs = np.unique(np.round(x, 3)); ys = np.unique(np.round(y, 3))
    G = np.full((len(ys), len(xs)), np.nan)
    ix = np.clip(np.searchsorted(xs, np.round(x, 3)), 0, len(xs) - 1)
    iy = np.clip(np.searchsorted(ys, np.round(y, 3)), 0, len(ys) - 1)
    G[iy, ix] = vals
    return xs, ys, G


# (a) truth tau_xy band-RMS field in MPa
xs, ys, G_tru = to_grid(tru_SXY / 1e6, x, y)
# (b) relative error in percent
xs2, ys2, G_err = to_grid(sxy_relerr * 100.0, x, y)

x0, x1, y0, y1 = mass_box
xt, yt = x[top10], y[top10]

fig, axes = plt.subplots(1, 2, figsize=(170 * MM_TO_INCH, 82 * MM_TO_INCH))

extent = [0, 150, 0, 150]

# ---- Panel (a) ----
axa = axes[0]
im0 = axa.imshow(G_tru, origin='lower', extent=extent, aspect='equal',
                 cmap='viridis')
axa.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                        edgecolor='white', linewidth=0.7))
axa.scatter(xt, yt, s=2.5, c='white', edgecolors='none', alpha=0.9, zorder=3)
axa.set_title(r'(a) true $\tau_{xy}$ field')
axa.set_xlabel(r'$x$ [mm]')
axa.set_ylabel(r'$y$ [mm]')
axa.set_xlim(0, 150); axa.set_ylim(0, 150)
axa.set_xticks([0, 50, 100, 150]); axa.set_yticks([0, 50, 100, 150])
cb0 = fig.colorbar(im0, ax=axa, fraction=0.046, pad=0.04)
cb0.set_label(r'[MPa]')

# ---- Panel (b) ----
axb = axes[1]
im1 = axb.imshow(G_err, origin='lower', extent=extent, aspect='equal',
                 cmap='inferno', vmin=0, vmax=60)
axb.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                        edgecolor='white', linewidth=0.7))
axb.scatter(xt, yt, s=2.5, c='white', edgecolors='none', alpha=0.9, zorder=3)
axb.set_title(r'(b) $\tau_{xy}$ recovery error')
axb.set_xlabel(r'$x$ [mm]')
axb.set_ylabel(r'$y$ [mm]')
axb.set_xlim(0, 150); axb.set_ylim(0, 150)
axb.set_xticks([0, 50, 100, 150]); axb.set_yticks([0, 50, 100, 150])
cb1 = fig.colorbar(im1, ax=axb, fraction=0.046, pad=0.04)
cb1.set_label(r'[\%]')

axb.annotate('error rings the corner block;\nthe fatigue-critical clamp edge is clean',
             xy=(0.5, -0.30), xycoords='axes fraction', ha='center', va='top',
             fontsize=7.5)

fig.subplots_adjust(left=0.07, right=0.95, bottom=0.24, top=0.90, wspace=0.35)
save_fig(fig, 'fig_tauxy_localization')
