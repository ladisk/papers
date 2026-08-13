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
S = json.load(open(str(_REPO / 'synthetic_validation' / 'figures_data' / 'fig_data_broad.json')))

x = d['x']; y = d['y']
mass_box = d['mass_box']  # [x0,x1,y0,y1]

def to_grid(vals, x, y):
    xs = np.unique(np.round(x, 3)); ys = np.unique(np.round(y, 3))
    G = np.full((len(ys), len(xs)), np.nan)
    ix = np.clip(np.searchsorted(xs, np.round(x, 3)), 0, len(xs)-1)
    iy = np.clip(np.searchsorted(ys, np.round(y, 3)), 0, len(ys)-1)
    G[iy, ix] = vals
    return xs, ys, G

# Rows: (label, truth_key, rec_key, mac_key)
rows = [
    (r'$\sigma_{xx}$', 'tru_SX',    'rec_SX',    'SX'),
    (r'$\sigma_{yy}$', 'tru_SY',    'rec_SY',    'SY'),
    (r'$\tau_{xy}$',   'tru_SXY',   'rec_SXY',   'SXY'),
    (r'$\sigma_{\mathrm{vM}}$', 'tru_VM', 'rec_VM', 'VM'),
]

nrows = len(rows)
# Width ~120 mm, tall
fig_w = 120 * MM_TO_INCH
fig_h = 205 * MM_TO_INCH
fig = plt.figure(figsize=(fig_w, fig_h))

# GridSpec: 2 map columns + 1 colorbar column
gs = fig.add_gridspec(nrows, 3, width_ratios=[1, 1, 0.06],
                      left=0.11, right=0.90, bottom=0.045, top=0.965,
                      wspace=0.08, hspace=0.10)

extent = [0, 150, 0, 150]

for r, (lab, tk, rk, mk) in enumerate(rows):
    tru = d[tk] / 1e6
    rec = d[rk] / 1e6
    xs, ys, Gt = to_grid(tru, x, y)
    _,  _,  Gr = to_grid(rec, x, y)
    vmax = np.nanmax([np.nanmax(Gt), np.nanmax(Gr)])
    vmin = 0.0

    axt = fig.add_subplot(gs[r, 0])
    axr = fig.add_subplot(gs[r, 1])
    for ax, G in ((axt, Gt), (axr, Gr)):
        im = ax.imshow(G, origin='lower', extent=extent, aspect='equal',
                       cmap='viridis', vmin=vmin, vmax=vmax)
        # mass block footprint (top-left)
        ax.add_patch(Rectangle((mass_box[0], mass_box[2]),
                               mass_box[1]-mass_box[0], mass_box[3]-mass_box[2],
                               fill=False, edgecolor='white', linewidth=0.8))
        # clamped bottom edge
        ax.plot([0, 150], [0, 0], color='white', linewidth=2.2,
                solid_capstyle='butt', zorder=5)
        ax.set_xlim(0, 150); ax.set_ylim(0, 150)
        ax.set_xticks([]); ax.set_yticks([])

    # MAC annotation (in-panel, white) on recovered panel — top-right,
    # away from the top-left mass-box outline
    axr.text(0.96, 0.95, r'MAC$=%.3f$' % S['mac'][mk],
             transform=axr.transAxes, color='white', va='top', ha='right',
             fontsize=7.5)

    # colorbar per row
    cax = fig.add_subplot(gs[r, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('[MPa]', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # row label on the left
    axt.set_ylabel(lab, fontsize=11, rotation=0, ha='right', va='center',
                   labelpad=14)

    # column headers on top row
    if r == 0:
        axt.set_title('Truth', fontsize=9, pad=4)
        axr.set_title('Recovered', fontsize=9, pad=4)

    # clamped-edge label once (top-left panel)
    if r == 0:
        axt.text(0.5, -0.02, 'clamped edge', transform=axt.transAxes,
                 color='0.25', va='top', ha='center', fontsize=6.5)

    # mm scale on bottom row
    if r == nrows - 1:
        for ax in (axt, axr):
            ax.set_xticks([0, 75, 150])
            ax.set_xticklabels(['0', '75', '150'], fontsize=7)
            ax.tick_params(axis='x', length=2)
        axt.set_yticks([0, 75, 150])
        axt.set_yticklabels(['0', '75', '150'], fontsize=7)
        axt.tick_params(axis='y', length=2)
        axt.set_xlabel('[mm]', fontsize=8)
        axr.set_xlabel('[mm]', fontsize=8)

save_fig(fig, 'fig_tensor_maps')
