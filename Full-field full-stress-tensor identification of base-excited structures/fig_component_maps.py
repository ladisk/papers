"""R1.11 (response letter only): full-field sigma_xx / sigma_yy / tau_xy PSD maps at a
RESONANT and an OFF-RESONANT frequency, truth vs recovered. Demonstrates the full-field
capability the reviewer asks to see, at both a resonance and off-resonance, without
committing off-resonant maps to the manuscript (they carry little fatigue information).

Per-frequency PSD fields (not band-RMS), from the same realistic SEMM-prior recovery as
every other validation figure. Colour scale is shared within each (component, frequency)
truth/recovered pair so both the spatial pattern and the truth-vs-recovered agreement are
visible; the resonant and off-resonant pairs use independent scales because the off-resonant
level is orders of magnitude lower.
"""
from pathlib import Path as _Path

# Repository root (this file's directory) - all paths below are relative to it.
_REPO = _Path(__file__).resolve().parent
(_REPO / 'synthetic_validation' / 'figures').mkdir(parents=True, exist_ok=True)

import os, sys, matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(_REPO))
from plot_style import apply_style, MM_TO_INCH
apply_style()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R = str(_REPO)
sys.path.insert(0, R)
from synthetic_validation.harness import run_case_from_modal

FIG = f'{R}/synthetic_validation/figures'
D = f'{R}/synthetic_validation/analysis/broad_analysis'
ACQ = dict(saa_level=1.0, fs=2000.0, n_frames=10000, seed=0)

def save_fig(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf'); fig.savefig(f'{FIG}/{name}.png'); print('saved', name)

ld = lambda n: {k: v for k, v in np.load(f'{D}/{n}.npz', allow_pickle=True).items()}
truth, prior = ld('modal_truth'), ld('modal_prior_semm')
res = run_case_from_modal(truth, prior, n_modes=2, **ACQ)
freqs = res['freqs']
x = truth['node_coords'][:, 0] * 1000.0
y = truth['node_coords'][:, 1] * 1000.0

# resonance = first-mode line; off-resonance = a line between the two modes
f_res = float(freqs[int(np.argmin(np.abs(freqs - 51.2)))])
f_off = float(freqs[int(np.argmin(np.abs(freqs - 110.0)))])
k_res = int(np.argmin(np.abs(freqs - f_res)))
k_off = int(np.argmin(np.abs(freqs - f_off)))

def to_grid(vals, x, y):
    xs = np.unique(np.round(x, 3)); ys = np.unique(np.round(y, 3))
    G = np.full((len(ys), len(xs)), np.nan)
    ix = np.clip(np.searchsorted(xs, np.round(x, 3)), 0, len(xs) - 1)
    iy = np.clip(np.searchsorted(ys, np.round(y, 3)), 0, len(ys) - 1)
    G[iy, ix] = vals
    return xs, ys, G

mass_box = [0.0, 21.4, 128.6, 150.0]     # top-left 2x2 cm block footprint (mm)
extent = [0, 150, 0, 150]
COMPS = [(r'$\sigma_{xx}$', 'SX'), (r'$\sigma_{yy}$', 'SY'), (r'$\tau_{xy}$', 'SXY')]

def relrms(a, b):
    a = np.abs(a); b = np.abs(b)
    return float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))

fig = plt.figure(figsize=(175 * MM_TO_INCH, 128 * MM_TO_INCH))
# cols: [truth@res, rec@res, cbar_res, spacer, truth@off, rec@off, cbar_off]
gs = fig.add_gridspec(3, 7, width_ratios=[1, 1, 0.06, 0.22, 1, 1, 0.06],
                      left=0.085, right=0.965, bottom=0.055, top=0.915,
                      wspace=0.08, hspace=0.12)

for r, (lab, c) in enumerate(COMPS):
    for pair, (kf, c0) in enumerate([(k_res, 0), (k_off, 4)]):
        tru = np.abs(res['truth'][c][kf]) / 1e12       # -> MPa^2/Hz
        rec = np.abs(res['recovered'][c][kf]) / 1e12
        vmax = float(np.nanmax([tru.max(), rec.max()]))
        _, _, Gt = to_grid(tru, x, y)
        _, _, Gr = to_grid(rec, x, y)
        axt = fig.add_subplot(gs[r, c0]); axr = fig.add_subplot(gs[r, c0 + 1])
        for ax, G in ((axt, Gt), (axr, Gr)):
            im = ax.imshow(G, origin='lower', extent=extent, aspect='equal',
                           cmap='viridis', vmin=0.0, vmax=vmax)
            ax.add_patch(Rectangle((mass_box[0], mass_box[2]),
                                   mass_box[1] - mass_box[0], mass_box[3] - mass_box[2],
                                   fill=False, edgecolor='white', linewidth=0.7))
            ax.plot([0, 150], [0, 0], color='white', linewidth=2.0,
                    solid_capstyle='butt', zorder=5)
            ax.set_xlim(0, 150); ax.set_ylim(0, 150)
            ax.set_xticks([]); ax.set_yticks([])
        cax = fig.add_subplot(gs[r, c0 + 2])
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6)
        if r == 0:
            axt.set_title('Truth', fontsize=8.5, pad=3)
            axr.set_title('Recovered', fontsize=8.5, pad=3)
        if r == 2:
            for ax in (axt, axr):
                ax.set_xticks([0, 75, 150]); ax.set_xticklabels(['0', '75', '150'], fontsize=6.5)
                ax.tick_params(axis='x', length=2)
        if c0 == 0:
            axt.set_ylabel(lab, fontsize=11, rotation=0, ha='right', va='center', labelpad=12)

fig.text(0.24, 0.955, r'Resonance ($f_1 = %.0f$~Hz)' % f_res, ha='center', fontsize=9)
fig.text(0.78, 0.955, r'Off-resonance ($%.0f$~Hz)' % f_off, ha='center', fontsize=9)
save_fig(fig, 'fig_component_maps')
