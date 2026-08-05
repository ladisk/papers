"""
Fig — Full-field stress-PSD spatial maps at resonance.

2x2 pcolormesh panels on the 31x31 node grid:
  (a) sigma_xx,  (b) sigma_yy
  (c) tau_xy,    (d) sigma_xx + sigma_yy  (first stress invariant)

Sequential (viridis) colormap — PSDs are non-negative.
Individual colorbars per panel (ranges differ significantly between components).
Axes in mm; equal aspect.

Data: spatial_fields.npz
  x, y       : (961,) node coordinates [m]  (31x31 grid)
  sigxx, sigyy, tauxy, invariant : (961,) stress-PSD [Pa^2/Hz] at resonance
  freq       : resonance frequency [Hz]
"""
import sys
sys.path.insert(0, r'C:\Users\jasas\Work\Clanki\Clanek_2\Writing_article\article_bundle\scripts')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from plot_style import apply_style, MM_TO_INCH

apply_style()

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(r'C:\Users\jasas\Work\Clanki\Clanek_2\thermoelastic-stress-expansion'
                r'\synthetic_validation\figures_data')
FIG_DIR  = Path(r'C:\Users\jasas\Work\Clanki\Clanek_2\thermoelastic-stress-expansion'
                r'\synthetic_validation\figures')

def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{name}.pdf')
    fig.savefig(FIG_DIR / f'{name}.png')
    print(f'Saved: {name}.pdf, {name}.png')

# ── Load data ─────────────────────────────────────────────────────────────
d    = np.load(DATA_DIR / 'spatial_fields.npz', allow_pickle=True)
x    = d['x']         # (961,) metres
y    = d['y']         # (961,) metres
sigxx     = d['sigxx']       # (961,) Pa^2/Hz
sigyy     = d['sigyy']
tauxy     = d['tauxy']
invariant = d['invariant']   # sigxx + sigyy
freq      = float(d['freq'])

N = 31

# ── Sort nodes into regular (N, N) grid — row-major (y ascending, then x) ─
sort_idx = np.lexsort((x, y))   # y first (ascending), then x

x_mm = x[sort_idx].reshape(N, N) * 1e3   # metres → mm
y_mm = y[sort_idx].reshape(N, N) * 1e3

def sort_and_reshape(arr):
    return arr[sort_idx].reshape(N, N)

Z = {
    r'$\sigma_{xx}$': sort_and_reshape(sigxx),
    r'$\sigma_{yy}$': sort_and_reshape(sigyy),
    r'$\tau_{xy}$':   sort_and_reshape(tauxy),
    r'$\sigma_{xx}+\sigma_{yy}$': sort_and_reshape(invariant),
}
keys = list(Z.keys())

print(f'Resonance freq: {freq:.2f} Hz')
for k, v in Z.items():
    print(f'  {k}: min={v.min():.3e}, max={v.max():.3e} Pa^2/Hz')

# ── Figure layout ─────────────────────────────────────────────────────────
fig_w = 170 * MM_TO_INCH
fig_h = 115 * MM_TO_INCH

fig, axes = plt.subplots(2, 2,
                          figsize=(fig_w, fig_h),
                          gridspec_kw={'hspace': 0.50, 'wspace': 0.55})
panel_letters = [['a', 'b'], ['c', 'd']]

for row in range(2):
    for col in range(2):
        ax  = axes[row, col]
        key = keys[row * 2 + col]
        Z_  = Z[key]

        # 99th-percentile vmax to clip any isolated hot pixels
        vmax = float(np.percentile(Z_[np.isfinite(Z_)], 99.5))
        vmax = max(vmax, 1.0)  # guard against zero

        pcm = ax.pcolormesh(x_mm, y_mm, Z_,
                            cmap='viridis', shading='gouraud',
                            vmin=0, vmax=vmax, rasterized=True)
        ax.set_aspect('equal')

        ax.set_xlim(x_mm.min(), x_mm.max())
        ax.set_ylim(y_mm.min(), y_mm.max())

        # Ticks
        x_range = x_mm.max() - x_mm.min()
        y_range = y_mm.max() - y_mm.min()
        x_step  = round(x_range / 2 / 5) * 5
        y_step  = round(y_range / 2 / 5) * 5
        ax.set_xticks(np.arange(np.ceil(x_mm.min()/x_step)*x_step,
                                x_mm.max() + 1, x_step))
        ax.set_yticks(np.arange(np.ceil(y_mm.min()/y_step)*y_step,
                                y_mm.max() + 1, y_step))
        ax.tick_params(labelsize=6.5, width=0.3, length=2)

        # Axis labels
        if row == 1:
            ax.set_xlabel(r'$x$ [mm]', fontsize=7.5)
        if col == 0:
            ax.set_ylabel(r'$y$ [mm]', fontsize=7.5)

        # Panel title
        letter = panel_letters[row][col]
        ax.set_title(f'({letter}) ' + key, fontsize=8, pad=4)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.06)
        cb = fig.colorbar(pcm, cax=cax)
        cb.ax.tick_params(labelsize=6, width=0.3, length=1.5, pad=1)
        cb.outline.set_linewidth(0.3)
        cb.locator = ticker.MaxNLocator(nbins=4)
        cb.formatter = ticker.ScalarFormatter(useMathText=True)
        cb.formatter.set_powerlimits((-1, 2))
        cb.update_ticks()
        cb.ax.yaxis.get_offset_text().set_fontsize(5.5)
        cb.set_label(r'[Pa$^2$/Hz]', fontsize=6.5, labelpad=2)

# ── Frequency title ────────────────────────────────────────────────────────
fig.suptitle(f'Stress-PSD maps at $f = {freq:.1f}$ Hz',
             fontsize=9, y=0.98)

fig.canvas.draw()
save_fig(fig, 'fig_stress_fields')
plt.close(fig)
print('Done.')
