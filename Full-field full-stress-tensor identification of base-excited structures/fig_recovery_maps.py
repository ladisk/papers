"""
Fig — Recovered vs True stress-PSD spatial maps at resonance (29×29 grid).

Layout: 3 rows (sigma_xx, sigma_yy, tau_xy) x 3 cols (Recovered | Truth | |Error|).
At resonance freqs[fk]. Rec+Truth share per-row colour scale (viridis);
Error column uses a separate scale. Axes in mm, equal aspect.
Component label + metrics (rel-RMS %, MAC) annotated inside each Recovered panel.

NOTE on sorting: node coordinates have float-precision variation within each
nominal x/y level; rounding to 5 d.p. before lexsort restores the correct
29x29 Cartesian grid ordering.

Data: synthetic_validation/figures_data/validation_fields.npz
  x, y          : (841,) node coords [m]  (29x29 grid)
  freqs          : (5001,) Hz
  fk             : int, resonance frequency index
  rec_SX, tru_SX etc. : (5001, 841) Pa^2/Hz
  metrics_relrms, metrics_mac : (4,) aligned to ['SX','SY','SXY','SX+SY']
"""
import sys
sys.path.insert(0, r'C:\Users\jasas\Work\Clanki\Clanek_2\Writing_article\article_bundle\scripts')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from pathlib import Path
from plot_style import apply_style, MM_TO_INCH

apply_style()

# ── Paths ─────────────────────────────────────────────────────────────────
BASE     = Path(r'C:\Users\jasas\Work\Clanki\Clanek_2\thermoelastic-stress-expansion')
DATA_DIR = BASE / 'synthetic_validation' / 'figures_data'
FIG_DIR  = BASE / 'synthetic_validation' / 'figures'

def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{name}.pdf')
    fig.savefig(FIG_DIR / f'{name}.png')
    print(f'Saved: {name}.pdf, {name}.png')

# ── Load data ─────────────────────────────────────────────────────────────
d       = np.load(DATA_DIR / 'validation_fields.npz', allow_pickle=True)
x       = d['x']         # (841,) metres
y       = d['y']
freqs   = d['freqs']     # (5001,) Hz
fk      = int(d['fk'])   # resonance freq index
freq_r  = freqs[fk]

metrics_rr  = d['metrics_relrms']  # [SX, SY, SXY, SX+SY]
metrics_mac = d['metrics_mac']

# Sort 841 nodes into regular 29×29 grid.
# IMPORTANT: round coordinates first to avoid float-precision interleaving.
N        = 29
y_r      = np.round(y, 5)
x_r      = np.round(x, 5)
sort_idx = np.lexsort((x_r, y_r))   # y ascending (outer), x ascending (inner)
x_mm     = x_r[sort_idx].reshape(N, N) * 1e3   # metres -> mm
y_mm     = y_r[sort_idx].reshape(N, N) * 1e3
X_ext    = [x_mm[0, 0], x_mm[0, -1], y_mm[0, 0], y_mm[-1, 0]]  # extent for imshow

def at_resonance(tag):
    """Return (rec, tru) grids at resonance, shaped (N, N)."""
    rec_raw = d['rec_SX+SY'] if tag == 'SX+SY' else d[f'rec_{tag}']
    rec = rec_raw[fk, sort_idx].reshape(N, N)
    tru = d[f'tru_{tag}'][fk, sort_idx].reshape(N, N)
    return rec, tru

# Three tensor components; metrics_idx maps to metrics arrays
ROWS = [
    ('SX',  r'$\sigma_{xx}$', 0),
    ('SY',  r'$\sigma_{yy}$', 1),
    ('SXY', r'$\tau_{xy}$',   2),
]

print(f'Resonance: {freq_r:.2f} Hz  (index {fk})')
for tag, tex, midx in ROWS:
    rec, tru = at_resonance(tag)
    rr  = metrics_rr[midx]  * 100
    mac = metrics_mac[midx]
    print(f'  {tag}: rel-RMS={rr:.2f}%, MAC={mac:.5f}, '
          f'rec range [{rec.min():.3e}, {rec.max():.3e}] Pa^2/Hz')

# ── Figure layout ─────────────────────────────────────────────────────────
# GridSpec: 3 rows x (Rec | Tru | cbar_RT | Err | cbar_E)
# figsize chosen so each plot cell ≈ square (plate is 150×150 mm).
fig_w = 170 * MM_TO_INCH
fig_h = 178 * MM_TO_INCH

fig = plt.figure(figsize=(fig_w, fig_h))
gs  = GridSpec(3, 5, figure=fig,
               width_ratios=[1, 1, 0.06, 1, 0.06],
               left=0.07, right=0.985,
               bottom=0.06, top=0.93,
               hspace=0.22, wspace=0.12)

def _style_cb(cb, label):
    """Consistent colorbar styling."""
    cb.ax.tick_params(labelsize=6, width=0.3, length=1.5, pad=1)
    cb.outline.set_linewidth(0.3)
    cb.locator   = ticker.MaxNLocator(nbins=4)
    cb.formatter = ticker.ScalarFormatter(useMathText=True)
    cb.formatter.set_powerlimits((-1, 2))
    cb.update_ticks()
    cb.ax.yaxis.get_offset_text().set_fontsize(5.5)
    cb.set_label(label, fontsize=6.5, labelpad=2)

# imshow extent: [xmin, xmax, ymin, ymax] in mm
extent_mm = [x_mm[0, 0], x_mm[0, -1], y_mm[0, 0], y_mm[-1, 0]]

for row, (tag, tex, midx) in enumerate(ROWS):
    rec, tru = at_resonance(tag)
    err = np.abs(rec - tru)

    # Shared vmax for Rec+Truth (99.5th percentile to clip hot pixels)
    vals_rt = np.concatenate([rec.ravel(), tru.ravel()])
    vmax_rt = float(np.percentile(vals_rt[np.isfinite(vals_rt)], 99.5))
    vmax_rt = max(vmax_rt, 1.0)

    vmax_e = float(np.percentile(err.ravel(), 99.5))
    vmax_e = max(vmax_e, 1.0)

    # Axes
    ax_r   = fig.add_subplot(gs[row, 0])
    ax_t   = fig.add_subplot(gs[row, 1])
    cax_rt = fig.add_subplot(gs[row, 2])
    ax_e   = fig.add_subplot(gs[row, 3])
    cax_e  = fig.add_subplot(gs[row, 4])

    # Maps via imshow (regular grid — cleaner than pcolormesh for near-regular mesh)
    kw_rt = dict(origin='lower', extent=extent_mm, aspect='equal',
                 cmap='viridis', vmin=0, vmax=vmax_rt,
                 interpolation='bilinear', rasterized=True)
    kw_e  = dict(origin='lower', extent=extent_mm, aspect='equal',
                 cmap='viridis', vmin=0, vmax=vmax_e,
                 interpolation='bilinear', rasterized=True)

    im_r = ax_r.imshow(rec, **kw_rt)
    im_t = ax_t.imshow(tru, **kw_rt)
    im_e = ax_e.imshow(err, **kw_e)

    # Axes cosmetics
    for ax in (ax_r, ax_t, ax_e):
        ax.set_xlim(extent_mm[0], extent_mm[1])
        ax.set_ylim(extent_mm[2], extent_mm[3])
        ax.set_xticks([0, 50, 100, 150])
        ax.set_yticks([0, 50, 100, 150])
        ax.tick_params(labelsize=6.5, width=0.3, length=2)

    # x-axis labels: bottom row only
    if row == 2:
        for ax in (ax_r, ax_t, ax_e):
            ax.set_xlabel(r'$x$ [mm]', fontsize=7.5)
    else:
        for ax in (ax_r, ax_t, ax_e):
            ax.set_xticklabels([])

    # y-axis label and ticks: left column only
    ax_r.set_ylabel(r'$y$ [mm]', fontsize=7.5)
    ax_t.set_yticklabels([])
    ax_e.set_yticklabels([])

    # Column titles: top row only
    if row == 0:
        ax_r.set_title('Recovered', fontsize=8, pad=4)
        ax_t.set_title('Truth',     fontsize=8, pad=4)
        ax_e.set_title(r'$|\mathrm{Rec} - \mathrm{Truth}|$', fontsize=8, pad=4)

    # Component label: top-left of Recovered panel, white text on dark bg
    ax_r.text(0.03, 0.97, tex, transform=ax_r.transAxes,
              va='top', ha='left', fontsize=9.0, color='white', clip_on=False,
              bbox=dict(facecolor='#00000088', edgecolor='none',
                        boxstyle='round,pad=0.25'))

    # Metrics: bottom-right of Recovered panel
    rr  = metrics_rr[midx]  * 100
    mac = metrics_mac[midx]
    met_txt = rf'rel-RMS $= {rr:.1f}\%$' + '\n' + rf'MAC $= {mac:.4f}$'
    ax_r.text(0.97, 0.03, met_txt, transform=ax_r.transAxes,
              va='bottom', ha='right', fontsize=6.0, color='white', clip_on=False,
              linespacing=1.4,
              bbox=dict(facecolor='#00000088', edgecolor='none',
                        boxstyle='round,pad=0.25'))

    # Colorbars
    cb_rt = fig.colorbar(im_t, cax=cax_rt)
    _style_cb(cb_rt, r'[Pa$^2$/Hz]')

    cb_e = fig.colorbar(im_e, cax=cax_e)
    _style_cb(cb_e, r'[Pa$^2$/Hz]')

# Suptitle
fig.suptitle(rf'Stress-PSD recovery at $f = {freq_r:.1f}$ Hz',
             fontsize=9, y=0.97)

fig.canvas.draw()
save_fig(fig, 'fig_recovery_maps')
plt.close(fig)
print('Done.')
