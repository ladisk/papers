"""
Fig — Noise robustness: transmissibility vs direct PSD.

Left panel (a):  sigma_xx rel. RMS error vs camera-noise % — two series.
Right panel (b): tau_xy  rel. RMS error vs camera-noise %.

Log y-axis: direct-PSD blows up (>500 % at 10 % noise); transmissibility
stays nearly flat (noise-robust thanks to the measured acceleration reference).

Data: method_contrast.npz
  noise_pct      : [0, 2, 5, 10] %
  trans_relrms   : (4, 2) — transmissibility, [SX, SXY], fraction
  direct_relrms  : (4, 2) — direct PSD,       [SX, SXY], fraction
"""
import sys
sys.path.insert(0, r'C:\Users\jasas\Work\Clanki\Clanek_2\Writing_article\article_bundle\scripts')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
d = np.load(DATA_DIR / 'method_contrast.npz', allow_pickle=True)
noise_pct     = d['noise_pct']           # (4,) [0, 2, 5, 10]
trans_relrms  = d['trans_relrms']        # (4, 2)
direct_relrms = d['direct_relrms']       # (4, 2)

# Convert fractions → %
trans_pct  = trans_relrms  * 100.
direct_pct = direct_relrms * 100.

print('noise_pct:', noise_pct)
print('trans  SX  [%]:', trans_pct[:, 0].round(2))
print('direct SX  [%]:', direct_pct[:, 0].round(2))
print('trans  SXY [%]:', trans_pct[:, 1].round(2))
print('direct SXY [%]:', direct_pct[:, 1].round(2))

# ── Colours (Okabe–Ito) ───────────────────────────────────────────────────
C_TRANS  = '#000000'   # black — transmissibility
C_DIRECT = '#0072B2'   # blue  — direct PSD

LS_TRANS  = '-'
LS_DIRECT = '--'

# ── Figure layout ─────────────────────────────────────────────────────────
fig_w = 170 * MM_TO_INCH
fig_h = 72  * MM_TO_INCH

fig, (ax_a, ax_b) = plt.subplots(1, 2,
                                  figsize=(fig_w, fig_h),
                                  sharey=False,
                                  gridspec_kw={'wspace': 0.38})

def plot_panel(ax, comp_idx, comp_tex, panel_letter):
    """Plot one noise-vs-relrms panel."""
    ax.semilogy(noise_pct, trans_pct[:, comp_idx],
                color=C_TRANS, ls=LS_TRANS, lw=1.0, marker='o', ms=3.5,
                label=r'Transmissibility')
    ax.semilogy(noise_pct, direct_pct[:, comp_idx],
                color=C_DIRECT, ls=LS_DIRECT, lw=1.0, marker='s', ms=3.5,
                label=r'Direct PSD')

    ax.set_xlabel(r'Camera noise [\%]', fontsize=8)
    ax.set_ylabel(r'Rel.\ RMS error [\%]', fontsize=8)
    ax.set_xlim(-0.3, 10.3)
    ax.set_xticks(noise_pct)
    ax.tick_params(labelsize=7.5)
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10,
                                                                 labelOnlyBase=False))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs='auto', numticks=10))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    title = f'({panel_letter}) ' + comp_tex
    ax.set_title(title, fontsize=8, pad=4)

    if panel_letter == 'a':
        ax.legend(fontsize=7.5, loc='upper left')

# Panel (a): sigma_xx
plot_panel(ax_a, comp_idx=0, comp_tex=r'$\sigma_{xx}$', panel_letter='a')

# Panel (b): tau_xy
plot_panel(ax_b, comp_idx=1, comp_tex=r'$\tau_{xy}$', panel_letter='b')

# ── Align y-axes for readability ──────────────────────────────────────────
# Let each panel pick its own limits (ranges differ by orders of magnitude).
ax_a.set_ylim(1e1, 1e3)
ax_b.set_ylim(1e1, 1e5)

fig.canvas.draw()
save_fig(fig, 'fig_method_contrast')
plt.close(fig)
print('Done.')
