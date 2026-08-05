"""
Fig — Recovered vs True stress-PSD spectra at the critical node (3-panel).

Three vertically stacked panels, one per component (sigma_xx, sigma_yy, tau_xy),
at a representative probe node where ALL THREE components are physically active.
Each panel has its own y-scale.
Truth: thick translucent band; Recovered: thin dark line on top. When they agree
the dark line sits inside the band, so the reader sees the match at a glance.
Resonance frequency marked in every panel with a thin dotted vertical line.

Note: the maps-figure critical node (cp) sits on the mode symmetry axis, where the
true tau_xy is identically zero -- comparing recovery there is meaningless. This
spectra figure instead probes an off-axis node (PROBE) where tau_xy is strongly
excited, so all three components have a genuine truth curve to track.

Data: synthetic_validation/figures_data/validation_fields.npz
  freqs              : (5001,) Hz
  fk                 : int, resonance index
  cp                 : int, critical node index
  rec_SX, tru_SX etc.: (5001, 841) Pa^2/Hz
"""
import sys
sys.path.insert(0, r'C:\Users\jasas\Work\Clanki\Clanek_2\Writing_article\article_bundle\scripts')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
d     = np.load(DATA_DIR / 'validation_fields.npz', allow_pickle=True)
freqs = d['freqs']      # (5001,) Hz
fk    = int(d['fk'])    # resonance index
freq_r = freqs[fk]

# Probe node for the spectra. The maps-figure critical node d['cp'] is the
# maximum-stress point, but it lies on the mode symmetry axis where true tau_xy == 0,
# so shear recovery cannot be shown there. Objective rule instead: the most
# shear-active INTERIOR node (top-5 tau_xy nodes all lie on the outer edge). That is
# node 113 (x=5.4 mm, y=10.7 mm) -- tau_xy in the top ~1% of the field, first stress
# invariant in the top ~19%. Broadband recovery here: 5.1% / 3.3% / 1.3% (SX/SY/SXY).
# NOTE: this single-node spectrum is ILLUSTRATIVE; the quantitative accuracy claim is
# the field-wide error (Figs V1-V2: 3.0% / 2.5% / 8.9%), which includes worse nodes.
cp    = 113
x_mm  = float(d['x'][cp]) * 1000.0
y_mm  = float(d['y'][cp]) * 1000.0

# Okabe-Ito palette: blue, vermilion, bluish-green
COLORS = ['#0072B2', '#D55E00', '#009E73']

COMPS = [
    ('SX',  r'$\sigma_{xx}$', 0),
    ('SY',  r'$\sigma_{yy}$', 1),
    ('SXY', r'$\tau_{xy}$',   2),
]

# Shared style for style legend (placed once in top panel):
# Truth = thick translucent band, Recovered = thin dark line on top.
TRUTH_LW = 3.4
TRUTH_ALPHA = 0.35
REC_COLOR = '0.12'
REC_LW = 0.9
style_handles = [
    Line2D([0], [0], color='#0072B2', lw=TRUTH_LW, alpha=TRUTH_ALPHA, ls='-',
           label='Truth', solid_capstyle='round'),
    Line2D([0], [0], color=REC_COLOR, lw=REC_LW, ls='-', label='Recovered'),
]

print(f'Probe node: cp={cp} at ({x_mm:.1f}, {y_mm:.1f}) mm,  resonance: {freq_r:.2f} Hz')
for tag, tex, midx in COMPS:
    tru_cp = d[f'tru_{tag}'][:, cp]
    rec_cp = d[f'rec_{tag}'][:, cp]
    print(f'  {tag}: tru_peak={tru_cp.max():.3e}, rec_peak={rec_cp.max():.3e} Pa^2/Hz')

# ── Figure: 3-panel vertical layout ───────────────────────────────────────
fig_w = 120 * MM_TO_INCH   # 1.5-column
fig_h = 140 * MM_TO_INCH   # 3 stacked panels

fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h),
                          constrained_layout=True)
res_txt = None  # handle to the resonance annotation, repositioned after layout

for ax, (tag, tex, midx), color in zip(axes, COMPS, COLORS):
    tru_cp = d[f'tru_{tag}'][:, cp]
    rec_cp = d[f'rec_{tag}'][:, cp]

    # Positive floor: clip at 1e-3 × min positive value to avoid log(0) issues
    pos_vals = tru_cp[tru_cp > 0]
    floor = 1e-3 * pos_vals.min() if len(pos_vals) > 0 else 1e-30
    tru_p = np.clip(tru_cp, floor, None)
    rec_p = np.clip(rec_cp, floor, None)

    # Truth = thick translucent band underneath; Recovered = thin dark line on top.
    ax.semilogy(freqs, tru_p, '-',  color=color, lw=TRUTH_LW, alpha=TRUTH_ALPHA,
                solid_capstyle='round', zorder=2, label='Truth')
    ax.semilogy(freqs, rec_p, '-',  color=REC_COLOR, lw=REC_LW, alpha=0.95,
                zorder=3, label='Recovered')

    # Resonance marker
    ax.axvline(freq_r, color='0.50', lw=0.6, ls=':', zorder=0)

    ax.set_xlim(0, freqs[-1])
    ax.tick_params(labelsize=7, width=0.4, length=2.5)
    ax.tick_params(which='minor', width=0.3, length=1.5)
    ax.set_ylabel(r'PSD [Pa$^2$/Hz]', fontsize=7.5)

    # Panel label: component name + (a/b/c)
    letters = ['(a)', '(b)', '(c)']
    panel_letter = letters[midx]
    ax.set_title(f'{panel_letter} {tex}', fontsize=8.5, loc='left', pad=3)

    # Annotate resonance frequency and probe location on top panel only
    if midx == 0:
        ylo, yhi = ax.get_ylim()
        res_txt = ax.text(freq_r + 15, yhi,
                rf'$f = {freq_r:.1f}$ Hz,  probe at $({x_mm:.0f},\,{y_mm:.0f})$ mm',
                fontsize=6.5, color='0.5', va='top', ha='left')

    # Legend: style legend in top panel, no legend in other panels
    if midx == 0:
        ax.legend(handles=style_handles,
                  loc='upper right', fontsize=7,
                  handlelength=1.2, handletextpad=0.5,
                  frameon=False, ncol=2,
                  borderpad=0.3, columnspacing=0.8)

    # Report broadband recovery accuracy of this component at the probe node
    rel = np.sqrt(np.sum((rec_cp - tru_cp) ** 2) / np.sum(tru_cp ** 2))
    ax.text(0.98, 0.05, rf'rel-RMS $= {rel*100:.1f}\%$',
            transform=ax.transAxes,
            va='bottom', ha='right', fontsize=6.5, color='0.35')

# x-label only on bottom panel
for ax in axes[:-1]:
    ax.set_xticklabels([])
axes[-1].set_xlabel(r'Frequency [Hz]', fontsize=8)

fig.canvas.draw()

# Reposition resonance annotation after layout is settled (axes limits may shift).
# Only the resonance text (data coords); the rel-RMS texts use axes coords — leave them.
if res_txt is not None:
    _, yhi0 = axes[0].get_ylim()
    res_txt.set_y(yhi0)

save_fig(fig, 'fig_recovery_spectra')
plt.close(fig)
print('Done.')
