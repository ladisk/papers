"""
Fig R2.1 — FE model sensitivity: metric masking contrast.

Panel (a): grouped bar chart — rel. RMS error (%) for sigma_xx and tau_xy,
           for each of 4 FE-error cases.  Poisson-ratio errors give ~40 %
           while E/thickness give ~3 %.
Panel (b): NRMSE (range-normalised) for sigma_xx on the same 4 cases.
           NRMSE ~0.19 % for the SAME Poisson-ratio cases — masks the
           true 40 % error by a factor ~200×.
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
import matplotlib.ticker as ticker
from pathlib import Path
from plot_style import apply_style, MM_TO_INCH

apply_style()

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = (_REPO / 'synthetic_validation' / 'figures_data')
FIG_DIR  = (_REPO / 'synthetic_validation' / 'figures')

def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{name}.pdf')
    fig.savefig(FIG_DIR / f'{name}.png')
    print(f'Saved: {name}.pdf, {name}.png')

# ── Load data ─────────────────────────────────────────────────────────────
d = np.load(DATA_DIR / 'study_c_sensitivity.npz', allow_pickle=True)
raw_labels = list(d['labels'])           # 4 case names
nrmse_SX   = d['nrmse_SX']              # (4,) range-normalised RMSE
relrms_SX  = d['relrms_SX']             # (4,) proportional RMS error, fraction
relrms_SXY = d['relrms_SXY']            # (4,) proportional RMS error, fraction

# Convert fractions → %
relrms_SX_pct  = relrms_SX  * 100.
relrms_SXY_pct = relrms_SXY * 100.
nrmse_SX_pct   = nrmse_SX   * 100.

print('relrms_SX  [%]:', relrms_SX_pct.round(1))
print('relrms_SXY [%]:', relrms_SXY_pct.round(1))
print('nrmse_SX   [%]:', nrmse_SX_pct.round(3))

# ── X-axis labels (LaTeX) ─────────────────────────────────────────────────
case_labels = [
    'Nominal\n(E, mass)',
    r'$\nu = 0.25$',
    r'$\nu = 0.40$',
    r'thick $+15\%$',
]

# ── Colours (Okabe–Ito) ───────────────────────────────────────────────────
C_SX  = '#0072B2'   # blue — sigma_xx
C_SXY = '#D55E00'   # vermilion — tau_xy
C_NRM = '#009E73'   # green — NRMSE

# ── Figure layout ─────────────────────────────────────────────────────────
fig_w = 170 * MM_TO_INCH    # Elsevier full width
fig_h = 90  * MM_TO_INCH
fig, (ax_a, ax_b) = plt.subplots(
    1, 2,
    figsize=(fig_w, fig_h),
    gridspec_kw={'width_ratios': [1.6, 1.0], 'wspace': 0.42},
)

n = len(case_labels)
x = np.arange(n)
w = 0.34   # bar width

# ── Panel (a): grouped bars relrms ────────────────────────────────────────
b1 = ax_a.bar(x - w/2, relrms_SX_pct,  w, label=r'$\sigma_{xx}$',
              color=C_SX,  edgecolor='white', linewidth=0.4)
b2 = ax_a.bar(x + w/2, relrms_SXY_pct, w, label=r'$\tau_{xy}$',
              color=C_SXY, edgecolor='white', linewidth=0.4)

ax_a.set_xticks(x)
ax_a.set_xticklabels(case_labels, fontsize=7.5)
ax_a.set_ylabel(r'Rel.\ RMS error [\%]', fontsize=8)
ax_a.set_ylim(0, 52)
ax_a.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax_a.tick_params(axis='y', labelsize=7.5)
ax_a.tick_params(axis='x', length=0)   # no x ticks, labels only
ax_a.legend(fontsize=7.5, loc='upper left',
            bbox_to_anchor=(0.0, 1.0))
ax_a.set_title(r'(a) Proportional error (rel.\ RMS)', fontsize=8, pad=4)

# Shade the two Poisson-ratio cases to emphasise them
for xi in [1, 2]:
    ax_a.axvspan(xi - 0.5, xi + 0.5, color='#cccccc', alpha=0.25, zorder=0)

# Annotate Poisson region (single label — no duplicate)
ax_a.text(1.5, 49.5, r'Poisson-ratio errors', ha='center', fontsize=6.5,
          color='0.35', style='italic')

# ── Panel (b): nrmse bars ─────────────────────────────────────────────────
ax_b.bar(x, nrmse_SX_pct, 0.55,
         color=C_NRM, edgecolor='white', linewidth=0.4,
         label=r'$\sigma_{xx}$ NRMSE')

ax_b.set_xticks(x)
ax_b.set_xticklabels(case_labels, fontsize=7.0, rotation=40, ha='right')
ax_b.set_ylabel(r'NRMSE [\%]', fontsize=8)
ax_b.set_ylim(0, 0.26)
ax_b.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax_b.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax_b.tick_params(axis='y', labelsize=7.5)
ax_b.tick_params(axis='x', length=0)
ax_b.set_title(r'(b) Range-norm.\ NRMSE (masks error)', fontsize=8, pad=4)

# Shade the Poisson cases
for xi in [1, 2]:
    ax_b.axvspan(xi - 0.5, xi + 0.5, color='#cccccc', alpha=0.25, zorder=0)

# Factor: relrms_SX[1] / nrmse_SX[1] ~ 38.5 % / 0.19 % ~ 200
factor = relrms_SX_pct[1] / nrmse_SX_pct[1]
ax_b.annotate(
    '',
    xy=(1, nrmse_SX_pct[1] + 0.005),
    xytext=(1, 0.235),
    arrowprops=dict(arrowstyle='->', color='#D55E00', lw=0.8),
)
ax_b.text(1.08, 0.232,
          fr'$\approx {factor:.0f}\times$ below true error',
          fontsize=6, color='#D55E00', va='top', ha='left')

fig.canvas.draw()
save_fig(fig, 'fig_fe_sensitivity')
plt.close(fig)
print('Done.')
