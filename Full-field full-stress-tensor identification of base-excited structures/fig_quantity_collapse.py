from pathlib import Path as _Path

# Repository root (this file's directory) - all paths below are relative to it.
_REPO = _Path(__file__).resolve().parent
(_REPO / 'synthetic_validation' / 'figures').mkdir(parents=True, exist_ok=True)

import sys, matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(_REPO))
from plot_style import apply_style, MM_TO_INCH
apply_style()

import json
import numpy as np
import matplotlib.pyplot as plt

FIG = str(_REPO / 'synthetic_validation' / 'figures')
def save_fig(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf')
    fig.savefig(f'{FIG}/{name}.png')
    print('saved', name)

S = json.load(open(str(_REPO / 'synthetic_validation' / 'figures_data' / 'fig_data_broad.json')))
C = S['collapse']

BLUE = '#0072B2'
VERM = '#D55E00'

# Blue collapse cascade, top-to-bottom
blue_labels = [
    r'$\tau_{xy}$ raw PSD',
    r'$\tau_{xy}$ band-RMS map',
    r'von Mises PSD field',
    r'von Mises top-10\% hotspots',
    r'von Mises damage-weighted ($b=5$)',
]
blue_keys = ['sxy_raw', 'sxy_bandmap', 'vm_field', 'vm_top10', 'vm_dmg_b5']
blue_vals = [C[k] * 100 for k in blue_keys]

verm_label = r'$\tau_{xy}$ FIELD damage-weighted ($b=5$)'
verm_val = C['sxy_dmg_b5'] * 100

# y positions: blue group at top, gap, vermilion bar apart at bottom
n = len(blue_labels)
y_blue = np.arange(n)[::-1] + 1.6   # 5.6 .. 1.6 (top to bottom)
y_verm = 0.0

fig, ax = plt.subplots(figsize=(120 * MM_TO_INCH, 66 * MM_TO_INCH))

bh = 0.62
ax.barh(y_blue, blue_vals, height=bh, color=BLUE, zorder=3)
ax.barh(y_verm, verm_val, height=bh, color=VERM, hatch='//',
        edgecolor='white', linewidth=0.0, zorder=3)

# Value labels at bar ends
for y, v in zip(y_blue, blue_vals):
    ax.text(v + 0.4, y, f'{v:.1f}', va='center', ha='left',
            fontsize=8, color=BLUE, zorder=4)
ax.text(verm_val + 0.4, y_verm, f'{verm_val:.1f}', va='center', ha='left',
        fontsize=8, color=VERM, zorder=4)

# y tick labels
yt = list(y_blue) + [y_verm]
ytl = blue_labels + [verm_label]
ax.set_yticks(yt)
ax.set_yticklabels(ytl)
# colour the vermilion tick label to match
ax.get_yticklabels()[-1].set_color(VERM)

# Divider between the blue collapse group and the honest caveat bar
ax.axhline(0.8, color='0.6', lw=0.5, ls=(0, (3, 2)), zorder=1)

ax.set_xlim(0, 27)
ax.set_ylim(-0.6, 6.5)
ax.set_xlabel(r'relative RMS error [\%]')
ax.set_title(r'From raw shear to the fatigue-driving quantity')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.xaxis.grid(True, color='0.85', lw=0.4, zorder=0)
ax.set_axisbelow(True)

# Mandatory honest caveat annotation
caveat = (r'the $\tau_{xy}$ field itself stays elevated (its damage is spatially'
          '\n'
          r'spread) -- the collapse is for the STRUCTURE' + "'" + r's von Mises'
          '\n'
          r'estimate, NOT for $\tau_{xy}$ itself.')
ax.annotate(caveat, xy=(verm_val * 0.5, y_verm + bh * 0.4), xytext=(11.5, 2.7),
            fontsize=7.0, color=VERM, va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.35', fc='#FDF0E8',
                      ec=VERM, lw=0.5),
            arrowprops=dict(arrowstyle='-|>', color=VERM, lw=0.7,
                            shrinkA=0, shrinkB=3))

fig.tight_layout()
save_fig(fig, 'fig_quantity_collapse')
