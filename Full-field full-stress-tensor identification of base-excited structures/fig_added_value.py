"""
Fig (R2.7) — added value of the camera-driven SEMM correction, shown on the shear.
Four rungs of the shear (tau_xy) rel-RMS error in the stress-amplitude domain:
  FE-only (block-free parent, no camera)      95%
  measured invariant + uncorrected FE basis   34.3%
  SEMM (the proposed method)                   11.3%
  exact prior (floor)                          0.1%
"""
from pathlib import Path as _Path

# Repository root (this file's directory) - all paths below are relative to it.
_REPO = _Path(__file__).resolve().parent
(_REPO / 'synthetic_validation' / 'figures').mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(_REPO))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from plot_style import apply_style, MM_TO_INCH
apply_style()

FIG = (_REPO / 'synthetic_validation' / 'figures')

labels = ['FE-only\n(no camera)', 'uncorrected\nFE basis', 'SEMM\n(this work)', 'exact prior\n(floor)']
vals   = [95.0, 34.3, 11.3, 0.1]
colors = ['0.72', '#9ecae1', '#0072B2', '#009E73']   # SEMM highlighted in strong blue

fig, ax = plt.subplots(figsize=(118 * MM_TO_INCH, 70 * MM_TO_INCH))
ax.grid(axis='y', color='0.90', lw=0.6, zorder=0)
ax.bar(range(4), vals, width=0.62, color=colors, edgecolor='0.35', linewidth=0.6, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + 2.0, r'$%s\%%$' % ('%g' % v), ha='center', va='bottom', fontsize=8.5)
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r'shear $\tau_{xy}$ rel-RMS error [\%]', fontsize=9)
ax.set_ylim(0, 104)
ax.set_title(r'Added value of the camera-driven SEMM correction', fontsize=9.5, loc='left')
fig.tight_layout()
fig.savefig(FIG / 'fig_added_value.pdf')
fig.savefig(FIG / 'fig_added_value.png')
print('saved fig_added_value')
plt.close(fig)
