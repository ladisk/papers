from pathlib import Path as _Path

# Repository root (this file's directory) - all paths below are relative to it.
_REPO = _Path(__file__).resolve().parent
(_REPO / 'synthetic_validation' / 'figures').mkdir(parents=True, exist_ok=True)

import sys, json
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(_REPO))
from plot_style import apply_style, MM_TO_INCH
apply_style()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# usetex guard
USETEX = True
try:
    fig_t = plt.figure(); fig_t.text(0.5, 0.5, r'$\tau_{xy}$'); fig_t.canvas.draw(); plt.close(fig_t)
except Exception as e:
    matplotlib.rcParams['text.usetex'] = False
    USETEX = False
    print('usetex FALLBACK:', e)

FIG = str(_REPO / 'synthetic_validation' / 'figures')
def save_fig(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf'); fig.savefig(f'{FIG}/{name}.png'); print('saved', name)

S = json.load(open(str(_REPO / 'synthetic_validation' / 'figures_data' / 'fig_data_broad.json')))

# ---- colours ----
GREEN = '#009E73'   # down-steps (error reduction)
GREY  = '#9a9a9a'   # level step (ratio-neutral)
BLUE  = '#0072B2'   # anchor totals
TXT   = '#000000'

# =====================================================================
fig = plt.figure(figsize=(170 * MM_TO_INCH, 82 * MM_TO_INCH))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.40,
                      left=0.078, right=0.985, top=0.80, bottom=0.185)

# =====================================================================
# (a) WATERFALL / BRIDGE  for tau_xy rel-RMS (%)
# =====================================================================
axa = fig.add_subplot(gs[0, 0])

# stage geometry: (label, kind, bottom, top, delta-string)
#   total: full bar 0..top ; down: bar low..high (green) ; level: sliver
stages = [
    ('Parent FE\n(no block)', 'total', 0.0,  61.3, None),
    (r'E:' + '\n' + r'63$\to$69', 'level', 61.3, 61.3, r'$+$0.0'),
    ('SEMM\ncorr.',           'down',  21.5, 61.3, r'$-$39.7'),
    ('Method\n(SEMM)',        'total', 0.0,  21.5, None),
    ('residual',              'down',  2.0,  21.5, r'$-$19.5'),
    ('Oracle\n(perfect)',     'total', 0.0,  2.0,  None),
]
bw = 0.62
xs = np.arange(len(stages))

# value / MAC label heights per anchor: {i: (value_y, mac_y, mac)}
anchor = {0: (62.6, 70.5, 0.744), 3: (22.8, 29.5, 0.973), 5: (3.6, 10.0, 1.00)}

for i, (lab, kind, bot, top, delta) in enumerate(stages):
    if kind == 'total':
        axa.bar(i, top - bot, bottom=bot, width=bw, color=BLUE,
                edgecolor='black', linewidth=0.5, zorder=3)
        vy, my, m = anchor[i]
        axa.text(i, vy, f'{top:.1f}', ha='center', va='bottom',
                 fontsize=8, fontweight='bold', color=BLUE, zorder=5)
        axa.text(i, my, f'MAC {m:.3f}' if m < 1 else 'MAC 1.00',
                 ha='center', va='bottom', fontsize=7, color='#333333',
                 fontstyle='italic', zorder=6)
    elif kind == 'down':
        axa.bar(i, top - bot, bottom=bot, width=bw, color=GREEN,
                edgecolor='black', linewidth=0.5, zorder=3)
        axa.text(i, (top + bot) / 2, delta, ha='center', va='center',
                 fontsize=8.5, fontweight='bold', color='white', zorder=5)
    else:  # level / ratio-neutral: thin grey sliver + label in empty column
        axa.bar(i, 1.4, bottom=bot - 0.7, width=bw, color=GREY,
                edgecolor='black', linewidth=0.5, zorder=3)
        axa.annotate(delta + '\n(ratio-\nneutral)', xy=(i, bot - 0.9),
                     xytext=(i, 40.0), ha='center', va='top', fontsize=7,
                     color='#555555', zorder=6,
                     arrowprops=dict(arrowstyle='-', lw=0.5, color='#888888'))

# connector lines (classic waterfall)
conn = [61.3, 61.3, 21.5, 21.5, 2.0]
for i in range(len(stages) - 1):
    axa.plot([i + bw / 2, i + 1 - bw / 2], [conn[i], conn[i]],
             ls=(0, (3, 2)), lw=0.6, color='#666666', zorder=2)

axa.set_xticks(xs)
axa.set_xticklabels([s[0] for s in stages], fontsize=7)
axa.set_ylim(0, 80)
axa.set_ylabel(r'$\tau_{xy}$ relative RMS error (\%)' if USETEX
               else r'$\tau_{xy}$ relative RMS error (%)')
axa.set_axisbelow(True)
axa.grid(axis='y', lw=0.4, color='#dddddd', zorder=0)
axa.spines['top'].set_visible(False)
axa.spines['right'].set_visible(False)
axa.set_title('(a) Where the $\\tau_{xy}$ error comes from', fontsize=9, pad=6)

# =====================================================================
# (b) TABLE-HEATMAP of rel_rms for the 5 priors x 4 components
# =====================================================================
axb = fig.add_subplot(gs[0, 1])

priors = ['oracle', 'method', 'parent (E63,no mass)',
          'parent (E69,no mass)', 'parent + sym mass']
row_lab = ['Oracle', 'Method\n(SEMM)', 'Parent $E{=}63$\n(no block)',
           'Parent $E{=}69$\n(no block)', 'Parent $+$\nsym. mass']
comps = ['SX', 'SY', 'SXY', 'SX+SY']
col_lab = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\tau_{xy}$',
           r'$\sigma_{xx}\!+\!\sigma_{yy}$']

M = np.array([[S['decomp'][p][c]['rel_rms'] * 100 for c in comps] for p in priors])

im = axb.imshow(M, cmap='Reds', vmin=0, vmax=75, aspect='auto')

for r in range(M.shape[0]):
    for c in range(M.shape[1]):
        v = M[r, c]
        col = 'white' if v > 42 else '#111111'
        axb.text(c, r, f'{v:.0f}\\%' if USETEX else f'{v:.0f}%',
                 ha='center', va='center', fontsize=8, color=col,
                 fontweight='bold' if (r >= 2 and c == 2) else 'normal')

axb.set_xticks(np.arange(4)); axb.set_xticklabels(col_lab, fontsize=8)
axb.set_yticks(np.arange(5)); axb.set_yticklabels(row_lab, fontsize=7.3)
axb.set_xticks(np.arange(-.5, 4, 1), minor=True)
axb.set_yticks(np.arange(-.5, 5, 1), minor=True)
axb.grid(which='minor', color='white', lw=1.2)
axb.tick_params(which='both', length=0)
axb.set_title(r'(b) rel-RMS by prior and stress component', fontsize=9, pad=6)

# bracket: the two 'parent ... no mass' rows are IDENTICAL (E irrelevant)
bx = 3.58
axb.plot([bx, bx], [1.55, 3.45], color='#333333', lw=1.0, clip_on=False)
axb.plot([3.52, bx], [1.55, 1.55], color='#333333', lw=1.0, clip_on=False)
axb.plot([3.52, bx], [3.45, 3.45], color='#333333', lw=1.0, clip_on=False)
axb.text(3.72, 2.5, r'identical rows ($E$ irrelevant)', rotation=90,
         ha='left', va='center', fontsize=6.8, color='#333333',
         clip_on=False)

# highlight the sym-mass SXY cell (WORSE than plain no-mass)
axb.add_patch(Rectangle((2 - 0.5, 4 - 0.5), 1, 1, fill=False,
                        edgecolor=GREEN, lw=1.8, zorder=6))
axb.annotate('sym. mass makes\n$\\tau_{xy}$ WORSE\n(73 vs.\\ 61\\,\\%)'
             if USETEX else 'sym. mass makes\n$\\tau_{xy}$ WORSE\n(73 vs. 61%)',
             xy=(2, 4.52), xytext=(0.45, 5.75), ha='center', va='top',
             fontsize=6.8, color=GREEN, annotation_clip=False,
             arrowprops=dict(arrowstyle='->', lw=0.8, color=GREEN))

# colorbar (dedicated axes at far right so the bracket has clean room)
cb = fig.colorbar(im, ax=axb, fraction=0.045, pad=0.16)
cb.set_label(r'rel-RMS (\%)' if USETEX else 'rel-RMS (%)', fontsize=8)
cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.5)

# =====================================================================
sup = (r'The $\tau_{xy}$ error is the missing block; '
       r'SEMM corrects $\sim$65\% of it') if USETEX else \
      r'The $\tau_{xy}$ error is the missing block; SEMM corrects ~65% of it'
fig.suptitle(sup, fontsize=10, y=0.965)

save_fig(fig, 'fig_fe_waterfall')
plt.close(fig)
print('USETEX =', USETEX)
