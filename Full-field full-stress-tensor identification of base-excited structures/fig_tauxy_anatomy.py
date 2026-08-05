import sys, matplotlib
matplotlib.use('Agg')
sys.path.insert(0, r'C:\Users\jasas\Work\Clanki\Clanek_2\Writing_article\article_bundle\scripts')
from plot_style import apply_style, MM_TO_INCH
apply_style()

import numpy as np, json
import matplotlib.pyplot as plt

FIG = r'C:/Users/jasas/Work/Clanki/Clanek_2/thermoelastic-stress-expansion/synthetic_validation/figures'

def save_fig(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf')
    fig.savefig(f'{FIG}/{name}.png')
    print('saved', name)

d = np.load(r'C:/Users/jasas/Work/Clanki/Clanek_2/thermoelastic-stress-expansion/synthetic_validation/figures_data/fig_data_broad.npz', allow_pickle=True)
S = json.load(open(r'C:/Users/jasas/Work/Clanki/Clanek_2/thermoelastic-stress-expansion/synthetic_validation/figures_data/fig_data_broad.json'))

OI = ['#0072B2', '#D55E00', '#009E73', '#000000']

# ── data ────────────────────────────────────────────────────────────────
tru = d['tru_SXY_res'].astype(float)
rec = d['rec_SXY_res'].astype(float)
k = float(S['k_sxy_res'])

thr = 1e-3 * tru.max()
m = tru > thr
xt, yr = tru[m], rec[m]

# grouped-bar data — RESONANCE-BIN (raw-PSD) decomposition, consistent with panel (a)
# and the headline rel-RMS numbers (SX 8.1 / SY 4.9 / SXY 21.5 / inv 4.9 %).
comps = ['SX', 'SY', 'SXY', 'SX+SY']
cat_lbl = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\tau_{xy}$', r'invariant']

def _bias_scatter_res(c):
    t = d[f'tru_{c}_res'].astype(float); r = d[f'rec_{c}_res'].astype(float)
    kk = float(np.sum(t * r) / np.sum(t * t))            # through-origin slope
    rr = float(np.sqrt(np.sum((r - t) ** 2) / np.sum(t * t)))  # rel-RMS
    b = abs(kk - 1.0)
    return b, float(np.sqrt(max(rr ** 2 - b ** 2, 0.0)))

bias = np.array([_bias_scatter_res(c)[0] for c in comps])
scat = np.array([_bias_scatter_res(c)[1] for c in comps])

# ── figure ──────────────────────────────────────────────────────────────
fig, (axa, axb) = plt.subplots(1, 2, figsize=(170 * MM_TO_INCH, 78 * MM_TO_INCH))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.145, top=0.90, wspace=0.30)

# ---- (a) log-log scatter ----
lo = min(xt.min(), yr.min())
hi = max(xt.max(), yr.max())
lo *= 0.6
hi *= 1.6

axa.scatter(xt, yr, s=5, c=OI[0], alpha=0.45, edgecolors='none', rasterized=True,
            zorder=2, label='nodes')
line = np.array([lo, hi])
axa.plot(line, line, ls='--', c='k', lw=0.8, zorder=3, label='1:1')
axa.plot(line, k * line, ls='-', c=OI[2], lw=1.1, zorder=4,
         label=r'fit $y=k\,x$')

axa.set_xscale('log')
axa.set_yscale('log')
axa.set_xlim(lo, hi)
axa.set_ylim(lo, hi)
axa.set_aspect('equal')
axa.set_xlabel(r'True $\tau_{xy}$ PSD [Pa$^2$/Hz]')
axa.set_ylabel(r'Recovered $\tau_{xy}$ PSD [Pa$^2$/Hz]')
axa.legend(loc='upper left', handlelength=1.6, borderaxespad=0.4)

axa.text(0.97, 0.09,
         'slope $k=%.2f$' % k + '\n' + r'peak-ratio 0.86' + '\n' + '(systematic under-estimate)',
         transform=axa.transAxes, ha='right', va='bottom', fontsize=7.5,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6', lw=0.5))
axa.set_title('(a)', loc='left', fontweight='bold')

# ---- (b) grouped bar ----
xpos = np.arange(len(comps))
w = 0.38
axb.bar(xpos - w / 2, bias, w, color=OI[1], label=r'bias $|k-1|$')
axb.bar(xpos + w / 2, scat, w, color=OI[0], label='scatter')

axb.set_xticks(xpos)
axb.set_xticklabels(cat_lbl)
axb.set_ylabel('error contribution (rel-RMS)')
axb.set_ylim(0, max(bias.max(), scat.max()) * 1.28)
axb.legend(loc='upper right', handlelength=1.3, borderaxespad=0.4)
axb.set_title('(b)', loc='left', fontweight='bold')

# highlight the tall tau_xy bias bar
i_sxy = comps.index('SXY')
axb.annotate(r'only $\tau_{xy}$ biased',
             xy=(i_sxy - w / 2, bias[i_sxy]),
             xytext=(i_sxy - 1.15, bias[i_sxy] + 0.030),
             fontsize=7.5, ha='left', va='bottom',
             arrowprops=dict(arrowstyle='->', lw=0.6, color='0.3'))

axb.text(0.5, -0.235,
         r'fatigue-relevant $\tau_{xy}$ band-RMS map: 11.3\%, $k=0.92$, MAC 0.99',
         transform=axb.transAxes, ha='center', va='top', fontsize=7.2)

save_fig(fig, 'fig_tauxy_anatomy')
plt.close(fig)
