"""
Fig — prior-insensitivity of the recovery.

(a) SEMM-corrected recovery error vs the parent FE Young's modulus, for four
    ratio-neutral parents (E 63->38 GPa, also thinner/lighter): every component
    is FLAT -> the method is insensitive to material/geometry scale errors.
(b) The one error that DOES propagate: the prior's Poisson ratio, swept over a range.
    sigma_xx is lowest at the true nu=0.33 (11.5%, its block-only floor) and rises as nu
    deviates (23% at nu=0.25), because nu sets the sigma_xx:sigma_yy split that an
    invariant-only measurement cannot constrain.

Metrics are in the AMPLITUDE domain (band-RMS stress field, 45-250 Hz) -- the domain
the method operates in and the one the response-letter tables report. The PSD is
quadratic and would roughly double every error shown here.

Data: synthetic_validation/analysis/broad_analysis/robustness_sweep_amp.json
"""
import sys, matplotlib
matplotlib.use('Agg')
sys.path.insert(0, r'C:\Users\jasas\Work\Clanki\Clanek_2\Writing_article\article_bundle\scripts')
from plot_style import apply_style, MM_TO_INCH
apply_style()

import json
import numpy as np
import matplotlib.pyplot as plt

BASE = r'C:/Users/jasas/Work/Clanki/Clanek_2/thermoelastic-stress-expansion'
FIG = f'{BASE}/synthetic_validation/figures'
R = json.load(open(f'{BASE}/synthetic_validation/analysis/broad_analysis/robustness_sweep_amp.json'))

def save_fig(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf'); fig.savefig(f'{FIG}/{name}.png'); print('saved', name)

COMPS = ['SX', 'SY', 'SXY', 'VM']
LBL   = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\tau_{xy}$', r'von Mises']
OI    = ['#0072B2', '#D55E00', '#009E73', '#000000']

neutral = [r for r in R if r['nu'] == 0.33 and r['name'].startswith('P')]   # E-sweep only (P0..P4); excludes clamp C1/C2
E = np.array([r['E'] / 1e9 for r in neutral])
base_row = [r for r in neutral if abs(r['E'] - 63e9) < 1e6][0]   # paper's prior

fig, (axa, axb) = plt.subplots(1, 2, figsize=(170 * MM_TO_INCH, 72 * MM_TO_INCH),
                               gridspec_kw={'width_ratios': [1.25, 1.0]})
fig.subplots_adjust(left=0.095, right=0.985, bottom=0.165, top=0.88, wspace=0.30)

# ── (a) flat vs E ─────────────────────────────────────────────────────────
for c, lb, col in zip(COMPS, LBL, OI):
    semm = np.array([r['amp'][c] * 100 for r in neutral])
    axa.plot(E, semm, '-o', color=col, lw=1.1, ms=3.5, label=lb, zorder=3)
axa.axvline(69, color='0.6', lw=0.6, ls=':', zorder=1)
axa.text(69, 15.6, ' true $E$', color='0.5', fontsize=6.5, va='top', ha='left')
axa.set_xlabel(r"parent Young's modulus $E$ [GPa]")
axa.set_ylabel(r'rel-RMS error, stress amplitude [\%]')
axa.set_ylim(0, 16)
axa.invert_xaxis()   # worse (lower E) to the right
axa.set_title('(a)', loc='left', fontsize=8.5)
axa.legend(loc='upper right', ncol=2, handlelength=1.4, fontsize=7.5, borderaxespad=0.6)

# ── (b) nu breaks the split — sweep as a line ────────────────────────────
NU = sorted(json.load(open(f'{BASE}/synthetic_validation/analysis/broad_analysis/robustness_sweep_nu.json')),
            key=lambda r: r['nu'])
nu_x = np.array([r['nu'] for r in NU])
for c, lb, col in zip(COMPS, LBL, OI):
    axb.plot(nu_x, [r['amp'][c] * 100 for r in NU], '-o', color=col, lw=1.1, ms=3.5, label=lb, zorder=3)
axb.axvline(0.33, color='0.6', lw=0.6, ls=':', zorder=1)
axb.text(0.33, 1.1, 'true $\\nu$', color='0.5', fontsize=6.5, va='bottom', ha='center')
axb.set_xlabel(r"parent Poisson's ratio $\nu$")
axb.set_ylabel(r'rel-RMS error, stress amplitude [\%]')
axb.set_ylim(0, 25)
axb.set_title('(b)', loc='left', fontsize=8.5)
axb.legend(loc='upper center', ncol=2, handlelength=1.4, fontsize=7.2, borderaxespad=0.5)

save_fig(fig, 'fig_prior_robustness')
plt.close(fig)
