"""
Fig — recovered cross-spectral stress-PSD structure (R2.9), corner-mass truth.

(a) band-integrated 3x3 stress cross-spectral (covariance) matrix |Sigma_ij| at a
    representative mid-field node, Truth vs Recovered (shared log scale): the full
    matrix, including the off-diagonal cross-terms, is recovered.
(b) band-integrated inter-component coherence gamma^2 for the three pairs, Truth vs
    Recovered: sigma_xx-sigma_yy is proportional (~0.98) while the shear pairs are
    genuinely non-proportional (~0.78-0.82) - multiaxial content the field now has
    (the old single-mode field was strictly rank-1), and the recovery preserves it.

Data: synthetic_validation/figures_data/straggler_data.npz
"""
import sys, matplotlib
matplotlib.use('Agg')
sys.path.insert(0, r'C:\Users\jasas\Work\Clanki\Clanek_2\Writing_article\article_bundle\scripts')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from plot_style import apply_style, MM_TO_INCH
apply_style()

DATA = Path(r'C:\Users\jasas\Work\Clanki\Clanek_2\thermoelastic-stress-expansion\synthetic_validation\figures_data')
FIG  = Path(r'C:\Users\jasas\Work\Clanki\Clanek_2\thermoelastic-stress-expansion\synthetic_validation\figures')
def save_fig(fig, name):
    fig.savefig(FIG / f'{name}.pdf'); fig.savefig(FIG / f'{name}.png'); print('saved', name)

d = np.load(DATA / 'straggler_data.npz', allow_pickle=True)
Shot = np.abs(d['Sig_hot']) / 1e12                       # mean |Sig_ij| over top-10% hotspots [MPa^2/Hz]
xerr = np.asarray(d['xspec_hot_err']) * 100.0            # per-cell rel-RMS error over hotspots [%]
cht, chr_ = d['coh_tru'], d['coh_rec']
LAB = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\tau_{xy}$']

fig = plt.figure(figsize=(170 * MM_TO_INCH, 66 * MM_TO_INCH))
outer = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.2], wspace=0.28)
gsL = outer[0].subgridspec(2, 2, height_ratios=[1, 0.06], hspace=0.55, wspace=0.18)
axT = fig.add_subplot(gsL[0, 0]); axR = fig.add_subplot(gsL[0, 1])
caxL = fig.add_subplot(gsL[1, 0]); caxR = fig.add_subplot(gsL[1, 1])
axB = fig.add_subplot(outer[1])

# (left) cross-spectral structure at the top-10% fatigue hotspots [MPa^2/Hz]
normM = LogNorm(vmin=Shot[Shot > 0].min(), vmax=Shot.max())
imT = axT.imshow(Shot, norm=normM, cmap='viridis')
axT.set_title(r'structure', fontsize=8.5)
for i in range(3):
    for j in range(3):
        v = Shot[i, j]
        axT.text(j, i, (r'%.0f' if v >= 10 else r'%.1f') % v, ha='center', va='center',
                 fontsize=5.8, color='k' if normM(v) > 0.6 else 'w')
# (right) per-cell recovery error [%] over the same hotspots (rel-RMS over nodes)
imR = axR.imshow(xerr, cmap='OrRd', vmin=0, vmax=25)
axR.set_title(r'recovery error', fontsize=8.5)
for i in range(3):
    for j in range(3):
        axR.text(j, i, r'%.0f\%%' % xerr[i, j], ha='center', va='center', fontsize=5.8,
                 color='w' if xerr[i, j] > 16 else 'k')
for ax in (axT, axR):
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LAB, fontsize=7.5); ax.set_yticklabels(LAB, fontsize=7.5)
axR.set_yticklabels([])
fig.colorbar(imT, cax=caxL, orientation='horizontal'); caxL.set_xlabel(r'$|\Sigma_{ij}|$ [MPa$^2$/Hz]', fontsize=6.2, labelpad=1.5)
fig.colorbar(imR, cax=caxR, orientation='horizontal'); caxR.set_xlabel(r'rel-RMS error [\%]', fontsize=6.2, labelpad=1.5)
axT.text(0.0, 1.18, r'(a) band-integrated cross-spectral matrix, top-10\% stressed nodes',
         transform=axT.transAxes, fontsize=8.5, ha='left')

# (b) coherence bars
pairs = [r'$\sigma_{xx}$--$\sigma_{yy}$', r'$\sigma_{xx}$--$\tau_{xy}$', r'$\sigma_{yy}$--$\tau_{xy}$']
ct = [cht[0, 1], cht[0, 2], cht[1, 2]]; cr = [chr_[0, 1], chr_[0, 2], chr_[1, 2]]
xp = np.arange(3); w = 0.38
axB.bar(xp - w/2, ct, w, color='0.6', label='truth')
axB.bar(xp + w/2, cr, w, color='#0072B2', label='recovered')
axB.axhline(1.0, color='0.7', lw=0.5, ls=':')
axB.set_xticks(xp); axB.set_xticklabels(pairs, fontsize=7)
axB.set_ylabel(r'coherence $\gamma^2$', fontsize=8.5); axB.set_ylim(0, 1.18)
axB.legend(fontsize=7, loc='upper right', ncol=1, borderaxespad=0.3)
axB.set_title(r'(b) inter-component coherence', fontsize=8.5, loc='left')

fig.subplots_adjust(left=0.06, right=0.985, bottom=0.155, top=0.86)
save_fig(fig, 'fig_cross_spectra')
plt.close(fig)
