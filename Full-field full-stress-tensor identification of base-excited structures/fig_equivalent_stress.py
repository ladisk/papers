import sys, matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
sys.path.insert(0, r'C:\Users\jasas\Work\Clanki\Clanek_2\Writing_article\article_bundle\scripts')
from plot_style import apply_style, MM_TO_INCH
apply_style()

FIG = r'C:/Users/jasas/Work/Clanki/Clanek_2/thermoelastic-stress-expansion/synthetic_validation/figures'
def save_fig(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf')
    fig.savefig(f'{FIG}/{name}.png')
    print('saved', name)

def to_grid(v, x, y):
    xs = np.unique(np.round(x, 3)); ys = np.unique(np.round(y, 3))
    G = np.full((len(ys), len(xs)), np.nan)
    ix = np.clip(np.searchsorted(xs, np.round(x, 3)), 0, len(xs)-1)
    iy = np.clip(np.searchsorted(ys, np.round(y, 3)), 0, len(ys)-1)
    G[iy, ix] = v
    return xs, ys, G

d = np.load(r'C:/Users/jasas/Work/Clanki/Clanek_2/thermoelastic-stress-expansion/synthetic_validation/figures_data/straggler_data.npz', allow_pickle=True)
x, y = d['x'], d['y']
freqs = d['freqs']
vm_field_tru = d['vm_field_tru'] / 1e12   # -> MPa^2/Hz
vm_field_rec = d['vm_field_rec'] / 1e12
vm_spec_tru = d['vm_spec_tru']
vm_spec_rec = d['vm_spec_rec']
vm_field_err = float(d['vm_field_err'])
f1 = freqs[int(d['cs_fk1'])]   # ~51.4 Hz
f2 = freqs[int(d['cs_fk2'])]   # ~187.6 Hz

_, _, Gt = to_grid(vm_field_tru, x, y)
_, _, Gr = to_grid(vm_field_rec, x, y)
vmax = float(np.nanmax([np.nanmax(Gt), np.nanmax(Gr)]))
vmin = 0.0

fig = plt.figure(figsize=(170 * MM_TO_INCH, 150 * MM_TO_INCH))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.82],
              hspace=0.48, wspace=0.22,
              left=0.085, right=0.885, top=0.88, bottom=0.085)

# ── (a) von Mises equivalent-stress PSD field ────────────────────────
axT = fig.add_subplot(gs[0, 0])
axR = fig.add_subplot(gs[0, 1])
imk = dict(origin='lower', extent=[0, 150, 0, 150], aspect='equal',
           cmap='viridis', vmin=vmin, vmax=vmax)
axT.imshow(Gt, **imk)
im = axR.imshow(Gr, **imk)
axT.set_title('Truth', fontsize=9)
axR.set_title('Recovered', fontsize=9)
for ax in (axT, axR):
    ax.set_xlabel(r'$x$ [mm]')
    ax.set_xticks([0, 50, 100, 150])
    ax.set_yticks([0, 50, 100, 150])
axT.set_ylabel(r'$y$ [mm]')
axR.set_yticklabels([])

# shared colorbar
pR = axR.get_position()
cax = fig.add_axes([pR.x1 + 0.018, pR.y0, 0.020, pR.height])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r'$\sigma_{\mathrm{vM}}$ PSD [MPa$^2$/Hz]')

# rel-RMS annotation
axR.text(0.96, 0.05, r'rel-RMS $\approx %.1f\%%$' % (vm_field_err * 100),
         transform=axR.transAxes, ha='right', va='bottom', fontsize=8,
         color='white',
         bbox=dict(boxstyle='round,pad=0.28', fc='black', ec='none', alpha=0.5))

# panel label + super-title for (a)
axT.text(-0.28, 1.16, r'\textbf{(a)}', transform=axT.transAxes,
         ha='left', va='bottom', fontsize=11)
fig.text((axT.get_position().x0 + axR.get_position().x1) / 2, 0.945,
         r'von Mises equivalent-stress PSD field at the first resonance',
         ha='center', va='bottom', fontsize=9)

# ── (b) von Mises equivalent-stress PSD spectrum at critical node ────
axS = fig.add_subplot(gs[1, :])
okabe = ['#0072B2', '#D55E00', '#009E73', '#000000']
mask = freqs <= 250
axS.semilogy(freqs, vm_spec_tru, color=okabe[0], lw=3.4, alpha=0.35,
             label='Truth', solid_capstyle='round')
axS.semilogy(freqs, vm_spec_rec, color=okabe[3], lw=0.9, label='Recovered')
axS.set_xlim(0, 250)
lo = min(vm_spec_tru[mask].min(), vm_spec_rec[mask].min())
hi = max(vm_spec_tru[mask].max(), vm_spec_rec[mask].max())
axS.set_ylim(lo * 0.5, hi * 4)
axS.set_xlabel(r'Frequency [Hz]')
axS.set_ylabel(r'$\sigma_{\mathrm{vM}}$ PSD [Pa$^2$/Hz]')

for fr, lab in [(f1, r'$f_1$'), (f2, r'$f_2$')]:
    axS.axvline(fr, color='0.55', ls=':', lw=0.7, zorder=0)
    axS.text(fr + 3, hi * 2.2, lab, color='0.35', fontsize=8, ha='left', va='top')

axS.legend(loc='upper right', handlelength=1.8)
axS.text(-0.075, 1.02, r'\textbf{(b)}', transform=axS.transAxes,
         ha='left', va='bottom', fontsize=11)
axS.set_title('von Mises equivalent-stress PSD at critical node', fontsize=9)

save_fig(fig, 'fig_equivalent_stress')
