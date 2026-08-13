"""
Fig (R2.3) — modal-truncation convergence. Oracle truncation (the truth model used
as its own prior) isolates the pure truncation error: reconstruct the stress response
with n retained modes and measure the rel-RMS error vs the full-mode truth, over the
full response band and over the 45-250 Hz analysis band. The error is flat from two
modes on (the next structural mode is at 554 Hz).
"""
from pathlib import Path as _Path

# Repository root (this file's directory) - all paths below are relative to it.
_REPO = _Path(__file__).resolve().parent
(_REPO / 'synthetic_validation' / 'figures').mkdir(parents=True, exist_ok=True)

import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, str(_REPO))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from plot_style import apply_style, MM_TO_INCH
apply_style()
from synthetic_validation.harness import run_case_from_modal

BASE = str(_REPO / 'synthetic_validation' / 'analysis' / 'broad_analysis')
FIG  = (_REPO / 'synthetic_validation' / 'figures')
COMPS = ['SX', 'SY', 'SXY']
KW = dict(saa_level=1.0, fs=2000.0, n_frames=10000, seed=0, noise_sigma_T=0.0)

truth = {k: v for k, v in np.load(f'{BASE}/modal_truth.npz', allow_pickle=True).items()}

def relrms(rec, tru, m):
    r, t = rec[m], tru[m]
    den = np.sqrt(np.sum(np.abs(t) ** 2))
    return float(np.sqrt(np.sum(np.abs(r - t) ** 2)) / den) if den > 0 else 0.0

# reference: full-mode oracle response
ref = run_case_from_modal(truth, truth, n_modes=None, **KW)
fr = ref['freqs']
full = np.ones_like(fr, bool)
band = (fr >= 45) & (fr <= 250)

Ns, full_err, band_err = [], [], []
for N in range(1, 6):
    try:
        res = run_case_from_modal(truth, truth, n_modes=N, **KW)
    except Exception as e:
        print(f'  n={N}: stop ({str(e)[:50]})'); break
    fe = 100 * np.mean([relrms(res['recovered'][c], res['truth'][c], full) for c in COMPS])
    be = 100 * np.mean([relrms(res['recovered'][c], res['truth'][c], band) for c in COMPS])
    Ns.append(N); full_err.append(fe); band_err.append(be)
    print(f'  n_modes={N}:  full-band {fe:6.2f}%   45-250 {be:6.3f}%')

fig, ax = plt.subplots(figsize=(112 * MM_TO_INCH, 72 * MM_TO_INCH))
ax.plot(Ns, band_err, '-o', color='#0072B2', lw=1.4, ms=5, zorder=3)
ax.set_ylim(0, max(band_err) * 1.12)
ax.axvline(2, color='0.6', ls=':', lw=0.7, zorder=0)
ax.text(2.08, ax.get_ylim()[1] * 0.96, ' 2 modes retained', color='0.5', fontsize=6.5, va='top', ha='left')
ax.annotate(r'$%.2f\%%$' % band_err[1], xy=(2, band_err[1]),
            xytext=(2.55, band_err[1] + max(band_err) * 0.13), fontsize=7.5, color='#0072B2',
            ha='left', va='bottom', arrowprops=dict(arrowstyle='->', lw=0.6, color='#0072B2'))
ax.set_xlabel('number of retained modes', fontsize=9)
ax.set_ylabel(r'stress rel-RMS error, 45--250~Hz [\%]', fontsize=9)
ax.set_xticks(Ns)
fig.tight_layout()
fig.savefig(FIG / 'fig_mode_convergence.pdf')
fig.savefig(FIG / 'fig_mode_convergence.png')
print('saved fig_mode_convergence')
plt.close(fig)
