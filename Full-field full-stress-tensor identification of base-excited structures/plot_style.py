"""
Shared matplotlib style for the article figures.
LADISK / MSSP publication quality.

Usage:
    from plot_style import apply_style, MM_TO_INCH
    apply_style()

The figures in the article are typeset with LaTeX (``text.usetex``). If no LaTeX
installation is found, the style falls back to matplotlib's mathtext so that the
figure scripts still run; the result is visually close but not identical to the
published figures. Set the environment variable ``PLOT_STYLE_USETEX`` to ``1`` or
``0`` to force either behaviour.
"""
import os
import shutil

import matplotlib
import matplotlib.pyplot as plt

# ── Component labels ───────────────────────────────────────────────────
COMP_KEYS   = ['SX', 'SY', 'SXY', 'SXplusSY']
COMP_LABELS = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\tau_{xy}$', r'$\sigma_{xx}+\sigma_{yy}$']
COMP_TEX    = dict(zip(COMP_KEYS, COMP_LABELS))

# ── Colormaps ──────────────────────────────────────────────────────────
CMAP_SIGNED   = 'RdBu_r'     # diverging — mode shapes, transmissibility
CMAP_SEQ      = 'viridis'    # sequential, perceptually uniform — PSD magnitudes
CMAP_SEQ_ALT  = 'viridis'    # same

# ── Figure sizes (Elsevier two-column: 170 mm full, 83 mm single) ─────
MM_TO_INCH    = 1 / 25.4
FIG_FULL      = (170 * MM_TO_INCH, None)   # full width, height set per figure
FIG_SINGLE    = (83  * MM_TO_INCH, None)   # single column
FIG_1p5       = (120 * MM_TO_INCH, None)   # 1.5 column


def _usetex_available():
    """Whether the figures should be typeset with a real LaTeX installation."""
    forced = os.environ.get('PLOT_STYLE_USETEX')
    if forced is not None:
        return forced.strip() not in ('', '0', 'false', 'False')
    return shutil.which('latex') is not None


# ── Style function ─────────────────────────────────────────────────────
def apply_style():
    """Apply publication-quality matplotlib settings."""
    matplotlib.rcParams['text.usetex'] = _usetex_available()
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = 9
    matplotlib.rcParams['axes.labelsize'] = 9
    matplotlib.rcParams['xtick.labelsize'] = 8
    matplotlib.rcParams['ytick.labelsize'] = 8
    matplotlib.rcParams['legend.fontsize'] = 8
    matplotlib.rcParams['axes.titlesize'] = 9
    matplotlib.rcParams['figure.dpi'] = 300
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['savefig.pad_inches'] = 0.02
    matplotlib.rcParams['lines.linewidth'] = 0.8
    matplotlib.rcParams['axes.linewidth'] = 0.5
    matplotlib.rcParams['xtick.major.width'] = 0.5
    matplotlib.rcParams['ytick.major.width'] = 0.5
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['legend.frameon'] = False


# ── Helper: reshape to 2D grid ────────────────────────────────────────
def to_grid(values, nx=34, ny=34):
    """Reshape (1156,) node vector to (ny, nx) spatial grid."""
    return values.reshape(ny, nx)
