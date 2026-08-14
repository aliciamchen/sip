"""Shared matplotlib style for the Python-generated manuscript figures.

One place for what all the figure scripts in figures/scripts/ (the main results
figures, the schematic panels, and the SI LM-elicitation figures
plot_si_validation.py / plot_alternatives.py) must agree on: the Arial Nova
font (matching the
R analysis figures), the despined white look, and the palettes. apply_style()
takes a context — "si" for print-sized supplement figures, "schematic" for the
large-type vector panels assembled in Illustrator.

The three observed actions use the schematic's seaborn colorblind palette
(desaturated), assigned no share = blue, low-risk share = green, high-risk
share = amber. The effort / desire / intimacy palettes match analysis/utils.R.
(The R elicitation notebooks still color actions with an older green/gold/red
scheme; update those scales if the R figures should match.)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LinearSegmentedColormap,
    hsv_to_rgb,
    rgb_to_hsv,
    to_hex,
)

_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root  # noqa: E402

# Output roots, one per consumer, so a script never hardcodes a path and the
# split stays enforceable: `panels/` holds Illustrator components and never an
# assembled figure, and `si/` holds finished figures that go straight into
# \includegraphics.
_FIGURES = get_project_root() / "figures"
SI_DIR = _FIGURES / "si"
PANELS_RESULTS = _FIGURES / "panels" / "results"
PANELS_LEGENDS = _FIGURES / "panels" / "legends"
PANELS_SCHEMATIC = _FIGURES / "panels" / "schematic"

# savefig's default: the finished-figure set. Panels and poster pass their own.
FIG_DIR = SI_DIR

# ----------------------------------------------------------------------- font


def _register_arial_nova():
    """Register Arial Nova from the user's font dirs so matplotlib can use it
    even when its font cache hasn't indexed it. Returns True if the regular
    face was registered (so mathtext can also use Arial Nova)."""
    variants = (
        "ArialNova.ttf",
        "ArialNova-Bold.ttf",
        "ArialNova-Italic.ttf",
        "ArialNova-BoldItalic.ttf",
    )
    dirs = (
        Path.home() / "Library" / "Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
    )
    found = False
    for d in dirs:
        for v in variants:
            p = d / v
            if p.exists():
                fm.fontManager.addfont(str(p))
                if v == "ArialNova.ttf":
                    found = True
    return found


# ---------------------------------------------------------------------- style

_COMMON = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial Nova", "Arial", "Helvetica", "DejaVu Sans"],
    # translucent white legend patch so in-axes legends stay readable over data
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.framealpha": 0.75,
    "legend.edgecolor": "none",
    "legend.fancybox": False,
    "axes.unicode_minus": False,  # Arial Nova lacks the U+2212 minus glyph
    "pdf.fonttype": 42,  # TrueType-embedded, editable text
    "svg.fonttype": "none",  # keep text editable in Illustrator
}

_CONTEXTS = {
    # print-sized supplement figures
    "si": {
        "axes.linewidth": 0.9,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
    },
    # large-type vector panels for the Illustrator schematic assembly
    "schematic": {
        "axes.linewidth": 2.2,  # thicker axis lines (easier to see in Illustrator)
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 17,
        "legend.fontsize": 17,
        # bar (action) plots set explicit x ticks → these show; the continuous
        # plots use empty tick lists → stay tickless. Tick width matches the
        # axis line width.
        "xtick.major.size": 5.5,
        "xtick.major.width": 2.2,
        "ytick.major.size": 0,  # no y ticks anywhere
        "ytick.major.width": 2.2,
        "axes.labelpad": 12,  # gap between axis labels and the tickless spines
    },
}


# Larger type for the standalone LM-elicitation validation figures. They are
# placed at reduced widths in the SI, so the base "si" sizes render small; these
# figures apply this rc profile (via plt.rc_context) around their drawing. The
# dense feature-structure grid and the UMAP maps keep the base "si" sizes.
SI_LARGE_RC = {
    "font.size": 11.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
}


def apply_style(context="si"):
    """Apply the shared manuscript aesthetic at the given context's sizes."""
    has_nova = _register_arial_nova()
    math = (
        {
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial Nova",
            "mathtext.it": "Arial Nova:italic",
            "mathtext.bf": "Arial Nova:bold",
        }
        if has_nova
        else {"mathtext.fontset": "dejavusans", "mathtext.default": "it"}
    )
    plt.rcParams.update({**_COMMON, **_CONTEXTS[context], **math})


def savefig(fig, name, png=True, *, out_dir=None, formats=("pdf",), tight=True):
    """Write <out_dir>/<name>.<ext> for each requested vector format, optionally
    with a PNG preview beside them, and close the figure.

    Defaults to a single PDF in figures/si/. Illustrator-bound panels pass
    their own directory (see the *_DIR constants above); the returned path is
    the first format's.
    """
    out_dir = Path(out_dir) if out_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    # `tight=False` keeps the full canvas, so a set of figures laid out with the
    # same fixed margins comes out with its axes in the SAME place on every page
    # -- which is what lets Illustrator stack them by page origin instead of by
    # eye. Cropping to content would undo that, since panels carry different
    # decorations (a top axis, a longer title) and would each crop differently.
    bbox = "tight" if tight else None
    for ext in formats:
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches=bbox)
    if png:
        fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches=bbox)
    plt.close(fig)
    return out_dir / f"{name}.{formats[0]}"


def panel_label(ax, letter, dx=-0.02, dy=1.04):
    """Bold panel letter at the top-left of an axes, in figure convention."""
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


# ------------------------------------------------------------------- palettes

OBSERVED_ACTIONS = ["no_share", "low_risk_share", "high_risk_share"]

# The schematic's palette: seaborn colorblind, muted via desat.
_CB = sns.color_palette("colorblind", desat=0.8)
ACTION_COLORS = {
    "no_share": to_hex(_CB[0]),  # blue
    "low_risk_share": to_hex(_CB[2]),  # green
    "high_risk_share": to_hex(_CB[1]),  # amber
}
# extra action beyond the three observed (the schematic's 4th alternative)
OTHER_ACTION_COLOR = to_hex(_CB[3])  # vermillion

ACTION_LABELS = {
    "no_share": "No share",
    "low_risk_share": "Low-risk share",
    "high_risk_share": "High-risk share",
}

# Inferred latent -> marker shape, keyed by study_registry's DV names. Every
# figure that distinguishes latents by shape reads this, so a shape means the
# same thing across the paper (the points panels, the poster set, and the
# aggregate correlation scatter).
DV_MARKERS = {"desire": "o", "intimacy": "s", "effort": "^"}

# Condition palettes — hex values match analysis/utils.R so Python and R
# figures share one visual language.
EFFORT_COLORS = {"low": "#B5C9A8", "high": "#4A7A4A"}
DESIRE_COLORS = {"low": "#C9A8B0", "high": "#7A4A5A"}

INTIMACY_LEVELS = [
    "max_formal",
    "somewhat_formal",
    "somewhat_intimate",
    "max_intimate",
]
INTIMACY_LABELS = {
    "max_formal": "Maximally formal",
    "somewhat_formal": "Somewhat formal",
    "somewhat_intimate": "Somewhat intimate",
    "max_intimate": "Maximally intimate",
}
# viridisLite::viridis(4, begin = 0.1, end = 0.85, option = "cividis") in utils.R
INTIMACY_COLORS = {
    lvl: to_hex(plt.get_cmap("cividis")(x))
    for lvl, x in zip(INTIMACY_LEVELS, np.linspace(0.1, 0.85, 4))
}

ALT_GREY = "#9AA0A6"

# Neutral marker for the observed actions in figures that color the alternatives
# by a continuous feature: the stars are drawn in this gray with text labels, so
# a feature colormap never competes with an action color.
OBSERVED_STAR_COLOR = "#8A8A8A"


def _muted_cmap(name, saturation=0.5, value=0.92, n=256):
    """A desaturated version of a matplotlib colormap: scales HSV saturation
    (and lightly caps brightness) while keeping the light->dark structure, so it
    matches the paper's muted palette (sage / mauve / cividis) without losing
    magnitude ordering or low-value visibility."""
    cols = plt.get_cmap(name)(np.linspace(0, 1, n))[:, :3]
    hsv = rgb_to_hsv(cols)
    hsv[:, 1] *= saturation
    hsv[:, 2] = np.minimum(hsv[:, 2] * value, 1.0)
    return LinearSegmentedColormap.from_list(f"{name}_muted", hsv_to_rgb(hsv))


# Sequential colormap for the LM-scored features (goal-satisfaction, risk): a
# desaturated viridis. Perceptually uniform (magnitude reads correctly) and
# colorblind-safe, muted to sit with the paper's earthy palette rather than
# viridis's neon; its dark low end keeps low-value points visible on white --
# important because the diagnostic minority (e.g. the low-g no-share-like
# alternatives) sits at the low end. Both features share it; change the feature
# hue here, not in the plotting scripts.
FEATURE_CMAP = _muted_cmap("viridis", saturation=0.6, value=0.92)
GOAL_CMAP = FEATURE_CMAP
RISK_CMAP = FEATURE_CMAP

# Slug -> paper label. Sourced from the shared study registry so the labels
# can't drift from the rest of the per-study metadata; re-exported here because
# the figure scripts already import it from plot_style.
from study_registry import STUDY_LABELS  # noqa: E402,F401
