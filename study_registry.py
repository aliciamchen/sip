"""Single source of truth for per-study metadata shared across the pipeline.

Both the model-comparison code (`model/cv/model_comparison.py`) and the figure
scripts (`figures/scripts/`) key off the same per-study facts: which conditions
are given (fixed and shown to participants), which latent(s) the study infers
and under which belief-update / prediction columns, the paper-facing label, and
the stimulus domain. Defining them once here keeps the model's cell-grid and
correlation code and the figures from drifting apart -- e.g. so Study 3a's
figure config can't disagree with Study 1b's, or with what
`model_comparison.py` correlates.

This module is metadata only. It deliberately does NOT know about JAX array
shapes or memo axes (those stay in `model/tables.py` and
`model/inverse/_helpers.py`, which build the fit-side tensors from the raw
condition columns), nor about matplotlib palettes (those stay in
`plot_style.py`, keyed by the same condition/DV names used here). It imports
nothing from the project, so it is safe to import from either side.

Column-name conventions (produced upstream by `_load_long` / `_prepare_data`
and by the CV dispatcher's `cv_preds_summary.json`):
  - `<rating>_update` -- a participant's belief update for a rating slider.
  - `delta_<latent>`  -- the model's out-of-sample per-cell prediction.
  - `action`          -- observed action as an int index (0/1/2); the figures
    map it to the `action_label` slug for display.
  - condition columns -- `intimacy_condition` (verbal slug) and
    `desire_condition` / `effort_condition` ('low'/'high').
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DV:
    """One inferred latent scored against human judgments."""

    update_col: str  # human belief-update column (from _prepare_data)
    delta_col: str  # model per-cell prediction column (cv_preds_summary.json)
    name: str  # short id: "desire" | "intimacy" | "effort"
    label: str  # display label for figure axes/panels


@dataclass(frozen=True)
class Study:
    slug: str
    paper_label: str  # e.g. "Study 1a"
    domain: str  # "food" | "nonfood"
    given_conditions: tuple[str, ...]  # condition columns fixed & shown
    dvs: tuple[DV, ...]  # inferred latents, in figure/panel order

    @property
    def cell_keys(self) -> list[str]:
        """Columns identifying a scenario x condition cell, matching both
        `_prepare_data` (human) and `cv_preds_summary.json` (model). Used for
        the per-cell correlations and the model-vs-human scatter."""
        return ["scenario_label", "action", *self.given_conditions]

    @property
    def is_joint(self) -> bool:
        """True for the two-latent joint studies (1b/2b/3a/3b)."""
        return len(self.dvs) > 1

    @property
    def short_label(self) -> str:
        """ "1a" -- the compact label the figures print, from `paper_label`."""
        return self.paper_label.removeprefix("Study ")

    @property
    def number(self) -> str:
        """ "1" -- the paper study number this sub-study belongs to."""
        return self.short_label[:-1]

    @property
    def substudy(self) -> str:
        """ "a" or "b" -- which half of its paper study this is."""
        return self.short_label[-1]


_DESIRE = DV("desire_rating_update", "delta_desire", "desire", "Desire")
_INTIMACY = DV("intimacy_rating_update", "delta_intimacy", "intimacy", "Intimacy")
_EFFORT = DV(
    "effort_rating_update", "delta_effort", "effort", "Effort of low-risk share"
)

STUDIES: dict[str, Study] = {
    # Study 1a: single-slider desire inference, so its human DV is the lone
    # `response` rating rather than a named `<latent>_rating`.
    "food_inv_desire": Study(
        slug="food_inv_desire",
        paper_label="Study 1a",
        domain="food",
        given_conditions=("intimacy_condition", "effort_condition"),
        dvs=(DV("response_update", "delta_desire", "desire", "Desire"),),
    ),
    "food_inv_joint_de": Study(
        slug="food_inv_joint_de",
        paper_label="Study 1b",
        domain="food",
        given_conditions=("intimacy_condition",),
        dvs=(_DESIRE, _EFFORT),
    ),
    "food_inv_intimacy": Study(
        slug="food_inv_intimacy",
        paper_label="Study 2a",
        domain="food",
        given_conditions=("desire_condition", "effort_condition"),
        dvs=(_INTIMACY,),
    ),
    "food_inv_joint_ie": Study(
        slug="food_inv_joint_ie",
        paper_label="Study 2b",
        domain="food",
        given_conditions=("desire_condition",),
        dvs=(_INTIMACY, _EFFORT),
    ),
    # Study 3 (nonfood stimulus set): 3a mirrors 1b, 3b mirrors 2b.
    "nonfood_inv_joint_de": Study(
        slug="nonfood_inv_joint_de",
        paper_label="Study 3a",
        domain="nonfood",
        given_conditions=("intimacy_condition",),
        dvs=(_DESIRE, _EFFORT),
    ),
    "nonfood_inv_joint_ie": Study(
        slug="nonfood_inv_joint_ie",
        paper_label="Study 3b",
        domain="nonfood",
        given_conditions=("desire_condition",),
        dvs=(_INTIMACY, _EFFORT),
    ),
}


def study(slug: str) -> Study:
    return STUDIES[slug]


# Slug -> paper label, for figures that only need the label (re-exported by
# plot_style.py, which many figure scripts already import it from).
STUDY_LABELS: dict[str, str] = {slug: s.paper_label for slug, s in STUDIES.items()}

# The active roster in paper order (1a, 1b, 2a, 2b, 3a, 3b). Every consumer that
# iterates the studies reads this rather than repeating the slug list, so adding
# or reordering a study is a one-place edit.
SLUGS: tuple[str, ...] = tuple(STUDIES)


def studies() -> list[Study]:
    """Every active study, in paper order."""
    return list(STUDIES.values())


def study_groups() -> list[tuple[str, list[Study]]]:
    """The sub-studies grouped by paper study number, in order:
    [("1", [1a, 1b]), ("2", [2a, 2b]), ("3", [3a, 3b])].

    Figures that pair a study's halves in one file (the model-vs-human scatters
    and correlation grids) iterate this instead of hardcoding the pairing.
    """
    groups: dict[str, list[Study]] = {}
    for s in STUDIES.values():
        groups.setdefault(s.number, []).append(s)
    return list(groups.items())


def slugs_given(condition: str) -> list[str]:
    """Slugs whose given (fixed and shown) conditions include `condition`.

    `slugs_given("intimacy_condition")` is the given-relationship set (1a/1b/3a,
    also the only studies with a relationship-free base elicitation);
    `slugs_given("desire_condition")` is the given-desire set (2a/2b/3b).
    """
    return [slug for slug, s in STUDIES.items() if condition in s.given_conditions]


# ---------------------------------------------------------------------------
# Which base ablation the paper reports
# ---------------------------------------------------------------------------
# The preregistrations specify a `base` ablation whose LM alternative set is
# elicited WITHOUT the relationship paragraph and then broadcast across the
# relationship axis, so base's predictions are relationship-invariant. In the
# given-relationship studies that makes base differ from full along TWO axes at
# once -- the discomfort term AND the comparison set -- so `full - base` is not
# a test of the discomfort term. Measured on the 2026-07-31 CV run the
# comparison-set half is large enough to reverse the sign in Study 1b (utility
# +0.0214, comparison set -0.0447, total -0.0232 per-trial held-out LL).
#
# The main text therefore reports the base utility scored against full's
# relationship-conditioned comparison set -- the `base_shared` fit -- as
# "Base", so `full - base` isolates the discomfort term. The preregistered
# broadcast variant is reported in the preregistration-deviation section.
#
# This is a REPORTING-layer promotion only: the model still fits both variants
# under their own names and every output file keeps its raw keys, so the two
# are always separable and no refitting is needed.
PROMOTED_BASE_VARIANT = "base_shared"
PREREG_BASE_KEY = "base_prereg"
PREREG_BASE_LABEL = "Base (preregistered)"


def reported_base(slug: str) -> str:
    """The variant key the paper's "Base" column refers to, for one study.

    Derived from the registry rather than a slug list: a relationship-free base
    vintage exists exactly when the relationship is a *given* condition, since
    the given-desire studies (2a/2b/3b) never show a relationship paragraph and
    so their `base` already shares full's comparison set.
    """
    given = STUDIES[slug].given_conditions
    return PROMOTED_BASE_VARIANT if "intimacy_condition" in given else "base"
