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
