"""Shared data layer for the main results figures (figures/results/).

Thin wrappers over the model pipeline's own loaders, so the figures consume
exactly the quantities the paper reports rather than re-deriving them:

- Per-trial human belief updates come from model/cv/model_comparison.py's
  _prepare_data (which itself uses model/inverse/_helpers.py::_load_long, the
  fail-fast prior/posterior pivot). Cell-key columns already match
  cv_preds_summary.json: `action` as an int index into
  plot_style.OBSERVED_ACTIONS, `intimacy_condition` as the verbal slug, and
  `desire_condition` / `effort_condition` as 'low'/'high'.
- Model predictions are the out-of-sample LOSO CV per-cell deltas
  (cv_preds_summary.json); correlation and held-out-LL annotations are read
  from cv_model_comparison.json (written by `make model-comparison`) so every
  number printed on a figure equals the paper's.
- Cell-mean CIs use the project-standard subject-cluster bootstrap (resample
  participants, recompute cell means; 95% percentile interval).

Studies whose inputs are missing return None so each figure script can skip
panels gracefully; CV outputs whose subject count no longer matches the data
CSV are flagged as stale (rendered anyway, with a warning, so layout iteration
can proceed before `make cv-<slug>` refreshes them).
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model" / "cv"))

import model_comparison as _mc  # noqa: E402  (also puts model/inverse on sys.path)
from plot_style import OBSERVED_ACTIONS  # noqa: E402
from study_registry import (  # noqa: E402
    PREREG_BASE_KEY,
    reported_base,
    study,
)
from utils import get_project_root  # noqa: E402

# model_comparison's per-study cell/DV spec is itself derived from the study
# registry, so this and the registry helpers below never disagree.
STUDY_SPECS = _mc.STUDY_SPECS


def condition_cols(slug):
    """Condition-level grouping columns for the by-condition panels: the
    display action label plus the study's given conditions (drops scenario, so
    the bar/line panels average over scenarios)."""
    return ["action_label", *study(slug).given_conditions]


def dvs_display(slug):
    """The study's inferred DVs as (human_update_col, model_delta_col,
    display_label) tuples, in panel order."""
    return [(dv.update_col, dv.delta_col, dv.label) for dv in study(slug).dvs]


MODEL_ORDER = ["base", "discomfort_only", "full"]
# Figure column headings. Named for what each model IS rather than for its role
# in the comparison, so a reader looking only at a figure can tell the ablations
# apart: the vanilla model is inverse planning without any social term, the full
# model is inverse planning with one. The manuscript prose and the generated
# tables now say "vanilla" too; the variant KEYS stay `base`, so nothing on disk
# was renamed.
MODEL_LABELS = {
    "base": "Vanilla inv plan",
    "discomfort_only": "Discomfort-only",
    "full": "Social inv plan",
}
PANEL_LABELS = {**MODEL_LABELS, "humans": "Humans"}

# Which base ablation the paper reports. Defined in study_registry.py (the
# single source of truth both this module and model_comparison.py read) —
# `reported_base` promotes the `base_shared` fit to "Base" in the studies whose
# preregistered base also swaps the comparison set. See that module for why.
_announced_promotion = set()


def _promotion_map(slug, present):
    """Raw-variant -> reported-variant renaming for one study, or None.

    Returns None when the study reports its preregistered base (2a/2b/3b), or
    when the promoted fit is not present in this artifact (so a partially
    computed study degrades to the preregistered base rather than losing its
    base column entirely).
    """
    promoted = reported_base(slug)
    if promoted == "base":
        return None
    if promoted not in present:
        # Falling back to the preregistered base would label a DIFFERENT model
        # "Base" than the other studies in the same figure — two definitions of
        # Base in one panel row, with nothing on the figure to say so. Degrade
        # rather than crash (so layouts can be iterated before a refit lands),
        # but never quietly.
        print(
            f"[{slug}] WARNING: `{promoted}` is missing from these outputs, so "
            f"Base falls back to the PREREGISTERED broadcast-set base — which is "
            f"not what the other studies show. Re-run `make fit-{slug} cv-{slug}` "
            f"and `make model-comparison` before using this figure."
        )
        return None
    if slug not in _announced_promotion:
        print(
            f"[{slug}] reporting `{promoted}` as Base (base utility on full's "
            f"comparison set); preregistered broadcast base kept as "
            f"`{PREREG_BASE_KEY}`"
        )
        _announced_promotion.add(slug)
    # Built as one simultaneous mapping and applied with .map, not chained
    # .replace calls, so `base` cannot be renamed twice.
    return {"base": PREREG_BASE_KEY, promoted: "base"}


# The pre-2026-06-19 intimacy label still present in older CV outputs; the
# human CSVs were normalized at parse time (json_to_csv.py), so model-side
# tables are reconciled here.
_LEGACY_INTIMACY = {"neither": "somewhat_formal"}


def seed_for(name):
    """Deterministic 32-bit seed derived from a purpose string (the repo's
    SHA-256 seeding convention), e.g. seed_for("figures:food_inv_desire")."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def _outputs_dir(slug, config_tag=None):
    """A study's model outputs. `config_tag` selects a non-default run config's
    directory (`RunConfig.tag()`, e.g. "uniform-noreweight"), mirroring
    `RunConfig.outputs_dir`; None is the reported config at the study root."""
    root = get_project_root() / "model" / "outputs" / slug
    return root if config_tag is None else root / "alt" / config_tag


def load_trials(slug):
    """Per-trial belief updates with cv-compatible cell keys, or None (with a
    printed note) when the study's data CSV doesn't exist yet."""
    csv = get_project_root() / "data" / slug / "main_trials_long.csv"
    if not csv.exists():
        print(
            f"[{slug}] no data yet ({csv.relative_to(get_project_root())}) — skipping human panels"
        )
        return None
    return _mc._prepare_data(slug)


def load_cv_preds(slug, config_tag=None):
    """Out-of-sample per-cell model predictions (cv_preds_summary.json), or
    None when the study's CV hasn't run. Drops the per-run columns and
    normalizes legacy intimacy labels.

    `config_tag` reads a non-default run config's CV instead of the reported one
    (see `_outputs_dir`) — used by the preregistration-deviation figure to draw
    the eta = 0 model's predictions beside the reported model's. Note that the
    reported-base promotion below applies to whichever config is loaded, which is
    what makes the two comparable: both sides label the same variant "base"."""
    path = _outputs_dir(slug, config_tag) / "cv_preds_summary.json"
    if not path.exists():
        where = "" if config_tag is None else f" for config {config_tag}"
        print(f"[{slug}] no CV predictions{where} yet ({path}) — skipping")
        return None
    with open(path) as f:
        preds = pd.DataFrame(json.load(f))
    preds = preds.drop(columns=[c for c in preds.columns if c.endswith("_runs")])
    if "intimacy_condition" in preds.columns:
        preds["intimacy_condition"] = preds["intimacy_condition"].replace(
            _LEGACY_INTIMACY
        )
    ren = _promotion_map(slug, set(preds["model"]))
    if ren is not None:
        preds["model"] = preds["model"].map(lambda v: ren.get(v, v))
    return preds


def load_comparison(slug):
    """cv_model_comparison.json (the paper's bootstrap statistics), or None."""
    path = _outputs_dir(slug) / "cv_model_comparison.json"
    if not path.exists():
        print(
            f"[{slug}] no cv_model_comparison.json (make model-comparison) — omitting r/LL annotations"
        )
        return None
    with open(path) as f:
        comparison = json.load(f)
    return _apply_reported_base(slug, comparison)


def _apply_reported_base(slug, comparison):
    """Rewrite a cv_model_comparison.json payload so `base` names the variant
    the paper reports (see study_registry.reported_base).

    Touches every block that names a variant: `primary` entries are keyed by a
    "full_minus_<variant>" string, and `secondary_correlations` and
    `condition_gradients` rows by a `model` field. Missing one of them would put
    two meanings of `base` in a single payload — the promoted variant under its
    raw name `base_shared`, and the preregistered broadcast variant under `base`.
    Returns a copy — the on-disk artifact keeps the raw keys.
    """
    _MODEL_KEYED = ("secondary_correlations", "condition_gradients")
    present = {
        row["model"] for block in _MODEL_KEYED for row in comparison.get(block, [])
    }
    present |= {
        entry["comparison"].removeprefix("full_minus_")
        for entry in comparison.get("primary", [])
    }
    ren = _promotion_map(slug, present)
    if ren is None:
        return comparison
    comparison = copy.deepcopy(comparison)
    for entry in comparison.get("primary", []):
        variant = entry["comparison"].removeprefix("full_minus_")
        entry["comparison"] = f"full_minus_{ren.get(variant, variant)}"
    for block in _MODEL_KEYED:
        for row in comparison.get(block, []):
            row["model"] = ren.get(row["model"], row["model"])
    return comparison


def warn_if_stale(slug, trials, comparison):
    """Print a staleness warning when the CV outputs were produced from a
    different participant sample than the current data CSV. Returns True when
    stale (callers still render, so layouts can be iterated pre-refit)."""
    if trials is None or comparison is None:
        return False
    n_data = trials["subject_id"].nunique()
    n_cv = comparison.get("n_subjects")
    if n_cv is not None and n_cv != n_data:
        print(
            f"[{slug}] STALE model outputs: CV ran on {n_cv} subjects but "
            f"data CSV has {n_data} — model panels show the old vintage; "
            f"re-run `make fit-{slug} cv-{slug}` and `make model-comparison`."
        )
        return True
    return False


def action_label_col(df, action_col="action"):
    """Map the int action index to its slug (plot_style.OBSERVED_ACTIONS)."""
    return df[action_col].map(dict(enumerate(OBSERVED_ACTIONS)))


def bootstrap_cell_means(
    df, value_cols, group_cols, *, n_boot=1000, rng=None, seed=None, return_boots=False
):
    """Observed per-cell means of `value_cols` grouped by `group_cols`, with
    95% subject-cluster bootstrap CIs (resampling subjects with replacement).

    Returns one row per observed cell with columns `<col>`, `<col>_ci_lower`,
    `<col>_ci_upper`. Cells that lose all trials in a resample are dropped
    pairwise from that resample.

    With `return_boots`, also returns {col: (n_boot, n_cells) resampled means}.
    Every `value_cols` entry is resampled under the SAME subject draws, so a
    statistic computed across DVs (the pooled model-vs-human correlation) can be
    recomputed per resample without the DVs drifting out of alignment.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    group_cols = list(group_cols)
    value_cols = list(value_cols)

    cell_codes, cells = pd.factorize(pd.MultiIndex.from_frame(df[group_cols]))
    cells = pd.MultiIndex.from_tuples(cells, names=group_cols)
    subj_codes, _subjects = pd.factorize(df["subject_id"])
    n_subj, n_cells = len(_subjects), len(cells)

    # Multinomial subject weights == counts of a with-replacement resample.
    weights = rng.multinomial(n_subj, np.full(n_subj, 1.0 / n_subj), size=n_boot)

    out = pd.DataFrame(cells.to_frame(index=False))
    boots = {}
    for col in value_cols:
        v = df[col].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        mat_sum = np.zeros((n_subj, n_cells))
        mat_cnt = np.zeros((n_subj, n_cells))
        np.add.at(mat_sum, (subj_codes[ok], cell_codes[ok]), v[ok])
        np.add.at(mat_cnt, (subj_codes[ok], cell_codes[ok]), 1.0)

        boot_sum = weights @ mat_sum
        boot_cnt = weights @ mat_cnt
        with np.errstate(invalid="ignore", divide="ignore"):
            boot_means = np.where(boot_cnt > 0, boot_sum / boot_cnt, np.nan)
            obs = mat_sum.sum(axis=0) / mat_cnt.sum(axis=0)
        out[col] = obs
        out[f"{col}_ci_lower"] = np.nanpercentile(boot_means, 2.5, axis=0)
        out[f"{col}_ci_upper"] = np.nanpercentile(boot_means, 97.5, axis=0)
        if return_boots:
            boots[col] = boot_means
    return (out, boots) if return_boots else out
