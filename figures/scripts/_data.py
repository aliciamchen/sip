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
  participants, recompute cell means; 95% percentile interval), the Python
  port of analysis/utils.R::boot_cluster_means.

Studies whose inputs are missing return None so each figure script can skip
panels gracefully; CV outputs whose subject count no longer matches the data
CSV are flagged as stale (rendered anyway, with a warning, so layout iteration
can proceed before `make cv-<slug>` refreshes them).
"""

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
from study_registry import study  # noqa: E402
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
MODEL_LABELS = {
    "base": "Base",
    "discomfort_only": "Discomfort-only",
    "full": "Full",
}
PANEL_ORDER = [*MODEL_ORDER, "humans"]
PANEL_LABELS = {**MODEL_LABELS, "humans": "Humans"}

# The pre-2026-06-19 intimacy label still present in older CV outputs; the
# human CSVs were normalized at parse time (json_to_csv.py), so model-side
# tables are reconciled here.
_LEGACY_INTIMACY = {"neither": "somewhat_formal"}


def seed_for(name):
    """Deterministic 32-bit seed derived from a purpose string (the repo's
    SHA-256 seeding convention), e.g. seed_for("figures:food_inv_desire")."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def _outputs_dir(slug):
    return get_project_root() / "model" / "outputs" / slug


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


def load_cv_preds(slug):
    """Out-of-sample per-cell model predictions (cv_preds_summary.json), or
    None when the study's CV hasn't run. Drops the per-run columns and
    normalizes legacy intimacy labels."""
    path = _outputs_dir(slug) / "cv_preds_summary.json"
    if not path.exists():
        print(
            f"[{slug}] no CV predictions yet (make cv-{slug}) — skipping model panels"
        )
        return None
    with open(path) as f:
        preds = pd.DataFrame(json.load(f))
    preds = preds.drop(columns=[c for c in preds.columns if c.endswith("_runs")])
    if "intimacy_condition" in preds.columns:
        preds["intimacy_condition"] = preds["intimacy_condition"].replace(
            _LEGACY_INTIMACY
        )
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
        return json.load(f)


def correlation_for(comparison, model, dv):
    """The reported secondary correlation entry for (model, dv), or None."""
    if comparison is None:
        return None
    for row in comparison.get("secondary_correlations", []):
        if row["model"] == model and row["dv"] == dv:
            return row
    return None


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
    df, value_cols, group_cols, *, n_boot=1000, rng=None, seed=None
):
    """Observed per-cell means of `value_cols` grouped by `group_cols`, with
    95% subject-cluster bootstrap CIs (resampling subjects with replacement).

    Returns one row per observed cell with columns `<col>`, `<col>_ci_lower`,
    `<col>_ci_upper`. Cells that lose all trials in a resample are dropped
    pairwise from that resample (matching utils.R::boot_cluster_means).
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
    return out
