"""Hypothesis-matched statistics to sit beside the preregistered model comparison.

The preregistered primary metric is the global per-trial held-out log-likelihood
(`model_comparison.py`). It is a valid test and stays primary. It is also a
low-power one for the claim the paper actually makes, and this module measures
why and supplies the matched alternative.

The problem, measured rather than asserted. The paper's claim is about a
*modulation*: how the given relationship changes what an action implies about
desire and the world state (Studies 1a/1b/3a), and how the given desire changes
what it implies about intimacy and the world state (2a/2b/3b). That modulation is
1--3% of trial-level variance. Most of the rest is within-cell response noise no
model can predict, and nearly all of the predictable remainder is the scenario x
action structure, which the base ablation already captures. A global fit index
averaged over every trial is therefore close to blind to it: a model can capture
the modulation essentially perfectly and move the global held-out likelihood by a
few thousandths of a nat.

Two functions, answering two questions:

`variance_decomposition` -- where does the trial-level variance live? Splits each
DV into within-cell (irreducible) and between-cell (explainable) parts, then
isolates the focal condition's own variance component, and splits *that* into a
scenario-consistent component and a scenario-specific one. All are sampling-bias
corrected; the raw moments overstate every one of them, badly at ~20 observations
per cell.

`condition_gradients` -- does the model predict the modulation? Scores the
contrast itself: the change in belief update across the focal condition's ordered
levels, per observed action, human against each model variant's held-out
predictions. This is the quantitative form of the directional predictions the
preregistrations state alongside the model comparison and commit to assessing.

Guardrails, so that this is a test and not a search over metrics. The contrast is
derived from the hypothesis, not from the residuals: the focal condition is
whichever latent the study *gives* the observer, which the study registry already
records, and the contrast is the ordered linear trend across its levels. It is
computed identically for every study and every model variant, including where it
is unflattering -- Study 2a's full model recovers well under half the human
desire gradient, and the scenario-specific component of the modulation is close
to unpredictable everywhere. Report those with the rest.

What stays fixed: the unit of resampling is the participant, as preregistered,
and every trial is used. Only the test statistic changes, from a global fit index
to the contrast the hypothesis is about. Aggregating the data to cell means
instead would be the wrong repair -- at ~20 observations per cell the focal
contrast still sits below the noise floor of a single cell mean (per-cell SNR
0.5--0.7 in Study 1b), so it buys no resolution on the effect of interest, while
cutting the observation count ~20-fold and weakening the cross-validation's
protection against overfitting. It would produce friendlier point estimates
carrying less evidence.

Two assumptions that the numbers here rest on, neither of which the design lets
us discharge, both stated so that a reader can discount accordingly:

1. The scenario-consistent / scenario-specific split treats a cell mean's
   sampling noise as independent across scenarios. The design is fully
   within-subject -- every participant sees all 16 scenarios -- so a
   participant's response bias recurs in the same (action, level) cell across
   scenarios and correlates those deviations. The correction subtracted from the
   scenario-consistent part is therefore too small and the one subtracted from
   the scenario-specific part too large, biasing the reported scenario-specific
   share DOWN. That is the direction that flatters the claim that the
   scenario-specific component is largely unpredictable, so treat that share as
   a lower bound rather than an estimate.
2. `human_minus_model_ci_95` resamples participants while holding the model
   gradient fixed. The held-out predictions are not in fact independent of the
   participant sample -- each LOSO fold refits the weights on the other 15
   scenarios of these same people -- so the interval is conditional on the fitted
   model rather than a joint resample of both. It answers "would a fresh sample
   of participants have produced a human gradient different from THIS fitted
   model's prediction", which is weaker than "the model's gradient is reliably
   the wrong size". Do not make the stronger claim from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Ordered levels of each condition column, lowest to highest. The contrast
#: below is signed by this order, so it fixes what "gradient" means: for the
#: relationship, formal -> intimate; for desire and effort, low -> high.
LEVEL_ORDER: dict[str, tuple[str, ...]] = {
    "intimacy_condition": (
        "max_formal",
        "somewhat_formal",
        "somewhat_intimate",
        "max_intimate",
    ),
    "desire_condition": ("low", "high"),
}


def focal_condition(study) -> str:
    """The given condition whose modulating effect the study tests.

    Read off the design rather than listed per slug: a study gives the observer
    exactly one of relationship or desire, and that is the variable whose effect
    on the *other* inferences the paper's hypothesis is about. The physical world
    state is not focal anywhere -- where it is given (1a, 2a) its effect is the
    standard inverse-planning one that every variant, ablated or not, already
    represents.
    """
    given = study.given_conditions
    if "intimacy_condition" in given:
        return "intimacy_condition"
    if "desire_condition" in given:
        return "desire_condition"
    raise ValueError(f"{study.slug} gives neither relationship nor desire")


def ordered_levels(data, focal, slug=""):
    """The focal condition's levels, in the declared order, validated.

    Filtering LEVEL_ORDER by the data (rather than the data by LEVEL_ORDER) would
    silently DROP any label this module has not been told about -- and silently
    rescale the contrast, since `trend_coefficients` normalizes to however many
    levels survive, so a 3-level survivor would report a 2-step change under the
    same field name as a 4-level study's 3-step one. The 'neither' ->
    'somewhat_formal' rename is exactly this class of change, and
    figures/scripts/_data.py still carries a legacy remap that this module does
    not, so an unrecognized label is an error rather than a filter.
    """
    present = set(data[focal].dropna().unique())
    known = set(LEVEL_ORDER[focal])
    if unknown := present - known:
        raise ValueError(
            f"{slug or focal}: level(s) {sorted(unknown)} of {focal} are not in "
            f"LEVEL_ORDER, so their position in the ordering is undefined. Add "
            f"them (or remap them upstream) rather than letting them be dropped."
        )
    return [lv for lv in LEVEL_ORDER[focal] if lv in present]


def trend_coefficients(n_levels: int) -> np.ndarray:
    """Linear-trend contrast weights in endpoint units.

    Centered (so they annihilate any additive offset) and scaled so that under a
    perfectly linear trend the contrast equals the total change from the lowest
    to the highest level. For two levels this is exactly the difference of means,
    so the two-level and four-level studies report the same quantity on the same
    scale. Using the trend rather than just the endpoints keeps the two
    intermediate relationship levels in the estimate instead of discarding half
    the design.
    """
    x = np.arange(n_levels, dtype=float)
    d = x - x.mean()
    return d * (n_levels - 1) / float(d @ x)


def _pooled_within_var(counts, within_var) -> float:
    """Pooled within-cell variance, the irreducible part of a trial's value.

    Weighted by degrees of freedom rather than averaged over cells, so cells with
    more observations count for more; cells with a single observation carry no
    variance estimate and drop out.
    """
    ok = counts > 1
    dof = counts[ok] - 1
    if dof.sum() == 0:
        return float("nan")
    return float((within_var[ok] * dof).sum() / dof.sum())


def _decompose(wide, rel_noise, n_scen, resid_of_row):
    """The focal variance component and its scenario split, from a (group, level)
    matrix of cell means. Shared by the point estimate and every bootstrap draw.

    Returns (ms_rel, ms_specific, ms_consistent), bias-corrected and coherent:
    the two parts are clipped so that they sum to the whole and each share lands
    in [0, 1]. Clipping the three independently -- which an earlier version did
    -- lets the scenario-specific share exceed 1 whenever the true focal effect
    is near zero and the consistent part clips first; measured on synthetic data
    with a near-null effect that happened on 12 of 200 seeds, once at 291%.
    """
    n_lev = wide.shape[1]
    rel = wide - wide.mean(axis=1, keepdims=True)
    cons = np.zeros_like(rel)
    for r in np.unique(resid_of_row):
        sel = resid_of_row == r
        cons[sel] = rel[sel].mean(axis=0, keepdims=True)
    spec = rel - cons

    ms_rel = max(float((rel**2).mean()) - rel_noise, 0.0)
    ms_spec = float((spec**2).mean()) - rel_noise * (1.0 - 1.0 / n_scen)
    ms_spec = float(np.clip(ms_spec, 0.0, ms_rel))
    return ms_rel, ms_spec, ms_rel - ms_spec


def _cell_layout(cells, group_keys, focal, levels):
    """Map each cell row to its (group row, focal column) slot in the wide matrix,
    plus each group row's residual-key id (everything but scenario)."""
    grp_key = pd.MultiIndex.from_frame(cells[group_keys])
    groups = grp_key.unique()
    row = groups.get_indexer(grp_key)
    col = cells[focal].map({lv: i for i, lv in enumerate(levels)}).to_numpy()

    # Which group rows share everything but the scenario -- the rows the
    # scenario-consistent component averages over.
    resid_keys = [k for k in group_keys if k != "scenario_label"]
    resid = pd.MultiIndex.from_frame(groups.to_frame(index=False)[resid_keys])
    resid_of_row = resid.unique().get_indexer(resid)
    return row, col, len(groups), resid_of_row, len(groups) / len(resid.unique())


def variance_decomposition(data, study, update_col, dv_name, *, n_boot=0, rng=None):
    """Where the trial-level variance of one DV lives, with sampling bias removed.

    Every "observed" moment here is inflated by sampling noise in the cell means,
    so each is corrected by the sampling variance of the quantity it is computed
    from. Uncorrected, a design with ~20 observations per cell reports a focal
    effect roughly twice its true size.

    With `n_boot > 0` the shares also carry participant-bootstrap intervals --
    the paper quotes these numbers against each other across studies, and a
    single estimate's SD is roughly 15% of its value at this cell size. The
    within-cell variance and the cell counts are held at their observed values
    across draws: both are estimated from every trial in the study, so their own
    sampling error is negligible beside the components being bootstrapped.
    """
    keys = study.cell_keys
    focal = focal_condition(study)
    levels = ordered_levels(data, focal, study.slug)
    if len(levels) < 2:
        return None

    g = data.groupby(keys)[update_col]
    cells = g.agg(["sum", "count", "var"]).reset_index()
    counts = cells["count"].to_numpy(dtype=float)
    sig2_w = _pooled_within_var(counts, cells["var"].to_numpy(dtype=float))
    cell_mean = (cells["sum"] / cells["count"]).to_numpy()

    # Between-cell (explainable) variance: observed spread of the cell means,
    # less the sampling variance each of those means carries.
    var_cell_obs = float(cell_mean.var(ddof=1))
    var_cell_noise = float((sig2_w / counts).mean())
    var_between = max(var_cell_obs - var_cell_noise, 0.0)
    total = var_between + sig2_w

    # The focal condition's own variance component: how far each cell mean sits
    # from the average over the focal levels of everything else held fixed
    # (scenario x action x any other given condition). This is a genuine variance
    # component of the DV -- it has mean zero by construction, so its mean square
    # IS its variance, and it is the part of the explainable variance that exists
    # only because the focal condition was manipulated. Deliberately NOT the
    # trend contrast: a contrast summarizes the effect's direction and size (that
    # is the gradient test below), but its mean square is not a share of the DV's
    # variance and cannot be quoted as one.
    group_keys = [k for k in keys if k != focal]
    row, col, n_groups, resid_of_row, n_scen = _cell_layout(
        cells, group_keys, focal, levels
    )
    n_lev = len(levels)
    if len(cells) != n_groups * n_lev:
        raise RuntimeError(
            f"{study.slug}: the {focal} grid is ragged ({len(cells)} cells for "
            f"{n_groups} groups x {n_lev} levels); the decomposition assumes "
            f"every group has every level."
        )
    wide = np.full((n_groups, n_lev), np.nan)
    wide[row, col] = cell_mean
    cnt_wide = np.full((n_groups, n_lev), np.nan)
    cnt_wide[row, col] = counts

    # Sampling variance of one deviation: each cell mean carries sigma_w^2 / n,
    # and subtracting the mean of the L levels leaves (1 - 1/L) of it (exact when
    # the levels' counts are equal, which the design balances).
    rel_noise = float((sig2_w / cnt_wide).mean() * (1.0 - 1.0 / n_lev))
    ms_rel, ms_spec, ms_cons = _decompose(wide, rel_noise, n_scen, resid_of_row)

    out = {
        "dv": dv_name,
        "focal_condition": focal,
        "n_levels": n_lev,
        "within_cell_var": sig2_w,
        "between_cell_var": var_between,
        "total_var": total,
        "frac_explainable": var_between / total if total > 0 else float("nan"),
        "focal_var": ms_rel,
        "focal_frac_of_total": ms_rel / total if total > 0 else float("nan"),
        # Ratio of two independently corrected estimates over two different cell
        # subsets, so nothing constrains it to <= 1; it is reported unclipped so
        # that a value above 1 shows up as the disagreement it is rather than
        # being hidden. Observed max across the six studies is 0.08.
        "focal_frac_of_explainable": (
            ms_rel / var_between if var_between > 0 else float("nan")
        ),
        "focal_scenario_consistent_var": ms_cons,
        "focal_scenario_specific_var": ms_spec,
        "focal_frac_scenario_specific": (
            ms_spec / ms_rel if ms_rel > 0 else float("nan")
        ),
        "bias_correction": {
            "pooled_within_var": sig2_w,
            "cell_mean_sampling_var": var_cell_noise,
            "focal_deviation_sampling_var": rel_noise,
            "uncorrected_between_cell_var": var_cell_obs,
            "uncorrected_focal_var": float(
                ((wide - wide.mean(1, keepdims=True)) ** 2).mean()
            ),
        },
    }
    if n_boot:
        out.update(
            _decomposition_ci(
                data,
                keys,
                update_col,
                cells,
                row,
                col,
                n_groups,
                n_lev,
                resid_of_row,
                n_scen,
                sig2_w,
                rel_noise,
                n_boot,
                rng,
            )
        )
    return out


def _decomposition_ci(
    data,
    keys,
    update_col,
    cells,
    row,
    col,
    n_groups,
    n_lev,
    resid_of_row,
    n_scen,
    sig2_w,
    rel_noise,  # unused by the bootstrap: recomputed per draw from multiplicities
    n_boot,
    rng,
):
    """Participant-bootstrap intervals on the decomposition's reported shares.

    The noise correction is recomputed per resample from that resample's subject
    MULTIPLICITIES, not held at its observed-data value. A bootstrap draw holds
    only ~63% unique participants, so a duplicated participant's response bias is
    counted twice and the resampled cell means are more variable than
    `sigma_w^2 / n` predicts. Correcting them by the observed-data amount
    under-corrects, and the whole bootstrap distribution lands ABOVE the point
    estimate -- measured here, a [0.047, 0.080] interval around a 0.039 estimate.
    That is the same mislocation `_secondary_correlation` documents for the
    correlation CI (notes/2026-08-03-correlation-ci-audit.md), reached by the
    same route.

    With `w_s` the multiplicity of subject `s` and `C_sc` their trial count in
    cell `c`, the resampled mean's sampling variance is
    `sigma_w^2 * sum_s w_s^2 C_sc / (sum_s w_s C_sc)^2`, which reduces to the
    observed `sigma_w^2 / n_c` at `w = 1`.
    """
    S, C = _subject_cell_matrices(data, keys, update_col, cells)
    n_subj = S.shape[0]
    idx = rng.integers(0, n_subj, size=(n_boot, n_subj))
    fracs, spec_fracs = [], []
    for i in idx:
        w = np.bincount(i, minlength=n_subj).astype(float)
        num, den = w @ S, w @ C
        if (den == 0).any():
            continue
        cm = num / den
        cell_noise_b = sig2_w * ((w**2) @ C) / den**2
        wide = np.full((n_groups, n_lev), np.nan)
        wide[row, col] = cm
        noise_wide = np.full((n_groups, n_lev), np.nan)
        noise_wide[row, col] = cell_noise_b
        rel_noise_b = float(np.nanmean(noise_wide) * (1.0 - 1.0 / n_lev))
        var_between_b = max(float(cm.var(ddof=1)) - float(cell_noise_b.mean()), 0.0)
        total_b = var_between_b + sig2_w
        ms_rel_b, ms_spec_b, _ = _decompose(wide, rel_noise_b, n_scen, resid_of_row)
        if total_b > 0:
            fracs.append(ms_rel_b / total_b)
        if ms_rel_b > 0:
            spec_fracs.append(ms_spec_b / ms_rel_b)

    def ci(v):
        return (
            [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            if len(v) > 1
            else [float("nan"), float("nan")]
        )

    return {
        "focal_frac_of_total_ci_95": ci(fracs),
        "focal_frac_scenario_specific_ci_95": ci(spec_fracs),
        "n_boot": int(n_boot),
    }


def _subject_cell_matrices(data, keys, update_col, cells):
    """Per-(subject, cell) sums and counts, columns in `cells`' row order.

    Same trial-weighted convention as `model_comparison.subject_cell_matrices`
    (a subset's cell mean is sum-of-sums over sum-of-counts, not a mean of
    subject means); kept local because that helper is keyed to a cells DataFrame
    while this module also needs the same layout for the group grid, and
    importing it would make model_comparison and this module circular.
    """
    cell_pos = {tuple(r): i for i, r in enumerate(cells[list(keys)].to_numpy())}
    subj_pos = {s: i for i, s in enumerate(sorted(data["subject_id"].unique()))}
    sc = data.groupby(["subject_id", *list(keys)])[update_col].agg(["sum", "count"])
    S = np.zeros((len(subj_pos), len(cell_pos)))
    C = np.zeros((len(subj_pos), len(cell_pos)))
    for row_key, r in sc.iterrows():
        cell = tuple(row_key[1:])
        if cell not in cell_pos:
            continue
        S[subj_pos[row_key[0]], cell_pos[cell]] = r["sum"]
        C[subj_pos[row_key[0]], cell_pos[cell]] = r["count"]
    return S, C


def condition_gradients(
    data, preds_by_model, study, update_col, delta_col, dv_name, *, n_boot, rng
):
    """Human and model-predicted gradients across the focal condition, per action.

    For each observed action, the linear-trend contrast of the belief update
    across the focal condition's ordered levels, pooled over scenarios (and over
    any other given condition). Positive means the update rises from the lowest
    to the highest level -- formal to intimate, or low to high desire.

    The human side carries a participant-bootstrap CI, resampling the same unit
    the preregistered primary does. It does not depend on the model, so it is
    computed ONCE here and reused for every variant: seeding it per variant
    published four different intervals for one identical human gradient
    (Study 1a action 0 came out as [-0.1227,-0.0649], [-0.1203,-0.0659],
    [-0.1220,-0.0672] and [-0.1215,-0.0679] for the same -0.0937), leaving which
    one to quote arbitrary.

    `preds_by_model` maps variant name -> that variant's predictions. See the
    module docstring for what `human_minus_model_ci_95` does and does not
    license.
    """
    keys = study.cell_keys
    focal = focal_condition(study)
    levels = ordered_levels(data, focal, study.slug)
    if len(levels) < 2:
        return []
    coef = trend_coefficients(len(levels))
    # Endpoint difference reported beside the trend: the two come apart when the
    # effect is not linear in the ordered levels, which it need not be -- the
    # fitted gamma bends the intimacy response by construction. The trend is the
    # efficient test and stays the headline; the endpoint is the directly
    # readable number, and charges the model nothing for the intermediate shape.
    endpoint = np.zeros(len(levels))
    endpoint[0], endpoint[-1] = -1.0, 1.0

    actions = sorted(data["action"].unique())
    act_pos = {a: i for i, a in enumerate(actions)}
    lvl_pos = {lv: i for i, lv in enumerate(levels)}
    n_groups = len(actions) * len(levels)

    def group_of(frame):
        return (
            frame["action"].map(act_pos).to_numpy() * len(levels)
            + frame[focal].map(lvl_pos).to_numpy()
        )

    # `count`, not `size`: the human means exclude NaN updates, so weighting the
    # model by raw row counts would aggregate the scenarios differently.
    cell_n = data.groupby(keys, as_index=False)[update_col].count()
    cell_n = cell_n.rename(columns={update_col: "n"})

    subjects = np.sort(data["subject_id"].unique())
    subj_pos = {s: i for i, s in enumerate(subjects)}
    sc = (
        data.groupby(["subject_id", *keys])[update_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    grp = group_of(sc)
    srow = sc["subject_id"].map(subj_pos).to_numpy()
    S = np.zeros((len(subjects), n_groups))
    C = np.zeros((len(subjects), n_groups))
    np.add.at(S, (srow, grp), sc["sum"].to_numpy(dtype=float))
    np.add.at(C, (srow, grp), sc["count"].to_numpy(dtype=float))

    def contrast_of(group_means, weights):
        return group_means.reshape(len(actions), len(levels)) @ weights

    def human_contrast(sums, counts, weights):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(counts > 0, sums / counts, np.nan)
        return contrast_of(m, weights)

    if (C.sum(0) == 0).any():
        raise RuntimeError(
            f"{study.slug}: an (action x {focal}) group is empty in the human "
            f"data, so the gradient is not estimable on this design."
        )

    human = human_contrast(S.sum(0), C.sum(0), coef)
    human_end = human_contrast(S.sum(0), C.sum(0), endpoint)
    idx = rng.integers(0, len(subjects), size=(n_boot, len(subjects)))
    boots = np.stack([human_contrast(S[i].sum(0), C[i].sum(0), coef) for i in idx])
    if not np.isfinite(boots).all():
        # A resample that leaves an (action x level) group with no trials at all.
        # np.percentile would propagate the NaN into a silently NaN interval.
        raise RuntimeError(
            f"{study.slug}: a bootstrap resample emptied an (action x {focal}) "
            f"group; too few participants for a stable gradient interval."
        )
    human_ci = [np.percentile(boots[:, a], [2.5, 97.5]) for a in range(len(actions))]

    rows = []
    for model, preds in preds_by_model.items():
        mp = preds[list(keys) + [delta_col]]
        if mp.duplicated(list(keys)).any():
            raise RuntimeError(
                f"{study.slug}/{model}: more than one prediction row per cell. "
                f"`preds_by_model` must hold ONE variant's rows per entry."
            )
        merged = cell_n.merge(mp, on=list(keys), how="left")
        if merged[delta_col].isna().any():
            # Same failure and same severity as _secondary_correlation's: a human
            # cell with no model prediction means stale CV outputs or a condition
            # label mismatch. Silently inner-joining it away would leave the human
            # contrast pooling cells the model contrast does not.
            missing = merged.loc[merged[delta_col].isna(), list(keys)]
            raise RuntimeError(
                f"{len(missing)} human cell(s) in {study.slug} have no matching "
                f"{model} prediction in cv_preds_summary.json. First offenders:\n"
                f"{missing.head(5)}\nStale CV outputs or a condition-label "
                f"mismatch; re-run `make cv-{study.slug}`."
            )
        mgrp = group_of(merged)
        w = merged["n"].to_numpy(dtype=float)
        num = np.bincount(
            mgrp, weights=merged[delta_col].to_numpy() * w, minlength=n_groups
        )
        den = np.bincount(mgrp, weights=w, minlength=n_groups)
        m_mean = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        if not np.isfinite(m_mean).all():
            raise RuntimeError(
                f"{study.slug}/{model}: an (action x {focal}) group has no "
                f"predictions; re-run `make cv-{study.slug}`."
            )
        model_grad = contrast_of(m_mean, coef)
        model_end = contrast_of(m_mean, endpoint)

        for a, action in enumerate(actions):
            lo, hi = human_ci[a]
            d_lo, d_hi = np.percentile(boots[:, a] - model_grad[a], [2.5, 97.5])
            rows.append(
                {
                    "model": model,
                    "dv": dv_name,
                    "focal_condition": focal,
                    "n_levels": len(levels),
                    "action": int(action),
                    "human_gradient": float(human[a]),
                    "human_ci_95": [float(lo), float(hi)],
                    "model_gradient": float(model_grad[a]),
                    "human_endpoint_gradient": float(human_end[a]),
                    "model_endpoint_gradient": float(model_end[a]),
                    "human_minus_model": float(human[a] - model_grad[a]),
                    "human_minus_model_ci_95": [float(d_lo), float(d_hi)],
                    # Only where the human gradient is reliably nonzero. A ratio
                    # to a null denominator is noise, and printed as a percentage
                    # it reads as a finding -- one real cell produced
                    # "-2129% recovered" against a human gradient of -0.002.
                    "recovered_fraction": (
                        float(model_grad[a] / human[a]) if lo * hi > 0 else None
                    ),
                }
            )
    return rows
