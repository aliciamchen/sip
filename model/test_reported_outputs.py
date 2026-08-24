#!/usr/bin/env python3
"""Independent audit of the committed artifacts behind the manuscript.

The numerical oracle in this file deliberately does not import the production
likelihood or model-comparison helpers. It starts from the public participant
CSVs, fold parameters, and per-run held-out predictions, then reconstructs the
trial likelihoods and headline statistics with NumPy/Pandas only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from study_registry import STUDIES
from utils import get_project_root


ROOT = get_project_root()
OUTPUTS = ROOT / "model" / "outputs"
ACTION_TO_INDEX = {"no_share": 0, "low_risk_share": 1, "high_risk_share": 2}
CONSTANT_TOL = 1e-6
CV_FILES = ("cv_preds_summary.json", "cv_folds.jsonl", "cv_trial_ll.jsonl")
FIT_FILES = ("fit_results.json", "fit_restarts.jsonl")


@dataclass(frozen=True)
class Spec:
    keys: tuple[str, ...]
    conditions: tuple[tuple[str, str], ...]
    dvs: tuple[tuple[str, str, str], ...]  # raw rating, update, predicted delta


SPECS = {
    "food_inv_desire": Spec(
        ("scenario_label", "action", "effort_condition", "intimacy_condition"),
        (("effort", "effort_condition"), ("intimacy", "intimacy_condition")),
        (("response", "desire_update", "delta_desire"),),
    ),
    "food_inv_joint_de": Spec(
        ("scenario_label", "action", "intimacy_condition"),
        (("intimacy", "intimacy_condition"),),
        (
            ("desire_rating", "desire_update", "delta_desire"),
            ("effort_rating", "effort_update", "delta_effort"),
        ),
    ),
    "food_inv_intimacy": Spec(
        ("scenario_label", "action", "desire_condition", "effort_condition"),
        (("desire", "desire_condition"), ("effort", "effort_condition")),
        (("intimacy_rating", "intimacy_update", "delta_intimacy"),),
    ),
    "food_inv_joint_ie": Spec(
        ("scenario_label", "action", "desire_condition"),
        (("desire", "desire_condition"),),
        (
            ("intimacy_rating", "intimacy_update", "delta_intimacy"),
            ("effort_rating", "effort_update", "delta_effort"),
        ),
    ),
    "nonfood_inv_joint_de": Spec(
        ("scenario_label", "action", "intimacy_condition"),
        (("intimacy", "intimacy_condition"),),
        (
            ("desire_rating", "desire_update", "delta_desire"),
            ("effort_rating", "effort_update", "delta_effort"),
        ),
    ),
    "nonfood_inv_joint_ie": Spec(
        ("scenario_label", "action", "desire_condition"),
        (("desire", "desire_condition"),),
        (
            ("intimacy_rating", "intimacy_update", "delta_intimacy"),
            ("effort_rating", "effort_update", "delta_effort"),
        ),
    ),
}

GROUPS = {
    "1": ("food_inv_desire", "food_inv_joint_de"),
    "2": ("food_inv_intimacy", "food_inv_joint_ie"),
    "3": ("nonfood_inv_joint_de", "nonfood_inv_joint_ie"),
}


def _read_json(path):
    return json.loads(Path(path).read_text())


def _read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def _assert_close(actual, expected, *, atol=2e-6, detail=""):
    assert math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=atol), (
        f"{detail}: {actual} != {expected} (atol={atol})"
    )


def _trial_updates(slug):
    """One row per participant/scenario, constructed without model loaders."""
    spec = SPECS[slug]
    data = pd.read_csv(ROOT / "data" / slug / "main_trials_long.csv")
    key = ["subject_id", "scenario_label"]
    assert not data.duplicated([*key, "stage"]).any()
    prior = data[data["stage"] == "prior"].set_index(key).sort_index()
    post = data[data["stage"] == "posterior"].set_index(key).sort_index()
    assert prior.index.equals(post.index)
    metadata = ["action_condition", *[source for source, _target in spec.conditions]]
    assert prior[metadata].equals(post[metadata])

    trials = post[metadata].reset_index()
    trials["action"] = trials["action_condition"].map(ACTION_TO_INDEX)
    assert trials["action"].notna().all()
    for source, target in spec.conditions:
        trials[target] = trials[source].replace({0: "low", 1: "high"})
    for raw, update, _delta in spec.dvs:
        trials[update] = post[raw].to_numpy() - prior[raw].to_numpy()
    return trials[["subject_id", *spec.keys, *[d[1] for d in spec.dvs]]]


def _logmeanexp(values):
    values = np.asarray(values, dtype=float)
    peak = np.max(values)
    return float(peak + np.log(np.exp(values - peak).mean()))


def _mixture_logpdf(observed, deltas, sigma):
    observed = np.asarray(observed, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    sigma = max(float(sigma), 1e-6)
    dimension = observed.size
    squared = np.square((deltas - observed) / sigma).sum(axis=1)
    components = -0.5 * (
        dimension * math.log(2 * math.pi) + 2 * dimension * math.log(sigma) + squared
    )
    return _logmeanexp(components)


def _predictions(slug, tag=None):
    directory = OUTPUTS / slug if tag is None else OUTPUTS / slug / "alt" / tag
    return pd.DataFrame(_read_json(directory / "cv_preds_summary.json"))


def _folds(slug, tag=None):
    directory = OUTPUTS / slug if tag is None else OUTPUTS / slug / "alt" / tag
    return pd.DataFrame(_read_jsonl(directory / "cv_folds.jsonl"))


def _trials(slug, tag=None):
    directory = OUTPUTS / slug if tag is None else OUTPUTS / slug / "alt" / tag
    return pd.DataFrame(_read_jsonl(directory / "cv_trial_ll.jsonl"))


def _oracle_trial_ll(slug, tag=None):
    """Reconstruct every stored held-out LL from deltas and fold sigma."""
    spec = SPECS[slug]
    human = _trial_updates(slug)
    preds = _predictions(slug, tag)
    folds = _folds(slug, tag)
    stored = _trials(slug, tag)
    result = []

    assert not preds.duplicated(["model", *spec.keys]).any()
    assert not folds.duplicated(["variant", "held_out_scenario"]).any()
    for model in sorted(preds["model"].unique()):
        model_preds = preds[preds["model"] == model]
        merged = human.merge(model_preds, on=list(spec.keys), validate="many_to_one")
        assert len(merged) == len(human), f"{slug}/{model}: incomplete prediction join"
        sigma = folds[folds["variant"] == model].set_index("held_out_scenario")[
            "param_sigma"
        ]
        assert set(merged["scenario_label"]) == set(sigma.index)
        for row in merged.itertuples(index=False):
            observed = [getattr(row, update) for _raw, update, _delta in spec.dvs]
            run_arrays = [
                np.asarray(getattr(row, f"{delta}_runs"), dtype=float)
                for _raw, _update, delta in spec.dvs
            ]
            n_runs = {len(values) for values in run_arrays}
            assert n_runs == {20}, f"{slug}/{model}: per-run prediction count drift"
            for _raw, _update, delta in spec.dvs:
                _assert_close(
                    getattr(row, delta),
                    np.mean(getattr(row, f"{delta}_runs")),
                    atol=2e-7,
                    detail=f"{slug}/{model}/{delta} mean",
                )
            deltas = np.column_stack(run_arrays)
            result.append(
                {
                    "model": model,
                    "subject_id": row.subject_id,
                    "scenario_label": row.scenario_label,
                    "held_out_ll": _mixture_logpdf(
                        observed, deltas, sigma.loc[row.scenario_label]
                    ),
                }
            )
    oracle = pd.DataFrame(result)
    joined = oracle.merge(
        stored,
        on=["model", "subject_id", "scenario_label"],
        suffixes=("_oracle", "_stored"),
        validate="one_to_one",
    )
    assert len(joined) == len(oracle) == len(stored)
    max_error = np.max(
        np.abs(joined["held_out_ll_oracle"] - joined["held_out_ll_stored"])
    )
    assert max_error < 2e-5, f"{slug}/{tag or 'reported'}: LL error {max_error}"
    return oracle


def _bootstrap_mean(values, subject_ids, n_boot, rng):
    frame = pd.DataFrame({"subject_id": subject_ids, "value": values})
    per_subject = frame.groupby("subject_id")["value"].agg(["sum", "count"])
    sums = per_subject["sum"].to_numpy()
    counts = per_subject["count"].to_numpy()
    draw = rng.integers(0, len(per_subject), size=(n_boot, len(per_subject)))
    return sums[draw].sum(axis=1) / counts[draw].sum(axis=1)


def _primary_oracle(trial_df, n_boot, seed):
    wide = trial_df.pivot(
        index=["subject_id", "scenario_label"], columns="model", values="held_out_ll"
    )
    assert not wide.isna().any().any()
    rng = np.random.default_rng(seed)
    subject_ids = wide.index.get_level_values("subject_id").to_numpy()
    out = {}
    for ablation in [column for column in wide.columns if column != "full"]:
        diff = (wide["full"] - wide[ablation]).to_numpy()
        boot = _bootstrap_mean(diff, subject_ids, n_boot, rng)
        out[f"full_minus_{ablation}"] = (
            float(diff.mean()),
            [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        )
    return out


def _pair_corr(x, y, seed_key, n_boot=1000):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x) < CONSTANT_TOL or np.std(y) < CONSTANT_TOL:
        return math.nan, [math.nan, math.nan]
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    estimate = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(_seed(seed_key))
    draws = []
    for _ in range(n_boot):
        index = rng.integers(0, len(x), len(x))
        if np.std(x[index]) > CONSTANT_TOL and np.std(y[index]) > CONSTANT_TOL:
            draws.append(np.corrcoef(x[index], y[index])[0, 1])
    draws = np.asarray(draws)
    interval = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    return estimate, interval


def _human_cell_means(slug, keys, update):
    return _trial_updates(slug).groupby(list(keys), as_index=False)[update].mean()


def _condition_points(slug, preds, model="full"):
    spec = SPECS[slug]
    keys = tuple(key for key in spec.keys if key != "scenario_label")
    xs, ys = [], []
    selected = preds[preds["model"] == model]
    for _raw, update, delta in spec.dvs:
        human = _human_cell_means(slug, keys, update)
        model_cells = selected.groupby(list(keys), as_index=False)[delta].mean()
        merged = human.merge(model_cells, on=list(keys), validate="one_to_one")
        assert len(merged) == len(human)
        xs.append(merged[delta].to_numpy())
        ys.append(merged[update].to_numpy())
    return np.concatenate(xs), np.concatenate(ys)


def _subject_cell_matrices(data, keys, update, cells):
    positions = {
        tuple(row): i for i, row in enumerate(cells.itertuples(index=False, name=None))
    }
    subjects = {value: i for i, value in enumerate(sorted(data["subject_id"].unique()))}
    sums = np.zeros((len(subjects), len(positions)))
    counts = np.zeros_like(sums)
    grouped = data.groupby(["subject_id", *keys])[update].agg(["sum", "count"])
    for index, row in grouped.iterrows():
        subject, cell = index[0], tuple(index[1:])
        if cell in positions:
            sums[subjects[subject], positions[cell]] = row["sum"]
            counts[subjects[subject], positions[cell]] = row["count"]
    return sums, counts


def _split_half(blocks, rng, n_splits=400):
    z_values = []
    for _ in range(n_splits):
        halves = ([], [])
        for matrices in blocks:
            n_subjects = matrices[0][0].shape[0]
            first = rng.permutation(n_subjects) < n_subjects // 2
            for sums, counts in matrices:
                for selected, target in ((first, halves[0]), (~first, halves[1])):
                    numerator = sums[selected].sum(axis=0)
                    denominator = counts[selected].sum(axis=0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        target.append(
                            np.where(denominator > 0, numerator / denominator, np.nan)
                        )
        a, b = np.concatenate(halves[0]), np.concatenate(halves[1])
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() > 2 and np.std(a[valid]) > 1e-12 and np.std(b[valid]) > 1e-12:
            r = np.corrcoef(a[valid], b[valid])[0, 1]
            z_values.append(np.arctanh(np.clip(r, -0.9999, 0.9999)))
    split = float(np.tanh(np.mean(z_values)))
    reliability = 2 * split / (1 + split)
    return split, float(reliability), float(np.sqrt(max(reliability, 0)))


def _verify_manifest(directory, kind):
    names = FIT_FILES if kind == "fit" else CV_FILES
    path = Path(directory) / f"{kind}_manifest.json"
    assert path.exists(), f"missing required manifest: {path}"
    manifest = _read_json(path)
    assert len(manifest.get("git_sha", "")) == 40
    int(manifest["git_sha"], 16)
    for name in names:
        target = Path(directory) / name
        assert target.exists(), f"manifest target missing: {target}"
        assert manifest["outputs"][name] == _sha256(target), f"stale {target}"
    data = ROOT / manifest["input_data"]["path"]
    assert manifest["input_data"]["sha256"] == _sha256(data)
    for relative, digest in manifest.get("lm_tables", {}).items():
        target = ROOT / relative
        if digest is None:
            assert not target.exists(), (
                f"manifest says absent but file exists: {target}"
            )
        else:
            assert digest == _sha256(target)


def test_all_reported_and_preregistered_manifests_are_mandatory_and_current():
    for slug in STUDIES:
        root = OUTPUTS / slug
        _verify_manifest(root, "fit")
        _verify_manifest(root, "cv")
        prereg = root / "alt" / "uniform-noreweight"
        _verify_manifest(prereg, "fit")
        _verify_manifest(prereg, "cv")
        _verify_manifest(root / "alt" / "pooled-all", "cv")
    for slug in ("nonfood_inv_joint_de", "nonfood_inv_joint_ie"):
        _verify_manifest(OUTPUTS / slug / "alt" / "transfer-pooled-food-refit", "cv")


def test_independent_oracle_reconstructs_every_reported_trial_likelihood():
    for slug in STUDIES:
        oracle = _oracle_trial_ll(slug)
        comparison = _read_json(OUTPUTS / slug / "cv_model_comparison.json")
        means = oracle.groupby("model")["held_out_ll"].mean().to_dict()
        assert set(means) == set(comparison["mean_held_out_ll_per_trial"])
        for model, expected in comparison["mean_held_out_ll_per_trial"].items():
            _assert_close(
                means[model], expected, atol=2e-6, detail=f"{slug}/{model} mean LL"
            )

        primary = _primary_oracle(oracle, comparison["n_boot"], comparison["seed"])
        for row in comparison["primary"]:
            estimate, interval = primary[row["comparison"]]
            _assert_close(estimate, row["mean_per_trial_ll_diff"], atol=2e-6)
            np.testing.assert_allclose(interval, row["ci_95"], rtol=0, atol=2e-6)


def test_independent_oracle_reconstructs_preregistered_trial_likelihoods():
    for slug in STUDIES:
        oracle = _oracle_trial_ll(slug, "uniform-noreweight")
        stored = _trials(slug, "uniform-noreweight")
        assert len(oracle) == len(stored)


def test_loso_folds_cover_each_scenario_once_and_match_the_trial_outputs():
    for slug in STUDIES:
        human = _trial_updates(slug)
        scenarios = sorted(human["scenario_label"].unique())
        for tag in (None, "uniform-noreweight", "pooled-all"):
            folds = _folds(slug, tag)
            trials = _trials(slug, tag)
            variants = sorted(folds["variant"].unique())
            assert len(folds) == 16 * len(variants)
            for variant in variants:
                selected = folds[folds["variant"] == variant]
                assert set(selected["fold"]) == set(range(16))
                assert sorted(selected["held_out_scenario"]) == scenarios
                assert (selected["n_train"] + selected["n_test"] == len(human)).all()
                assert selected["n_test"].sum() == len(human)
                model_trials = trials[trials["model"] == variant]
                assert len(model_trials) == len(human)
                for row in selected.itertuples(index=False):
                    fold_ll = model_trials[
                        model_trials["scenario_label"] == row.held_out_scenario
                    ]["held_out_ll"].sum()
                    _assert_close(-fold_ll, row.test_nll, atol=2e-4)


def test_generalization_arms_freeze_exactly_the_declared_utility():
    food_fit = _read_json(OUTPUTS / "pooled" / "food" / "pooled_fit.json")
    food_names = food_fit["param_names"][:4]
    food_utility = dict(zip(food_names, food_fit["full_data_params"][:4]))
    for slug in ("nonfood_inv_joint_de", "nonfood_inv_joint_ie"):
        directory = OUTPUTS / slug / "alt" / "transfer-pooled-food-refit"
        provenance = _read_json(directory / "transfer_provenance.json")
        assert provenance["donor_utility"] == food_utility
        assert set(provenance["free_params"]) == {"alpha_observer", "sigma", "eta"}
        for fold in _read_jsonl(directory / "cv_folds.jsonl"):
            for name, expected in food_utility.items():
                _assert_close(fold[f"param_{name}"], expected, atol=1e-12)

    pooled = _read_json(OUTPUTS / "pooled" / "all" / "pooled_fit.json")
    for slug in STUDIES:
        folds = _read_jsonl(OUTPUTS / slug / "alt" / "pooled-all" / "cv_folds.jsonl")
        for row in folds:
            expected = pooled["per_fold_params"][str(row["fold"])][:4]
            for name, value in zip(pooled["param_names"][:4], expected):
                _assert_close(row[f"param_{name}"], value, atol=1e-12)


def test_independent_correlations_and_noise_ceilings_match_artifacts():
    for slug, spec in SPECS.items():
        preds = _predictions(slug)
        comparison = _read_json(OUTPUTS / slug / "cv_model_comparison.json")
        stored = {
            (row["model"], row["dv"]): row
            for row in comparison["secondary_correlations"]
        }
        for model in sorted(preds["model"].unique()):
            selected = preds[preds["model"] == model]
            for _raw, update, delta in spec.dvs:
                human = _human_cell_means(slug, spec.keys, update)
                merged = human.merge(selected[[*spec.keys, delta]], on=list(spec.keys))
                estimate, interval = _pair_corr(
                    merged[delta], merged[update], f"{slug}|{model}|{delta}|pair_ci"
                )
                row = stored[(model, delta.removeprefix("delta_"))]
                if math.isnan(estimate):
                    assert pd.isna(row["r"]) and all(
                        pd.isna(bound) for bound in row["ci_95"]
                    ), (
                        f"{slug}/{model}/{delta}: oracle is constant but artifact is {row}"
                    )
                else:
                    _assert_close(estimate, row["r"], atol=1e-12)
                    np.testing.assert_allclose(
                        interval, row["ci_95"], rtol=0, atol=1e-12
                    )

        full = preds[preds["model"] == "full"]
        rng = np.random.default_rng(comparison["seed"] + 1)
        ceilings = {row["dv"]: row for row in comparison["noise_ceilings"]}
        data = _trial_updates(slug)
        for _raw, update, delta in spec.dvs:
            cells = data.groupby(list(spec.keys), as_index=False)[update].mean()
            cells = cells.merge(
                full[list(spec.keys)].drop_duplicates(), on=list(spec.keys)
            )
            matrices = _subject_cell_matrices(
                data, list(spec.keys), update, cells[list(spec.keys)]
            )
            split, reliability, ceiling = _split_half([[matrices]], rng)
            row = ceilings[delta.removeprefix("delta_")]
            _assert_close(split, row["split_half"], atol=1e-12)
            _assert_close(reliability, row["reliability"], atol=1e-12)
            _assert_close(ceiling, row["ceiling"], atol=1e-12)


def test_group_and_generalization_correlations_reconstruct_independently():
    groups = {
        entry["study"]: entry
        for entry in _read_json(OUTPUTS / "group_correlations.json")
    }
    for number, slugs in GROUPS.items():
        entry = groups[number]
        for model in ("base", "discomfort_only", "full"):
            xs, ys = [], []
            for slug in slugs:
                preds = _predictions(slug)
                if model == "base" and slug in {
                    "food_inv_desire",
                    "food_inv_joint_de",
                    "nonfood_inv_joint_de",
                }:
                    selected_model = "base_shared"
                else:
                    selected_model = model
                x, y = _condition_points(slug, preds, selected_model)
                xs.append(x)
                ys.append(y)
            estimate, interval = _pair_corr(
                np.concatenate(xs),
                np.concatenate(ys),
                f"group|0|{number}|{model}|pair_ci",
            )
            row = next(item for item in entry["correlations"] if item["model"] == model)
            if math.isnan(estimate):
                assert pd.isna(row["r"])
                assert all(pd.isna(bound) for bound in row["ci_95"])
            else:
                _assert_close(estimate, row["r"], atol=1e-12)
                np.testing.assert_allclose(interval, row["ci_95"], rtol=0, atol=1e-12)

    artifact = _read_json(OUTPUTS / "generalization_primary.json")
    for row in artifact["per_experiment"]:
        tag = {
            "own": None,
            "food": "transfer-pooled-food-refit",
            "pooled": "pooled-all",
        }[row["arm"]]
        x, y = _condition_points(row["slug"], _predictions(row["slug"], tag))
        estimate, interval = _pair_corr(x, y, f"gen|{row['slug']}|{row['arm']}|pair_ci")
        _assert_close(estimate, row["r"], atol=1e-12)
        np.testing.assert_allclose(interval, row["ci_95"], rtol=0, atol=1e-12)

    for row in artifact["combined"]:
        xs, ys = [], []
        tag = {
            "own": None,
            "food": "transfer-pooled-food-refit",
            "pooled": "pooled-all",
        }[row["arm"]]
        for slug in row["slugs"]:
            x, y = _condition_points(slug, _predictions(slug, tag))
            xs.append(x)
            ys.append(y)
        estimate, interval = _pair_corr(
            np.concatenate(xs),
            np.concatenate(ys),
            f"gen|nonfood|{row['arm']}|pair_ci",
        )
        _assert_close(estimate, row["r"], atol=1e-12)
        np.testing.assert_allclose(interval, row["ci_95"], rtol=0, atol=1e-12)


def test_structural_null_predictions_are_numerically_zero():
    desire_slugs = ("food_inv_desire", "food_inv_joint_de", "nonfood_inv_joint_de")
    intimacy_slugs = ("food_inv_intimacy", "food_inv_joint_ie", "nonfood_inv_joint_ie")
    for slug in desire_slugs:
        preds = _predictions(slug)
        null = preds[preds["model"] == "discomfort_only"]
        for delta in ("delta_desire", "delta_effort"):
            if delta in null:
                assert null[delta].abs().max() < CONSTANT_TOL
    for slug in intimacy_slugs:
        preds = _predictions(slug)
        vanilla = preds[preds["model"] == "base"]
        assert vanilla["delta_intimacy"].abs().max() < CONSTANT_TOL
        discomfort = preds[preds["model"] == "discomfort_only"]
        if "delta_effort" in discomfort:
            assert discomfort["delta_effort"].abs().max() < CONSTANT_TOL


def test_qualitative_human_action_signatures_match_the_results_claims():
    for slug in ("food_inv_desire", "food_inv_joint_de", "nonfood_inv_joint_de"):
        data = _trial_updates(slug)
        by_action = data.groupby("action")["desire_update"].mean()
        assert by_action[1] > by_action[0] and by_action[2] > by_action[0]
        risky = (
            data[data["action"] == 2]
            .groupby("intimacy_condition")["desire_update"]
            .mean()
        )
        assert risky["max_formal"] > risky["max_intimate"]
        no_share = (
            data[data["action"] == 0]
            .groupby("intimacy_condition")["desire_update"]
            .mean()
        )
        assert no_share["max_intimate"] < no_share["max_formal"]
        if "effort_update" in data:
            effort = data.groupby("action")["effort_update"].mean()
            assert effort[1] < effort[0] and effort[1] < effort[2]

    for slug in ("food_inv_intimacy", "food_inv_joint_ie", "nonfood_inv_joint_ie"):
        data = _trial_updates(slug)
        by_action = data.groupby("action")["intimacy_update"].mean()
        assert by_action[2] > by_action[1] > by_action[0]
        risky = (
            data[data["action"] == 2]
            .groupby("desire_condition")["intimacy_update"]
            .mean()
        )
        assert risky["low"] > risky["high"]
        if "effort_update" in data:
            effort = data.groupby("action")["effort_update"].mean()
            assert effort[1] < effort[0] and effort[1] < effort[2]


def _normal_cdf(value):
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def test_full_model_predictive_distributions_are_calibrated_and_nearly_bounded():
    """Audit the response-noise scale on the observable update scale.

    The fitted likelihood is Gaussian and therefore technically unbounded even
    though rating updates lie in [-1, 1]. The committed fits should put little
    mass outside that range, and their marginal probability-integral transforms
    should be centered with approximately nominal 95% coverage. These are broad
    regression guards, not formal goodness-of-fit hypothesis tests.
    """
    for slug, spec in SPECS.items():
        human = _trial_updates(slug)
        predictions = _predictions(slug)
        predictions = predictions[predictions["model"] == "full"]
        merged = human.merge(predictions, on=list(spec.keys), validate="many_to_one")
        sigma = _folds(slug)
        sigma = sigma[sigma["variant"] == "full"].set_index("held_out_scenario")[
            "param_sigma"
        ]
        pits = []
        outside_mass = []
        for row in merged.itertuples(index=False):
            scale = float(sigma.loc[row.scenario_label])
            for _raw, update, delta in spec.dvs:
                observed = float(getattr(row, update))
                components = np.asarray(getattr(row, f"{delta}_runs"), dtype=float)
                pits.append(
                    np.mean(
                        [
                            _normal_cdf((observed - center) / scale)
                            for center in components
                        ]
                    )
                )
                outside_mass.extend(
                    _normal_cdf((-1 - center) / scale)
                    + 1
                    - _normal_cdf((1 - center) / scale)
                    for center in components
                )
        pits = np.asarray(pits)
        coverage = np.mean((pits >= 0.025) & (pits <= 0.975))
        assert 0.47 <= pits.mean() <= 0.53, f"{slug}: mean PIT {pits.mean():.3f}"
        assert 0.90 <= coverage <= 0.97, f"{slug}: 95% coverage {coverage:.3f}"
        assert np.mean(outside_mass) < 0.01, slug
        assert np.max(outside_mass) < 0.06, slug


def _assert_interval_brackets(point, interval, detail):
    if pd.isna(point):
        assert all(pd.isna(bound) for bound in interval), detail
        return
    assert not any(pd.isna(bound) for bound in interval), detail
    assert interval[0] <= point <= interval[1], f"{detail}: {point} outside {interval}"


def test_every_reported_interval_brackets_its_point_estimate():
    for slug in STUDIES:
        comparison = _read_json(OUTPUTS / slug / "cv_model_comparison.json")
        for row in comparison["primary"]:
            _assert_interval_brackets(
                row["mean_per_trial_ll_diff"], row["ci_95"], row["comparison"]
            )
        for row in comparison["secondary_correlations"]:
            _assert_interval_brackets(
                row["r"], row["ci_95"], f"{slug}/{row['model']}/{row['dv']}"
            )
        config = _read_json(
            OUTPUTS / slug / "alt" / "compare_uniform-noreweight_vs_reported.json"
        )
        for row in config["per_variant"]:
            _assert_interval_brackets(
                row["mean_per_trial_ll_diff"], row["ci_95"], f"{slug}/{row['variant']}"
            )
    for path, rows_key in (
        (OUTPUTS / "pooled" / "pooled_summary.json", ("rows", "combined")),
        (OUTPUTS / "transfer" / "transfer_summary.json", ("pooled_donor",)),
    ):
        artifact = _read_json(path)
        for key in rows_key:
            for row in artifact[key]:
                _assert_interval_brackets(
                    row["diff"], row["ci_95"], f"{path.name}/{key}"
                )


def main():
    tests = [
        test_all_reported_and_preregistered_manifests_are_mandatory_and_current,
        test_independent_oracle_reconstructs_every_reported_trial_likelihood,
        test_independent_oracle_reconstructs_preregistered_trial_likelihoods,
        test_loso_folds_cover_each_scenario_once_and_match_the_trial_outputs,
        test_generalization_arms_freeze_exactly_the_declared_utility,
        test_independent_correlations_and_noise_ceilings_match_artifacts,
        test_group_and_generalization_correlations_reconstruct_independently,
        test_structural_null_predictions_are_numerically_zero,
        test_qualitative_human_action_signatures_match_the_results_claims,
        test_full_model_predictive_distributions_are_calibrated_and_nearly_bounded,
        test_every_reported_interval_brackets_its_point_estimate,
    ]
    print("=" * 72)
    print("Reported-results audit tests")
    print("=" * 72)
    for test in tests:
        test()
        print(f"✓ {test.__name__}")
    print("=" * 72)
    print(f"All {len(tests)} reported-results audit tests passed!")
    print("=" * 72)


if __name__ == "__main__":
    main()
