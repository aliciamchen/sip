"""Cross-study parameter transfer: does one study's fitted utility explain another?

Every reported fit estimates its study's parameters from that study's own data,
so six studies give six parameter sets and nothing yet says whether the utility
is one psychological object or six flexible fits. This script answers that
directly: take a DONOR study's fitted utility weights, put them in a RECIPIENT
study's model, and score the recipient's data out of sample.

Two arms, run on the same designed pairs:

  frozen   Zero free parameters. The donor's whole vector -- utility weights,
           alpha_observer and sigma -- is used verbatim on the recipient. The
           strictest test, and the one that conflates two things: whether the
           utility transfers, and whether the response layer is even comparable
           across studies (alpha_observer spans 0.84 to 27 across the six, and
           it trades off against the overall weight scale, since larger weights
           and a larger alpha both sharpen the observer's posterior).

  refit    The donor's utility weights are frozen; alpha_observer, sigma and eta
           are re-estimated on the recipient under the reported LOSO protocol.
           This isolates "does the theory transfer" from "is the response scale
           comparable", and is the arm to read for the psychological claim.

Both arms are scored the same way as every reported number -- per-trial held-out
log-likelihood under leave-one-scenario-out CV -- so each is directly comparable
to the recipient's own CV, which is the ceiling: the same model with parameters
estimated on the recipient itself. Held-out likelihood carries no parameter
penalty, so a 0-free-parameter transfer beating a 6-parameter own-fit is
possible and would mean the own-fit overfits.

WHY PAIRS RATHER THAN LEAVE-ONE-STUDY-OUT. With six studies whose fitted gamma
spans 0.32-3.10 and w_d 0.14-4.24, a single leave-one-out number averages over
the disagreement and hides its structure. The designed pairs each isolate one
difference:

  within domain, across inference problem (1a<->2a, 1b<->2b)
      Same stimuli, same domain; the observer infers a different latent. The
      actor's utility should be invariant to what the observer is asked, so this
      is the strongest test of the utility as a stable object.

  across domain, matched design (1b<->3a, 2b<->3b)
      Same observer family and design, different stimulus set. Partial failure
      is expected here and already has a reading (the feature-validity account
      in notes/decisions.md, 2026-08-04).

The utility weights are commensurable across all six by construction: d and I
are in [0, 1] whether inferred over the 101-bin grid or given as an LM scalar,
and risk/effort/g are all elicited 0-6 and normalized to [0, 1]. That is what
makes transferring them meaningful at all.

ETA. The comparison-set reweighting's gain is study-specific by construction --
its scope rule is defined by which questions are contrastive-only in each study
(_reweighting.STUDY_CONTRASTIVE), and 1a gets none at all. It is therefore
treated as recipient-side, not part of the transferred utility: the `refit` arm
re-estimates it, and the `frozen` arm takes the donor's value when the donor has
one and the recipient's scope wants one, and eta = 0 (the preregistered
comparison set) otherwise. Each pair records which rule applied.

EXPLORATORY. This analysis is in no preregistration.

Usage:
    uv run python model/cv/transfer.py                    # every pair, both arms
    uv run python model/cv/transfer.py --pair 1b:3a       # one ordered pair
    uv run python model/cv/transfer.py --arm refit
    uv run python model/cv/transfer.py --summary-only     # re-summarize on disk
"""

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))
sys.path.insert(0, str(_project_root / "model" / "cv"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _fit_dispatcher import fit_context  # noqa: E402
from _helpers import ALPHA_OBS_SEEDS, read_jsonl, write_json  # noqa: E402
from _inverse_dispatcher import (  # noqa: E402
    N_RESTARTS_CV,
    RunOverride,
    _run_loso,
    _write_outputs,
)
from model_comparison import _bootstrap_mean_by_subject  # noqa: E402
from observers import VARIANT_PARAM_NAMES  # noqa: E402
from study_registry import STUDIES, reported_base  # noqa: E402
from utils import get_project_root  # noqa: E402

# The designed donor -> recipient pairs, both directions, keyed by the paper's
# short labels. `kind` names what the pair holds constant and what it varies,
# which is what the result means.
PAIR_KINDS = {
    "inference": "within domain, across inference problem",
    "domain": "across domain, matched design",
}
PAIRS = (
    ("1a", "2a", "inference"),
    ("2a", "1a", "inference"),
    ("1b", "2b", "inference"),
    ("2b", "1b", "inference"),
    ("1b", "3a", "domain"),
    ("3a", "1b", "domain"),
    ("2b", "3b", "domain"),
    ("3b", "2b", "domain"),
)
PAIR_KIND = {(d, r): k for d, r, k in PAIRS}

# Ablation ROLES, not variant names: the paper's "Base" is `base_shared` in the
# given-relationship studies and `base` in the rest (study_registry.reported_base),
# so a role resolves to a possibly different variant on each side of a pair. The
# utility parameter names are the same either way, which is what transfer needs.
ROLES = ("full", "discomfort_only", "base")

ARMS = ("frozen", "refit")

#: The transfer analysis only ever moves the full model's utility; the ablation
#: rows exist to be read as controls, not transferred on their own.
VARIANT_FULL = "full"

SLUG_BY_LABEL = {s.short_label: slug for slug, s in STUDIES.items()}


def _variant_for(slug, role):
    return reported_base(slug) if role == "base" else role


# Which utility term prices each inferred latent. An ablation that lacks the
# term predicts an identically zero belief update for that latent -- its actor's
# policy does not vary with it, so the observer's posterior stays at the prior
# (manuscript: "a model without the utility term a variable needs predicts the
# same update in every condition ... a vertical stripe at x = 0"). Measured at
# |delta| <= 1e-7 in every reported CV run.
_PRICED_BY = {"desire": "w_v", "effort": "w_e", "intimacy": "w_d"}


def null_latents(slug, variant):
    """The study's inferred latents this variant cannot move, as a tuple.

    A transfer delta on such a latent carries no information about the utility:
    the predictions are zero whatever the weights are, so the difference is
    entirely the response layer. Empty tuple means the variant is informative
    on everything the study infers.
    """
    have = set(VARIANT_PARAM_NAMES["base" if variant.startswith("base") else variant])
    return tuple(dv.name for dv in STUDIES[slug].dvs if _PRICED_BY[dv.name] not in have)


def transfer_status(donor, recipient, role):
    """Whether this (pair, role) is a meaningful transfer test, and why not.

    An ablation row only tests transfer when the ablation is informative on BOTH
    sides: a donor that cannot move a latent never estimated the weights that
    move it (they sit wherever the optimizer left them), and a recipient that
    cannot move it would not use them. Either way the delta is response-layer
    only. `full` prices every latent, so it is the one row that is a clean test
    in every pair.
    """
    d_null = null_latents(donor, _variant_for(donor, role))
    r_null = null_latents(recipient, _variant_for(recipient, role))
    if not d_null and not r_null:
        return "test", ""
    if d_null and r_null:
        both = sorted(set(d_null) & set(r_null))
        if both and set(d_null) == set(r_null):
            return "control", f"null on {', '.join(both)} in both studies"
    if d_null:
        return (
            "control",
            f"donor cannot identify these weights ({', '.join(d_null)} null)",
        )
    return "control", f"recipient cannot use them ({', '.join(r_null)} null)"


def _slug(label):
    try:
        return SLUG_BY_LABEL[label]
    except KeyError:
        raise SystemExit(
            f"unknown study label {label!r} — expected one of {sorted(SLUG_BY_LABEL)}"
        ) from None


def _pooled_fit(group):
    """One pooled group's fit record (`model/cv/pooled.py`)."""
    path = (
        get_project_root() / "model" / "outputs" / "pooled" / group / "pooled_fit.json"
    )
    if not path.exists():
        raise SystemExit(
            f"{path} missing — run `make pooled ARGS='--group {group}'` first"
        )
    with open(path) as f:
        return json.load(f)


def _pooled_utility(group):
    """A pooled group's SHARED utility weights, by name.

    The pooled vector names its per-experiment response slots `<slug>:<param>`,
    so the shared block is exactly the names without a colon."""
    d = _pooled_fit(group)
    return {
        n: float(v)
        for n, v in zip(d["param_names"], d["full_data_params"])
        if ":" not in n
    }


def _tag(donor_label, arm):
    """Output tag under outputs/<recipient>/alt/, mirroring RunConfig.tag()."""
    return f"transfer-{donor_label}-{arm}"


def outputs_dir(recipient_slug, donor_label, arm):
    return (
        get_project_root()
        / "model"
        / "outputs"
        / recipient_slug
        / "alt"
        / _tag(donor_label, arm)
    )


def _fit_params(slug, variant):
    """One variant's reported full-data fitted parameters, as a plain dict."""
    path = get_project_root() / "model" / "outputs" / slug / "fit_results.json"
    if not path.exists():
        raise SystemExit(f"{path} missing — run `make fit-{slug}` first")
    with open(path) as f:
        rows = json.load(f)
    for row in rows:
        if row["model"] == variant:
            return row
    raise SystemExit(f"{slug}: no `{variant}` fit in {path}")


def _param_layout(ctx, variant):
    """(utility_param_names, has_eta) for one (study, variant) — the layout of
    the fit's parameter vector, which is [*utility, alpha_observer, sigma, eta?].
    Uniform priors only; an informative-prior config would insert `prior_nu`."""
    if ctx.priors[variant] is not None:
        raise SystemExit(
            "the transfer analysis runs on the reported (uniform-prior) config "
            "only — an informative-prior vector carries an extra `prior_nu` slot"
        )
    _, utility_names = ctx.variants[variant]
    return list(utility_names), ctx.reweighting(variant) is not None


def _eta_policy(donor_row, recipient_has_eta):
    """The eta the `frozen` arm scores with, plus the rule that produced it.

    eta is a per-study gain whose very scope is defined per study, so it is not
    part of the transferred utility. Where the recipient's scope grants one and
    the donor has none (Study 1a has no reweighting at all), eta = 0 falls back
    to the preregistered comparison set rather than borrowing an unrelated gain.
    """
    if not recipient_has_eta:
        return None, "recipient has no reweighting; donor eta (if any) dropped"
    if donor_row.get("param_eta") is not None:
        return float(donor_row["param_eta"]), "donor eta transferred"
    return 0.0, "donor has no reweighting; eta = 0 (preregistered comparison set)"


def _frozen_vector(donor_row, utility_names, eta):
    """The donor's parameters laid out for the recipient's fit vector."""
    vec = [float(donor_row[f"param_{n}"]) for n in utility_names]
    vec += [float(donor_row["alpha_observer"]), float(donor_row["param_sigma"])]
    if eta is not None:
        vec.append(float(eta))
    return np.asarray(vec, dtype=float)


def _free_mask(n_utility, n_params, arm):
    """Which slots each arm estimates on the recipient.

    `frozen` estimates nothing. `refit` frees everything after the utility
    weights — alpha_observer, sigma, and eta where the study has one.
    """
    mask = np.zeros(n_params, dtype=bool)
    if arm == "refit":
        mask[n_utility:] = True
    return mask


def _full_data_refit(ctx, variant, init, mask, label):
    """Full-data fit of the recipient with the utility weights frozen — the warm
    start every fold refit begins from, exactly as the reported protocol
    warm-starts folds from the reported full-data fit.

    alpha_observer's two known basins are covered explicitly (ALPHA_OBS_SEEDS)
    rather than by luck: `_fit_multistart` only adds its basin seeds when it is
    given no init, and a masked fit always has one. Cold lognormal draws are
    centered at 1 and would essentially never reach the sharp basin, which three
    of the six studies sit in.
    """
    obs_fn, utility_names = ctx.variants[variant]
    i_alpha = len(utility_names)
    inits = [np.asarray(init, dtype=float)]
    for seed in ALPHA_OBS_SEEDS:
        seeded = inits[0].copy()
        seeded[i_alpha] = float(seed)
        inits.append(seeded)

    best = None
    for k, start in enumerate(inits):
        params, nll, _ = ctx.fitter(
            observer_fn=obs_fn,
            utility_param_names=utility_names,
            table_kwargs=ctx.table_kwargs[variant],
            priors=ctx.priors[variant],
            reweighting=ctx.reweighting(variant),
            seed_key=f"{label}|init{k}",
            init_params=start,
            free_mask=mask,
            n_restarts=1,
            verbose=False,
            **ctx.data_kwargs,
        )
        if best is None or nll < best[1]:
            best = (np.asarray(params, dtype=float), float(nll))
    return best


def run_from_pooled(donor_group, recipient_label, workers=None):
    """Apply a POOLED group's utility to an experiment outside that group.

    The paper's generalization claim in its most direct form: the utility
    estimated from the food experiments, used unchanged to predict the nonfood
    ones. The donor is `model/cv/pooled.py`'s full-data fit for `donor_group`;
    only that group's four SHARED utility weights cross over, and the
    recipient's alpha_observer / sigma / eta are re-estimated on its own data
    under the ordinary LOSO protocol -- the `refit` arm's split, for the reasons
    it established.

    Cross-validation note: the donor utility is estimated from the donor group's
    data ALONE, and the two stimulus sets are disjoint, so every recipient trial
    is out of sample with respect to it no matter which fold it falls in. Only
    the response layer needs holding out, which the LOSO folds do. That is why
    the donor's FULL-DATA utility is the right one to carry over here (and it is
    also the one the paper would quote), rather than a per-fold vector.
    """
    recipient = _slug(recipient_label)
    if recipient in _pooled_fit(donor_group)["slugs"]:
        raise SystemExit(
            f"{recipient_label} is inside the `{donor_group}` group — its own "
            "data helped fit that utility, so this would not be a transfer"
        )
    weights = _pooled_utility(donor_group)
    ctx = fit_context(recipient)
    out_dir = outputs_dir(recipient, f"pooled-{donor_group}", "refit")

    print("=" * 70)
    print(f"Transfer [refit]: pooled {donor_group} utility -> {recipient_label}")
    print("=" * 70)
    print("  donor utility: " + ", ".join(f"{k}={v:.4f}" for k, v in weights.items()))

    utility_names, has_eta = _param_layout(ctx, VARIANT_FULL)
    missing = [n for n in utility_names if n not in weights]
    if missing:
        raise SystemExit(f"pooled {donor_group} fit has no {missing}")
    # The donor group has no single eta (each of its experiments kept its own),
    # so there is nothing to carry over; the recipient re-estimates its own.
    own = _fit_params(recipient, VARIANT_FULL)
    vec = np.asarray(
        [weights[n] for n in utility_names]
        + [float(own["alpha_observer"]), float(own["param_sigma"])]
        + ([float(own["param_eta"])] if has_eta else []),
        dtype=float,
    )
    mask = _free_mask(len(utility_names), len(vec), "refit")
    vec, nll = _full_data_refit(
        ctx, VARIANT_FULL, vec, mask, f"pooled|{donor_group}|{recipient}"
    )
    names = utility_names + ["alpha_observer", "sigma"] + (["eta"] if has_eta else [])
    print(
        f"  full-data refit -> NLL {nll:.2f}, "
        + ", ".join(
            f"{n}={v:.3f}" for n, v in zip(names, vec) if n not in utility_names
        )
    )

    override = RunOverride(
        variants=(VARIANT_FULL,),
        init_params={VARIANT_FULL: vec},
        free_mask={VARIANT_FULL: mask},
        outputs_dir=out_dir,
        fingerprint={
            "transfer_donor": f"pooled-{donor_group}",
            "transfer_arm": "refit",
            "transfer_start": {VARIANT_FULL: [float(x) for x in vec]},
            "transfer_free": {VARIANT_FULL: mask.tolist()},
        },
    )
    _write_outputs(
        recipient,
        *_run_loso(ctx.family, recipient, workers=workers, override=override),
        outputs_dir=out_dir,
    )
    write_json(
        out_dir / "transfer_provenance.json",
        {
            "donor": f"pooled-{donor_group}",
            "donor_utility": weights,
            "recipient": recipient,
            "recipient_label": recipient_label,
            "arm": "refit",
            "param_names": names,
            "start_vector": [float(x) for x in vec],
            "free_params": [n for n in names if n not in utility_names],
            "n_restarts_cv": N_RESTARTS_CV,
        },
    )
    print(f"  wrote {out_dir}")


def run_pair(donor_label, recipient_label, arm, workers=None):
    """One ordered (donor -> recipient) pair under one arm. Writes the standard
    CV output set to outputs/<recipient>/alt/transfer-<donor>-<arm>/, so
    `model_comparison.py --compare-configs` reads it like any other run config.
    """
    donor, recipient = _slug(donor_label), _slug(recipient_label)
    out_dir = outputs_dir(recipient, donor_label, arm)
    ctx = fit_context(recipient)

    print("=" * 70)
    print(
        f"Transfer [{arm}]: {donor_label} -> {recipient_label} "
        f"({PAIR_KINDS.get(PAIR_KIND.get((donor_label, recipient_label)), 'undesignated pair')})"
    )
    print("=" * 70)

    inits, masks, provenance = {}, {}, []
    for role in ROLES:
        d_variant = _variant_for(donor, role)
        r_variant = _variant_for(recipient, role)
        utility_names, has_eta = _param_layout(ctx, r_variant)
        donor_row = _fit_params(donor, d_variant)
        missing = [n for n in utility_names if donor_row.get(f"param_{n}") is None]
        if missing:
            raise SystemExit(
                f"{donor}/{d_variant} has no {missing} — donor and recipient "
                f"variants must price the same utility terms"
            )
        eta, eta_rule = _eta_policy(donor_row, has_eta)
        vec = _frozen_vector(donor_row, utility_names, eta)
        mask = _free_mask(len(utility_names), len(vec), arm)
        names = (
            utility_names + ["alpha_observer", "sigma"] + (["eta"] if has_eta else [])
        )
        free_names = [n for n, is_free in zip(names, mask) if is_free]

        if arm == "refit":
            vec, nll = _full_data_refit(
                ctx, r_variant, vec, mask, f"transfer|{donor}|{recipient}|{r_variant}"
            )
            print(
                f"  {role}: full-data refit of [{', '.join(free_names)}] "
                f"-> NLL {nll:.2f}, "
                + ", ".join(
                    f"{n}={v:.3f}" for n, v in zip(names, vec) if n in free_names
                )
            )
        inits[r_variant], masks[r_variant] = vec, mask
        provenance.append(
            {
                "role": role,
                "donor_variant": d_variant,
                "recipient_variant": r_variant,
                "param_names": names,
                "transferred_utility": {
                    n: float(donor_row[f"param_{n}"]) for n in utility_names
                },
                "start_vector": [float(x) for x in vec],
                "free_params": free_names,
                "eta_rule": eta_rule,
            }
        )

    override = RunOverride(
        variants=tuple(inits),
        init_params=inits,
        free_mask=masks,
        outputs_dir=out_dir,
        fingerprint={
            "transfer_donor": donor,
            "transfer_arm": arm,
            "transfer_start": {v: [float(x) for x in inits[v]] for v in inits},
            "transfer_free": {v: masks[v].tolist() for v in masks},
        },
    )
    _write_outputs(
        recipient,
        *_run_loso(ctx.family, recipient, workers=workers, override=override),
        outputs_dir=out_dir,
    )
    write_json(
        out_dir / "transfer_provenance.json",
        {
            "donor": donor,
            "donor_label": donor_label,
            "recipient": recipient,
            "recipient_label": recipient_label,
            "arm": arm,
            "n_restarts_cv": N_RESTARTS_CV,
            "variants": provenance,
        },
    )
    print(f"  wrote {out_dir}")


def _trial_ll(path):
    return pd.DataFrame(read_jsonl(path))


def _paired(frame_a, frame_b, suffixes, what):
    """Inner-join two per-trial LL frames on (subject, scenario), refusing a
    partial match — that would mean the two sides were scored on different data
    vintages, and the mean difference would be silently over a subset."""
    wide = frame_a.merge(
        frame_b, on=["subject_id", "scenario_label"], suffixes=suffixes, how="inner"
    )
    if len(wide) != len(frame_a) or len(wide) != len(frame_b):
        raise RuntimeError(
            f"{what}: {len(frame_a)} vs {len(frame_b)} trials, matched "
            f"{len(wide)} — mixed data vintages; re-run CV"
        )
    return wide


def _boot_ci(values, subject_ids, n_boot, rng):
    boots = _bootstrap_mean_by_subject(values, subject_ids, n_boot, rng)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def summarize_pooled_donor(n_boot=1000, seed=0):
    """Runs whose donor is a POOLED group's utility (`run_from_pooled`).

    Reported per recipient AND combined over the recipients, because the paper's
    claim is about the nonfood experiments as a whole: the two can differ in
    sign and cancel, which the per-experiment rows have to show rather than the
    combined number hiding.
    """
    rng = np.random.default_rng(seed)
    root = get_project_root() / "model" / "outputs"
    by_group = {}
    for slug, study in STUDIES.items():
        for d in sorted((root / slug / "alt").glob("transfer-pooled-*-refit")):
            group = d.name[len("transfer-pooled-") : -len("-refit")]
            own_path = root / slug / "cv_trial_ll.jsonl"
            if not (d / "cv_trial_ll.jsonl").exists() or not own_path.exists():
                continue
            own = _trial_ll(own_path)
            own = own[own["model"] == VARIANT_FULL]
            xfer = _trial_ll(d / "cv_trial_ll.jsonl")
            xfer = xfer[xfer["model"] == VARIANT_FULL]
            wide = _paired(own, xfer, ("_own", "_xfer"), f"{slug}/pooled-{group}")
            by_group.setdefault(group, []).append((study.short_label, wide))

    out = []
    for group, parts in by_group.items():
        rows = []
        for label, wide in parts:
            diff = (wide["held_out_ll_xfer"] - wide["held_out_ll_own"]).to_numpy()
            rows.append(
                {
                    "donor_group": group,
                    "recipient": label,
                    "n_trials": int(len(wide)),
                    "transfer_ll": float(wide["held_out_ll_xfer"].mean()),
                    "own_ll": float(wide["held_out_ll_own"].mean()),
                    "diff": float(diff.mean()),
                    "ci_95": _boot_ci(diff, wide["subject_id"].to_numpy(), n_boot, rng),
                }
            )
        # Combined: participants are namespaced by recipient so the cluster
        # bootstrap cannot merge two experiments' subject ids.
        allw = pd.concat(
            [
                w.assign(subject_id=lab + "|" + w["subject_id"].astype(str))
                for lab, w in parts
            ]
        )
        diff = (allw["held_out_ll_xfer"] - allw["held_out_ll_own"]).to_numpy()
        rows.append(
            {
                "donor_group": group,
                "recipient": "combined",
                "n_trials": int(len(allw)),
                "transfer_ll": float(allw["held_out_ll_xfer"].mean()),
                "own_ll": float(allw["held_out_ll_own"].mean()),
                "diff": float(diff.mean()),
                "ci_95": _boot_ci(diff, allw["subject_id"].to_numpy(), n_boot, rng),
            }
        )
        out.extend(rows)
    return out


def summarize(n_boot=1000, seed=0):
    """Each arm against the recipient's own LOSO ceiling, on matched trials.

    Two numbers per (pair, arm, role): the transfer's per-trial held-out LL, and
    its paired difference from the recipient's own-fit CV with the standard
    participant bootstrap. Plus the ablation contrast under transfer (full minus
    the reported base), which is the claim that survives even where the absolute
    level does not: the model still has to order its own ablations correctly on
    a study whose parameters it never saw.
    """
    rng = np.random.default_rng(seed)
    root = get_project_root() / "model" / "outputs"
    rows, contrasts = [], []
    for donor_label, recipient_label, kind in PAIRS:
        donor, recipient = _slug(donor_label), _slug(recipient_label)
        own_path = root / recipient / "cv_trial_ll.jsonl"
        if not own_path.exists():
            print(f"  skipping {recipient_label}: no reported CV outputs")
            continue
        own = _trial_ll(own_path)
        v_full, v_base = (
            _variant_for(recipient, "full"),
            _variant_for(recipient, "base"),
        )
        for arm in ARMS:
            path = outputs_dir(recipient, donor_label, arm) / "cv_trial_ll.jsonl"
            if not path.exists():
                continue
            xfer = _trial_ll(path)
            for role in ROLES:
                variant = _variant_for(recipient, role)
                a = own[own["model"] == variant]
                b = xfer[xfer["model"] == variant]
                if a.empty or b.empty:
                    continue
                wide = _paired(a, b, ("_own", "_xfer"), f"{recipient}/{variant}")
                diff = (wide["held_out_ll_xfer"] - wide["held_out_ll_own"]).to_numpy()
                status, why = transfer_status(donor, recipient, role)
                rows.append(
                    {
                        "donor": donor_label,
                        "recipient": recipient_label,
                        "kind": kind,
                        "arm": arm,
                        "role": role,
                        "variant": variant,
                        # "test" = a real transfer test; "control" = the delta is
                        # response-layer only, because one side cannot move the
                        # latent the weights price. See `transfer_status`.
                        "status": status,
                        "status_reason": why,
                        "n_trials": int(len(wide)),
                        "transfer_ll": float(wide["held_out_ll_xfer"].mean()),
                        "own_ll": float(wide["held_out_ll_own"].mean()),
                        "diff": float(diff.mean()),
                        "ci_95": _boot_ci(
                            diff, wide["subject_id"].to_numpy(), n_boot, rng
                        ),
                    }
                )
            # Does the ablation ordering survive? full - base with the donor's
            # utility on both sides, against the same contrast in the recipient's
            # own fit. Each side is paired within its own run, so the two
            # contrasts are over the same trials but never mixed across runs.
            have = set(xfer["model"])
            if {v_full, v_base} <= have:
                x = _paired(
                    xfer[xfer["model"] == v_full],
                    xfer[xfer["model"] == v_base],
                    ("_f", "_b"),
                    f"{recipient} transfer full vs base",
                )
                o = _paired(
                    own[own["model"] == v_full],
                    own[own["model"] == v_base],
                    ("_f", "_b"),
                    f"{recipient} own full vs base",
                )
                d_x = (x["held_out_ll_f"] - x["held_out_ll_b"]).to_numpy()
                d_o = (o["held_out_ll_f"] - o["held_out_ll_b"]).to_numpy()
                contrasts.append(
                    {
                        "donor": donor_label,
                        "recipient": recipient_label,
                        "kind": kind,
                        "arm": arm,
                        "full_minus_base_transfer": float(d_x.mean()),
                        "ci_95": _boot_ci(d_x, x["subject_id"].to_numpy(), n_boot, rng),
                        "full_minus_base_own": float(d_o.mean()),
                    }
                )

    out = {
        "n_boot": n_boot,
        "seed": seed,
        "pairs": rows,
        "ablation_contrasts": contrasts,
        "pooled_donor": summarize_pooled_donor(n_boot, seed),
    }
    out_path = root / "transfer" / "transfer_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, out)

    if rows:
        df = pd.DataFrame(rows)
        print("\n=== Transfer vs own-fit (per-trial held-out LL) ===")
        print(
            "    rows marked (control) are NOT transfer tests: one side cannot "
            "move the\n    latent those weights price, so the delta is "
            "response-layer only."
        )
        for arm in ARMS:
            sub = df[df["arm"] == arm]
            if sub.empty:
                continue
            print(f"\n  arm: {arm}")
            for _, r in sub.iterrows():
                lo, hi = r["ci_95"]
                mark = "" if r["status"] == "test" else "  (control)"
                print(
                    f"    {r['donor']}->{r['recipient']:<3} {r['role']:<16} "
                    f"{r['transfer_ll']:+.4f} vs own {r['own_ll']:+.4f}  "
                    f"delta {r['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]{mark}"
                )
    if contrasts:
        print("\n=== full - base under transferred parameters ===")
        for c in contrasts:
            lo, hi = c["ci_95"]
            print(
                f"    [{c['arm']}] {c['donor']}->{c['recipient']:<3} "
                f"{c['full_minus_base_transfer']:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                f"(own fit: {c['full_minus_base_own']:+.4f})"
            )
    if out["pooled_donor"]:
        print("\n=== a pooled group's utility applied outside that group ===")
        for r in out["pooled_donor"]:
            lo, hi = r["ci_95"]
            print(
                f"    {r['donor_group']} -> {r['recipient']:<8} "
                f"{r['transfer_ll']:+.4f} vs own {r['own_ll']:+.4f}  "
                f"delta {r['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            )
    print(f"\nWrote {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pair",
        action="append",
        metavar="DONOR:RECIPIENT",
        help="ordered pair by paper label, e.g. 1b:3a (repeatable; default all)",
    )
    parser.add_argument("--arm", choices=ARMS, action="append", help="default: both")
    parser.add_argument(
        "--from-pooled",
        metavar="GROUP",
        help="apply a pooled group's shared utility (model/cv/pooled.py) to "
        "experiments outside it, e.g. --from-pooled food --to 3a --to 3b",
    )
    parser.add_argument(
        "--to", action="append", metavar="LABEL", help="recipient(s) for --from-pooled"
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="re-summarize the transfer runs already on disk, without running any",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run pairs whose outputs already exist (default: skip them)",
    )
    args = parser.parse_args()

    if args.from_pooled:
        if not args.to:
            raise SystemExit("--from-pooled needs at least one --to LABEL")
        for label in args.to:
            run_from_pooled(args.from_pooled, label, workers=args.workers)
        summarize(n_boot=args.n_boot, seed=args.seed)
        return

    if not args.summary_only:
        arms = args.arm or list(ARMS)
        if args.pair:
            wanted = []
            for spec in args.pair:
                d, _, r = spec.partition(":")
                match = [p for p in PAIRS if p[0] == d and p[1] == r]
                if not match:
                    raise SystemExit(
                        f"{spec!r} is not a designed pair — one of "
                        + ", ".join(f"{a}:{b}" for a, b, _ in PAIRS)
                    )
                wanted += match
        else:
            wanted = list(PAIRS)
        for donor_label, recipient_label, _ in wanted:
            for arm in arms:
                out = outputs_dir(_slug(recipient_label), donor_label, arm)
                if (out / "cv_trial_ll.jsonl").exists() and not args.force:
                    print(f"skipping {donor_label}->{recipient_label} [{arm}]: {out}")
                    continue
                run_pair(donor_label, recipient_label, arm, workers=args.workers)

    summarize(n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
