"""Shared full-data fit protocol for the six inverse-planning studies.

Every `fit_<slug>.py` runs the same protocol and differs only in which family
it belongs to. That protocol -- resolve every variant's LM tables up front,
build the per-variant prior kwargs, check K alignment, fit, assemble the result
row, flatten the restart records, write outputs + manifest -- used to be copied
into all six wrappers, so a change to the fit contract (adding the reweighting's
`eta`, say) meant six identical hand-edits with nothing to catch a divergence.
It now lives here once, and the wrappers are thin, mirroring how
`model/cv/_inverse_dispatcher.py` already backs the `cv_<slug>.py` wrappers.

The studies differ along exactly four axes, captured in `_FAMILIES`:

  - which arrays the data loader returns (and hence what the fitter is passed),
  - which observer-table builder to call,
  - whether that builder takes `base=` (only the given-relationship studies have
    a relationship-free alternatives vintage for the base ablation),
  - which variant registry defines the ablations.

Everything else -- including which variants exist and which carry an `eta` --
is derived, not configured per study, so a new study is a registry entry plus a
wrapper rather than another copy of the protocol.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import jax.numpy as jnp  # noqa: E402
import pandas as pd  # noqa: E402

import _reweighting  # noqa: E402
from _helpers import (  # noqa: E402
    ALPHA_OBS_MAX,
    desire_table_kwargs,
    fit_desire_observer_joint,
    fit_intimacy_observer_joint,
    fit_joint_de_observer_joint,
    fit_joint_ie_observer_joint,
    intimacy_table_kwargs,
    joint_de_table_kwargs,
    joint_ie_table_kwargs,
    load_desire_data,
    load_intimacy_data,
    load_joint_de_data,
    load_joint_ie_data,
    parse_run_config_args,
    resolve_variant_table_kwargs,
    restart_records_to_rows,
    write_fit_manifest,
    write_json,
    write_jsonl,
)
from _priors import build_priors_kwarg, priors_base_variant  # noqa: E402
from observers import (  # noqa: E402
    VARIANTS_DESIRE,
    VARIANTS_INTIMACY,
    VARIANTS_JOINT_DE,
    VARIANTS_JOINT_IE,
)

# `data_names` names the loader's return values AFTER the leading
# (data, action, scenario_idx), in order, using the keyword each one is passed
# to the fitter under. That single list is what ties a family's loader to its
# fitter, so the two can't silently disagree about argument order.
_FAMILIES = {
    "desire": {
        "variants": VARIANTS_DESIRE,
        "loader": load_desire_data,
        "fitter": fit_desire_observer_joint,
        "table_kwargs": desire_table_kwargs,
        "tables_take_base": True,
        "data_names": ("effort_condition", "relationship_condition", "response"),
    },
    "intimacy": {
        "variants": VARIANTS_INTIMACY,
        "loader": load_intimacy_data,
        "fitter": fit_intimacy_observer_joint,
        "table_kwargs": intimacy_table_kwargs,
        "tables_take_base": False,
        "data_names": ("desire_condition", "effort_condition", "response"),
    },
    "joint_de": {
        "variants": VARIANTS_JOINT_DE,
        "loader": load_joint_de_data,
        "fitter": fit_joint_de_observer_joint,
        "table_kwargs": joint_de_table_kwargs,
        "tables_take_base": True,
        "data_names": ("relationship_condition", "response_desire", "response_effort"),
    },
    "joint_ie": {
        "variants": VARIANTS_JOINT_IE,
        "loader": load_joint_ie_data,
        "fitter": fit_joint_ie_observer_joint,
        "table_kwargs": joint_ie_table_kwargs,
        "tables_take_base": False,
        "data_names": ("desire_condition", "response_intimacy", "response_effort"),
    },
}

# Only the given-relationship studies have a base-ablation alternatives vintage
# elicited without the relationship paragraph; 2a/2b/3b infer intimacy and never
# show one. `tables_take_base` above encodes the same fact at the builder level.
FAMILY_BY_SLUG = _reweighting.FAMILY_BY_SLUG


def _domain_for(slug):
    """Stimulus set for a slug — the nonfood studies read the nonfood tables.
    Mirrors `_inverse_dispatcher._domain_for`."""
    return "nonfood" if slug.startswith("nonfood_") else "food"


def _table_kwargs_builder(family, slug):
    """Per-variant observer-table builder for one study, closing over the
    family's builder plus this study's stimulus domain."""
    spec = _FAMILIES[family]
    builder, takes_base = spec["table_kwargs"], spec["tables_take_base"]
    domain = _domain_for(slug)

    def build(variant_name, utility_names):
        kwargs = {}
        if domain != "food":
            kwargs["domain"] = domain
        if takes_base:
            kwargs["base"] = variant_name == "base"
        return builder(utility_names, **kwargs)

    return build


def _check_priors_k_alignment(slug, variant, priors, table_kwargs):
    """The priors' run axis must match the feature tables' K.

    A K=1 priors file (the human-ceiling vintage) tiles up to the tables' K;
    any other mismatch is a hard error rather than a silent broadcast, since
    mismatched K would pair each run's tables with the wrong run's priors.
    Mutates `priors` in place, as the fitters expect aligned arrays.
    """
    if priors is None:
        return
    k_tables = table_kwargs["risk_table"].shape[0]
    for key, arr in priors.items():
        if arr is None:
            continue
        if arr.shape[0] == 1 and k_tables > 1:
            priors[key] = jnp.repeat(arr, k_tables, axis=0)
        elif arr.shape[0] != k_tables:
            raise ValueError(
                f"{slug}/{variant}: priors K={arr.shape[0]} != feature tables "
                f"K={k_tables} — re-run the priors elicitation with matching "
                "K_RUNS."
            )


def _result_row(slug, variant, utility_names, params, nll, use_grid, rw):
    """One `fit_results.json` record.

    The parameter vector is laid out [*utility, alpha_observer, sigma, *extras]
    with extras in a fixed order (prior_nu, then eta), so the offsets below
    follow the same order the fitters pack.
    """
    n_util = len(utility_names)
    alpha_obs = float(params[n_util])
    row = {
        "model": variant,
        "experiment": slug,
        "nll": nll,
        "n_params": n_util + 2 + (1 if use_grid else 0) + (1 if rw else 0),
        "param_alpha": 1.0,
        "alpha_observer": alpha_obs,
        # True only when the optional alpha_observer bound is enabled AND this
        # fit sits on it (a CONSTRAINED optimum). Off by default, so normally
        # False — see _helpers.ALPHA_OBS_MAX.
        "alpha_observer_at_bound": bool(
            ALPHA_OBS_MAX is not None and alpha_obs >= ALPHA_OBS_MAX - 1e-4
        ),
        "param_sigma": float(params[n_util + 1]),
    }
    if use_grid:
        row["param_prior_nu"] = float(params[n_util + 2])
    if rw:
        row["param_eta"] = float(params[-1])
        row["reweighting_targets"] = list(rw["targets"])
    for i, name in enumerate(utility_names):
        row[f"param_{name}"] = float(params[i])
    return row


def main(slug, config=None, description=None):
    """Run the full-data fit for one study and write its outputs.

    Called by each `fit_<slug>.py` with its hardcoded slug. `config` is a
    RunConfig (parsed from argv when not supplied).
    """
    config = (
        config
        if config is not None
        else parse_run_config_args(description=description or __doc__)
    )
    family = FAMILY_BY_SLUG[slug]
    spec = _FAMILIES[family]
    variants = spec["variants"]
    fitter = spec["fitter"]

    print("=" * 60)
    print(f"Joint inverse fit: {slug}")
    print("Fitting utility weights + alpha_observer + sigma per variant")
    print("=" * 60)

    data, action, scenario_idx, *rest = spec["loader"](slug)
    fit_data_kwargs = dict(zip(spec["data_names"], rest))
    assert len(rest) == len(spec["data_names"]), (
        f"{family}: loader returned {len(rest)} arrays after scenario_idx but "
        f"data_names lists {len(spec['data_names'])}"
    )
    fit_data_kwargs["action"] = action
    fit_data_kwargs["scenario_idx"] = scenario_idx

    # Resolve every variant's LM tables before any fitting starts, so a missing
    # table fails up front rather than after hours of fitting earlier variants.
    table_kwargs_by_variant = resolve_variant_table_kwargs(
        variants, _table_kwargs_builder(family, slug)
    )
    # Informative-prior kwargs per variant (None in the preregistered uniform
    # config, which keeps the fit byte-identical). priors_base_variant (the
    # single source of truth, shared with the CV dispatcher) routes only the
    # base variant to its relationship-free priors vintage, and only when no
    # explicit --priors-file is set (see _priors.py).
    priors_by_variant = {
        name: build_priors_kwarg(
            slug,
            config,
            base=priors_base_variant(slug, name, config.priors_file),
        )
        for name in variants
    }
    for name, pr in priors_by_variant.items():
        _check_priors_k_alignment(slug, name, pr, table_kwargs_by_variant[name])

    results = []
    restart_rows = []
    for variant_name, (obs_fn, utility_names) in variants.items():
        print(
            f"\n{'-' * 40}\nJointly fitting {variant_name} "
            f"({len(utility_names)} weights + alpha_observer + sigma)...\n"
            f"{'-' * 40}"
        )
        rw = _reweighting.config_for(slug, variant_name, list(utility_names))
        params, nll, restarts = fitter(
            observer_fn=obs_fn,
            utility_param_names=utility_names,
            table_kwargs=table_kwargs_by_variant[variant_name],
            priors=priors_by_variant[variant_name],
            reweighting=rw,
            seed_key=f"{slug}|{variant_name}",
            **fit_data_kwargs,
        )
        priors = priors_by_variant[variant_name]
        use_grid = priors is not None and priors["m_latent"] is not None
        results.append(
            _result_row(slug, variant_name, utility_names, params, nll, use_grid, rw)
        )
        restart_rows.extend(
            restart_records_to_rows(
                slug,
                variant_name,
                utility_names,
                restarts,
                extra_param_names=(("prior_nu",) if use_grid else ())
                + (("eta",) if rw else ()),
            )
        )

    output_dir = config.outputs_dir(slug)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"config: {config.tag() if not config.is_preregistered else 'preregistered'} "
        f"-> {output_dir}"
    )
    print("\n" + "=" * 60 + "\nRESULTS SUMMARY\n" + "=" * 60)
    print(pd.DataFrame(results).to_string(index=False))
    results_path = output_dir / "fit_results.json"
    write_json(results_path, results)
    print(f"\nSaved fit results to {results_path}")
    restarts_path = output_dir / "fit_restarts.jsonl"
    write_jsonl(restarts_path, restart_rows)
    print(f"Saved per-restart fits to {restarts_path}")
    write_fit_manifest(slug, output_dir)
