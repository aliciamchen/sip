"""Fit observer + actor utility weights for food_inv_intimacy.

Study 2a — observer knows (desire, effort), infers intimacy. Each variant jointly
fits its utility weights, alpha_observer, and the response-noise sigma from this
experiment's belief-update data (no transfer between studies). Writes
outputs/food_inv_intimacy/fit_results.json.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import jax.numpy as jnp  # noqa: E402
import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    fit_intimacy_observer_joint,
    intimacy_table_kwargs,
    load_intimacy_data,
    parse_run_config_args,
    resolve_variant_table_kwargs,
    restart_records_to_rows,
    write_fit_manifest,
    write_json,
    write_jsonl,
)
from _priors import build_priors_kwarg  # noqa: E402
from observers import VARIANTS_INTIMACY as VARIANTS  # noqa: E402

EXPERIMENT_SLUG = "food_inv_intimacy"


def main(config=None):
    config = (
        config if config is not None else parse_run_config_args(description=__doc__)
    )
    print("=" * 60)
    print(f"Joint inverse fit: {EXPERIMENT_SLUG}")
    print("Fitting utility weights + alpha_observer + sigma per variant")
    print("=" * 60)

    data, action, scenario_idx, desire_condition, effort_condition, response = (
        load_intimacy_data(EXPERIMENT_SLUG)
    )
    # Resolve every variant's LM tables before any fitting starts, so a missing
    # table fails up front rather than after hours of fitting earlier variants.
    table_kwargs_by_variant = resolve_variant_table_kwargs(
        VARIANTS,
        lambda name, utility_names: intimacy_table_kwargs(
            utility_names, suffix=config.alts_suffix
        ),
    )
    # Informative-prior kwargs per variant (None in the canonical uniform config,
    # which keeps the fit byte-identical). The given-desire studies show no
    # relationship paragraph, so every variant reads the standard priors file
    # (base=False), mirroring intimacy_table_kwargs' single alternatives vintage.
    priors_by_variant = {
        name: build_priors_kwarg(EXPERIMENT_SLUG, config) for name in VARIANTS
    }
    # K alignment: the priors' run axis must match the feature tables' K. A K=1
    # priors file (e.g. the human-ceiling vintage) tiles up to the tables' K;
    # any other mismatch is a hard error.
    for name, pr in priors_by_variant.items():
        if pr is None:
            continue
        k_tables = table_kwargs_by_variant[name]["risk_table"].shape[0]
        for key, arr in pr.items():
            if arr is None:
                continue
            if arr.shape[0] == 1 and k_tables > 1:
                pr[key] = jnp.repeat(arr, k_tables, axis=0)
            elif arr.shape[0] != k_tables:
                raise ValueError(
                    f"{EXPERIMENT_SLUG}/{name}: priors K={arr.shape[0]} != "
                    f"feature tables K={k_tables} — re-run the priors "
                    f"elicitation with matching K_RUNS."
                )

    results = []
    restart_rows = []
    for variant_name, (obs_fn, utility_names) in VARIANTS.items():
        print(
            f"\n{'-' * 40}\nJointly fitting {variant_name} ({len(utility_names)} weights + alpha_observer + sigma)...\n{'-' * 40}"
        )
        params, nll, restarts = fit_intimacy_observer_joint(
            observer_fn=obs_fn,
            utility_param_names=utility_names,
            action=action,
            scenario_idx=scenario_idx,
            desire_condition=desire_condition,
            effort_condition=effort_condition,
            response=response,
            table_kwargs=table_kwargs_by_variant[variant_name],
            priors=priors_by_variant[variant_name],
            seed_key=f"{EXPERIMENT_SLUG}|{variant_name}",
        )
        use_grid = (
            priors_by_variant[variant_name] is not None
            and priors_by_variant[variant_name]["m_latent"] is not None
        )
        row = {
            "model": variant_name,
            "experiment": EXPERIMENT_SLUG,
            "nll": nll,
            "n_params": len(utility_names) + 2 + (1 if use_grid else 0),
            "param_alpha": 1.0,
            "alpha_observer": float(params[len(utility_names)]),
            "param_sigma": float(params[len(utility_names) + 1]),
        }
        if use_grid:
            row["param_prior_nu"] = float(params[len(utility_names) + 2])
        for i, name in enumerate(utility_names):
            row[f"param_{name}"] = float(params[i])
        results.append(row)
        restart_rows.extend(
            restart_records_to_rows(
                EXPERIMENT_SLUG,
                variant_name,
                utility_names,
                restarts,
                extra_param_names=("prior_nu",) if use_grid else (),
            )
        )

    output_dir = config.outputs_dir(EXPERIMENT_SLUG)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"config: {config.tag() if not config.is_canonical else 'canonical'} "
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
    write_fit_manifest(EXPERIMENT_SLUG, output_dir)


if __name__ == "__main__":
    main()
