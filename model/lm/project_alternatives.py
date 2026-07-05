#!/usr/bin/env python3
"""
Per-scenario 2D projection of the LM-generated alternatives, exported as a tidy
JSONL artifact for downstream semantic-space diagnostics.

plot_alternatives.py runs a single *global* UMAP over all 16 scenarios at once,
which is right for the manuscript "semantic map" (the space splits into scenario
blobs). But to look at the alternatives *within one scenario* — what kinds of
counterfactual the LM proposes there, and how they spread — a global layout just
collapses that scenario to a tight blob. This script instead projects each
scenario's alternatives (plus that scenario's three observed actions) into their
own 2D layout, so the within-scenario action-type structure is legible.

The projection is computed here (Python: UMAP needs the embeddings) and written as
a flat JSONL the R notebook reads, joining each row's UMAP coordinates to the
scored features (g / risk / effort, averaged over runs and conditions per distinct
action) and the per-scenario cluster / nearest-observed-action labels. The R notebook
renders the figure in ggplot for styling consistency; UMAP is never re-run in R.

Output (outputs/lm/<slug>/):
  - lm_alternatives_projection.jsonl — one record per (scenario_label, action_text):
    {is_observed, observed_action, cluster, nearest_observed_action, sim_to_observed_action,
     g, risk, effort, dim1, dim2}. Observed-action rows carry observed_action (their action
     label) and null cluster; alternatives carry cluster/nearest_observed_action/sim and
     null observed_action.

Usage:
    uv run python model/lm/project_alternatives.py --study food_inv_desire

Requires (produced by embed_alternatives.py for the study):
    outputs/lm/<slug>/lm_embeddings.npz, lm_alternatives_semantic.jsonl,
    lm_runs.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import umap

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root


def _action_features(runs):
    """Mean g / risk / effort per distinct (scenario_label, action_text), averaged
    over runs and (for the joint studies) effort/intimacy cells. Also returns, per
    observed-action text, the observed_action label it stands for."""
    rows = []
    for actions, scen, obs in zip(
        runs["actions"], runs["scenario_label"], runs["observed_action"]
    ):
        for a in actions:
            rows.append(
                dict(
                    scenario_label=scen,
                    action_text=a["action_text"],
                    is_observed=bool(a["is_observed"]),
                    observed_action=obs if a["is_observed"] else None,
                    risk=a["risk"],
                    effort=a["effort"],
                    g=a["g"],
                )
            )
    df = pd.DataFrame(rows)
    feat = df.groupby(["scenario_label", "action_text"], as_index=False).agg(
        is_observed=("is_observed", "max"),
        observed_action=("observed_action", "first"),
        risk=("risk", "mean"),
        effort=("effort", "mean"),
        g=("g", "mean"),
    )
    return feat


def _project_scenario(emb, seed, n_neighbors, min_dist):
    """UMAP a single scenario's stacked embedding matrix to 2D. n_neighbors is
    clamped below the row count so small scenarios don't error. The defaults
    (n_neighbors=50, min_dist=0.4) favor a filled, globally coherent layout
    over tightly packed micro-clusters, which reads better at figure size."""
    n = emb.shape[0]
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, max(2, n - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(emb)


def main(study, seed, n_neighbors=50, min_dist=0.4):
    d = get_project_root() / "model" / "outputs" / "lm" / study
    npz_path = d / "lm_embeddings.npz"
    if not npz_path.exists():
        raise SystemExit(
            f"No embeddings at {npz_path}. Run embed_alternatives.py --study {study} first."
        )

    npz = np.load(npz_path, allow_pickle=False)
    alt_emb = npz["alt_emb"]
    obs_emb = npz["obs_emb"]
    obs_scenario = npz["obs_scenario"]
    obs_action = npz["obs_action"]

    sem = pd.read_json(d / "lm_alternatives_semantic.jsonl", lines=True)
    if len(sem) != len(alt_emb):
        raise SystemExit(
            f"alt_emb ({len(alt_emb)}) and semantic jsonl ({len(sem)}) are not aligned."
        )
    runs = pd.read_json(d / "lm_runs.jsonl", lines=True)
    feat = _action_features(runs)

    out_path = d / "lm_alternatives_projection.jsonl"
    scenarios = sorted(pd.unique(sem["scenario_label"]))
    n_written = 0
    with open(out_path, "w") as fh:
        for scen in scenarios:
            alt_mask = (sem["scenario_label"] == scen).to_numpy()
            obs_mask = obs_scenario == scen
            scen_alt_emb = alt_emb[alt_mask]
            scen_obs_emb = obs_emb[obs_mask]
            scen_obs_action = obs_action[obs_mask]

            stacked = np.vstack([scen_alt_emb, scen_obs_emb])
            xy = _project_scenario(stacked, seed, n_neighbors, min_dist)
            alt_xy = xy[: len(scen_alt_emb)]
            obs_xy = xy[len(scen_alt_emb) :]

            feat_scen = feat[feat["scenario_label"] == scen]
            feat_by_text = feat_scen.set_index("action_text")

            # alternatives
            sem_scen = sem[alt_mask].reset_index(drop=True)
            for i, srow in sem_scen.iterrows():
                txt = srow["action_text"]
                f = feat_by_text.loc[txt] if txt in feat_by_text.index else None
                rec = dict(
                    scenario_label=scen,
                    action_text=txt,
                    is_observed=False,
                    observed_action=None,
                    cluster=int(srow["cluster"]),
                    nearest_observed_action=srow["nearest_observed_action"],
                    sim_to_observed_action=float(srow["sim_to_observed_action"]),
                    risk=(None if f is None else _num(f["risk"])),
                    effort=(None if f is None else _num(f["effort"])),
                    g=(None if f is None else _num(f["g"])),
                    dim1=float(alt_xy[i, 0]),
                    dim2=float(alt_xy[i, 1]),
                )
                fh.write(json.dumps(rec) + "\n")
                n_written += 1

            # observed actions (one per observed_action label)
            obs_feat = feat_scen[feat_scen["is_observed"]].set_index("observed_action")
            for j, act in enumerate(scen_obs_action):
                f = obs_feat.loc[act] if act in obs_feat.index else None
                rec = dict(
                    scenario_label=scen,
                    action_text=(
                        None
                        if f is None
                        else f["action_text"]
                        if "action_text" in obs_feat.columns
                        else None
                    ),
                    is_observed=True,
                    observed_action=str(act),
                    cluster=None,
                    nearest_observed_action=str(act),
                    sim_to_observed_action=None,
                    risk=(None if f is None else _num(f["risk"])),
                    effort=(None if f is None else _num(f["effort"])),
                    g=(None if f is None else _num(f["g"])),
                    dim1=float(obs_xy[j, 0]),
                    dim2=float(obs_xy[j, 1]),
                )
                fh.write(json.dumps(rec) + "\n")
                n_written += 1
            print(f"  {scen}: {len(scen_alt_emb)} alts + {len(scen_obs_emb)} observed")

    print(f"\nWrote {n_written} rows for {len(scenarios)} scenarios to {out_path}")


def _num(x):
    return None if pd.isna(x) else float(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="food_inv_desire")
    parser.add_argument("--seed", type=int, default=42, help="UMAP random_state.")
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--min-dist", type=float, default=0.4)
    args = parser.parse_args()
    main(args.study, args.seed, args.n_neighbors, args.min_dist)
