#!/usr/bin/env python3
"""
Semantic structure of the LM-generated alternatives (diagnostic, not part of the
core fitting pipeline). The two exact-text diagnostics in the elicitation notebook
(run-to-run diversity, the cross-condition Jaccard panel) treat paraphrases as
distinct, so they are a lower bound on how much the LM is genuinely repeating
itself or reshaping the set. This script adds a *semantic* view by embedding each
unique alternative text and:

  1. clustering the alternatives PER SCENARIO into action *types* (e.g. "use
     separate utensils", "ask for extra plates", "tear by hand", "save for later"),
     so the within-cell diversity metric can count distinct kinds of action rather
     than distinct surface strings. Clustering is per scenario because the dominant
     axis of embedding variation is the scenario itself (scenario-specific names,
     food, setting), so a single global clustering would mostly recover the 16
     scenarios instead of abstract action types.
  2. labeling each alternative by its nearest observed action (no_share /
     low_risk_share / high_risk_share, by embedding cosine). This is scenario-
     invariant by construction — every scenario shares the same three-action frame
     — so it supports the cross-scenario anchoring view (does the imagined set skew
     toward lower-risk actions once a high-risk share has been observed?). The
     cosine to that nearest observed action also flags near-paraphrases of an
     already-listed action, which add little contrast to the actor's choice set.

It is kept separate from score_merged.py because it is an optional, embedding-based
analysis: the inverse fit never reads its output. The elicitation notebook
(analysis/food-inv-desire-lm-elicitation.qmd) reads the artifacts below behind a
`file.exists()` guard, so the notebook renders fine whether or not this has run.

Embeddings go through the Together AI API (reusing `client.load_api_key`);
clustering uses scipy (both already project deps — no new dependency). Set the
embedding model with --model.

Outputs (one folder per study, outputs/lm/<slug>/):
  - lm_alternatives_semantic.jsonl — one record per unique (scenario_label,
    action_text): {cluster (per-scenario id), nearest_observed_action, sim_to_observed_action}.
    Joined back to lm_alternatives.jsonl by (scenario_label, action_text) in the
    notebook.
  - lm_clusters.json — {model, k_per_scenario, dup_threshold, clusters:
    [{scenario, cluster, size, exemplars}]}, the per-scenario action types with
    nearest-centroid exemplar texts for interpretation.
  - lm_embeddings.npz — the mean-centered, normalized embeddings: `alt_emb` (aligned
    row-for-row with lm_alternatives_semantic.jsonl) + `obs_emb` with parallel
    `obs_scenario`/`obs_action` labels. Consumed by model/lm/plot_alternatives.py,
    which runs the UMAP projection and renders the figures.

Usage:
    uv run python model/lm/embed_alternatives.py --study food_inv_desire
    uv run python model/lm/embed_alternatives.py --study food_inv_desire --k 6

Requires:
    - TOGETHER_API_KEY in env or .env
    - outputs/lm/<slug>/lm_alternatives.jsonl produced by generate_alternatives.py
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from together import Together

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import MAX_RETRIES, load_api_key

# Embedding model on Together AI. intfloat/multilingual-e5-large-instruct is the
# active serverless embedding model (1024-dim, 514-token input limit); override
# with --model if the account has a different one provisioned. The alternative
# texts are single-sentence actions (~20-40 tokens), well under the 514-token cap,
# so no chunking is needed. The embeddings are mean-centered then re-normalized in
# _embed_all (counters e5's anisotropy; cosine == dot still holds).
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"
EMBED_BATCH = 100
# Batches are independent, so fire them concurrently (the Together client is shared
# across threads, as in client.get_ratings_concurrent). Capped to stay under the
# account's request-rate limit; lower it on a tighter tier.
EMBED_WORKERS = 8
OBSERVED_ACTIONS = ["no_share", "low_risk_share", "high_risk_share"]
SEED = 0


def _norm_text(text):
    return text.lower().strip()


def _embed_batch(client, batch, model):
    """Embed one batch, returning vectors in input order. Concurrent responses can
    arrive out of order, so sort by the response item's `index` rather than trusting
    the returned order."""
    resp = client.embeddings.create(model=model, input=batch)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def _embed_all(client, texts, model, workers=EMBED_WORKERS):
    """Embed `texts` (list of unique strings) and return an (n, d) mean-centered,
    L2-normalized matrix, one row per text in input order. Batches of EMBED_BATCH
    are independent, so they're fired concurrently (capped at `workers`) and
    reassembled by position; the shared client is thread-safe, as in
    client.get_ratings_concurrent."""
    batches = [
        (start, texts[start : start + EMBED_BATCH])
        for start in range(0, len(texts), EMBED_BATCH)
    ]
    out = [None] * len(texts)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {
            ex.submit(_embed_batch, client, batch, model): start
            for start, batch in batches
        }
        for fut in as_completed(futures):
            start = futures[fut]
            vecs = fut.result()
            out[start : start + len(vecs)] = vecs
            done += len(vecs)
            print(f"  embedded {done}/{len(texts)}", flush=True)
    arr = np.asarray(out, dtype=np.float64)
    # Mean-center to counter embedding anisotropy: raw e5 vectors sit in a narrow
    # high-cosine cone (every pair scores ~0.8+), so absolute cosines — e.g. the
    # near-paraphrase threshold — are uninformative. Subtracting the global mean
    # spreads them out; re-normalizing keeps cosine == dot.
    arr = arr - arr.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _kmeans(data, k):
    """scipy kmeans2 with a fixed seed, robust across scipy versions (the rng/seed
    kwarg was renamed). Returns (centroids, labels)."""
    try:
        return kmeans2(data, k, minit="++", rng=np.random.default_rng(SEED))
    except TypeError:
        pass
    try:
        return kmeans2(data, k, minit="++", seed=SEED)
    except TypeError:
        np.random.seed(SEED)
        return kmeans2(data, k, minit="++")


def main(study, k, dup_threshold, model, embed_workers=EMBED_WORKERS):
    api_key = load_api_key()
    scenarios_path = get_project_root() / "experiments" / "scenarios.csv"
    study_dir = get_project_root() / "model" / "outputs" / "lm" / study
    alts_path = study_dir / "lm_alternatives.jsonl"
    if not alts_path.exists():
        raise SystemExit(
            f"Alternatives JSONL not found at {alts_path}. Run "
            f"model/lm/generate_alternatives.py --study {study} first."
        )

    scenarios_df = pd.read_csv(scenarios_path).set_index("scenario_label", drop=False)
    alts_df = pd.read_json(alts_path, lines=True)

    # Unique (scenario, alternative text) pairs — what we report on. Embedding is
    # text-level (scenario-independent), so we embed unique texts once and reuse.
    alt_pairs = (
        alts_df[["scenario_label", "action_text"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    obs_texts = {
        s: [scenarios_df.loc[s, c] for c in OBSERVED_ACTIONS]
        for s in scenarios_df.index
    }

    unique_alt_texts = sorted(alts_df["action_text"].unique())
    unique_obs_texts = sorted({t for ts in obs_texts.values() for t in ts})
    all_texts = sorted(set(unique_alt_texts) | set(unique_obs_texts))

    print(
        f"\nEmbedding {len(all_texts)} unique texts "
        f"({len(unique_alt_texts)} alternatives + {len(unique_obs_texts)} "
        f"observed) with {model} ({embed_workers} concurrent batches)...",
        flush=True,
    )
    client = Together(api_key=api_key, max_retries=MAX_RETRIES)
    emb_matrix = _embed_all(client, all_texts, model, workers=embed_workers)
    emb = {t: emb_matrix[i] for i, t in enumerate(all_texts)}

    # Cluster the alternatives PER SCENARIO (not globally). The dominant axis of
    # embedding variation is the scenario itself (scenario-specific names, food,
    # setting), so a global clustering would mostly recover the 16 scenarios rather
    # than abstract action types. Per-scenario cluster ids (local to each scenario)
    # feed the within-cell semantic-diversity metric, where comparisons only ever
    # happen inside one scenario. Cross-scenario comparison instead uses the
    # scenario-invariant `nearest_observed_action` label computed below.
    scenario_clusters = {}  # scenario -> {alt_text: local cluster id}
    cluster_summary = []  # [{scenario, cluster, size, exemplars}]
    for s in scenarios_df.index:
        s_texts = sorted(
            set(alts_df.loc[alts_df["scenario_label"] == s, "action_text"])
        )
        if not s_texts:
            continue
        mat = np.vstack([emb[t] for t in s_texts])
        ks = min(k, len(s_texts))
        centroids, labels = _kmeans(mat, ks)
        local = {t: int(lab) for t, lab in zip(s_texts, labels)}
        scenario_clusters[s] = local
        cl_norm = centroids / np.maximum(
            np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12
        )
        for c in range(ks):
            members = [t for t in s_texts if local[t] == c]
            if not members:
                continue
            sims = np.array([emb[t] @ cl_norm[c] for t in members])
            order = np.argsort(-sims)
            cluster_summary.append(
                {
                    "scenario": s,
                    "cluster": c,
                    "size": len(members),
                    "exemplars": [members[i] for i in order[:3]],
                }
            )

    # Per (scenario, alt text): the per-scenario cluster id, the nearest observed
    # action (scenario-invariant action-type label for the anchoring view), and the
    # cosine to it (for the near-paraphrase check). `alt_emb` collects each row's
    # embedding in the SAME order as `records`, so the plotting script can index-
    # align lm_alternatives_semantic.jsonl rows with lm_embeddings.npz.
    records = []
    alt_emb = []
    for _, row in alt_pairs.iterrows():
        s, text = row["scenario_label"], row["action_text"]
        obs_mat = np.vstack([emb[t] for t in obs_texts[s]])
        sims = obs_mat @ emb[text]  # unit vectors -> cosine to each observed action
        nearest = int(np.argmax(sims))
        records.append(
            {
                "scenario_label": s,
                "action_text": text,
                "cluster": scenario_clusters[s][text],
                "nearest_observed_action": OBSERVED_ACTIONS[nearest],
                "sim_to_observed_action": float(np.max(sims)),
            }
        )
        alt_emb.append(emb[text])

    sem_path = study_dir / "lm_alternatives_semantic.jsonl"
    with open(sem_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Persist the embeddings (mean-centered, normalized) for the plotting script,
    # which runs the UMAP projection itself. `alt_emb` aligns row-for-row with
    # sem_path; observed-action embeddings are stored with parallel scenario/action labels
    # so they can be placed in the same projection (as anchor points).
    obs_emb, obs_scenario, obs_action = [], [], []
    for s in scenarios_df.index:
        for ci, c in enumerate(OBSERVED_ACTIONS):
            obs_emb.append(emb[obs_texts[s][ci]])
            obs_scenario.append(s)
            obs_action.append(c)
    emb_path = study_dir / "lm_embeddings.npz"
    np.savez(
        emb_path,
        alt_emb=np.asarray(alt_emb, dtype=np.float32),
        obs_emb=np.asarray(obs_emb, dtype=np.float32),
        obs_scenario=np.array(obs_scenario),
        obs_action=np.array(obs_action),
    )

    cluster_summary.sort(key=lambda d: (d["scenario"], -d["size"]))
    clusters_path = study_dir / "lm_clusters.json"
    with open(clusters_path, "w") as f:
        json.dump(
            {
                "model": model,
                "k_per_scenario": k,
                "dup_threshold": dup_threshold,
                "clusters": cluster_summary,
            },
            f,
            indent=2,
        )

    near_dup = np.mean([r["sim_to_observed_action"] >= dup_threshold for r in records])
    print("\n=== Done ===")
    print(f"  {sem_path.name}  ({len(records)} unique (scenario, alt) pairs)")
    print(
        f"  {clusters_path.name}  ({len(cluster_summary)} per-scenario types "
        f"across {len(scenario_clusters)} scenarios)"
    )
    print(
        f"  {emb_path.name}  ({len(alt_emb)} alt + {len(obs_emb)} observed "
        f"embeddings for plotting)"
    )
    print(
        f"  near-paraphrases of an observed action (cos >= {dup_threshold}): "
        f"{100 * near_dup:.1f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="food_inv_desire")
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Max action-type clusters per scenario.",
    )
    parser.add_argument(
        "--dup-threshold",
        type=float,
        default=0.85,
        help="Cosine threshold for flagging an alternative as an observed-action paraphrase.",
    )
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument(
        "--embed-workers",
        type=int,
        default=EMBED_WORKERS,
        help="Concurrent embedding-batch requests (lower on a tighter rate limit).",
    )
    args = parser.parse_args()
    main(
        args.study,
        args.k,
        args.dup_threshold,
        args.model,
        embed_workers=args.embed_workers,
    )
