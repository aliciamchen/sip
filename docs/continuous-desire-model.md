# Continuous-desire model — design note

Status: model code implemented May 2026. The K=20 LM elicitation has since been
run for all four studies, Study 1a has fits + LOSO CV run on a partial sample,
and the other three studies have pilot-scale data only (see "Status / what's
left" at the bottom). This note records the design decision and the
implementation plan so the manuscript wording can be settled against it.

## Desire

The desire dependent variable is a continuous **0–100** rating ("how much do the
two people would like the food?", not-at-all → extremely) — the same scale as the
other two DVs (effort and intimacy are both 0–100); none are Likert. An earlier
version of the model inferred a **binary** latent `desire_condition ∈ {LOW,
HIGH}`, produced a posterior probability `P(reward = HIGH)`, and mapped it onto
the rating scale (e.g. `1 + 6 · P(high)` for the 1–7 Likert the manuscript used
at the time; `compute_desire_likert_se`).

That mapping conflates two different quantities:

- `P(high)` is the observer's **inferential uncertainty** — the posterior
  probability that the actor is in the high-desire condition.
- the rating is a **magnitude** — how much the actor wants the food.

Under the old mapping the model can only output a mid-scale rating by being
maximally *uncertain* (50/50 between low and high), never by confidently
inferring a *moderate* desire. The latent has only two values, so "moderate
desire" is not representable; the continuous rating is reconstructed by
interpolating a probability between two discrete hypotheses. It is also the only
latent treated as binary while its DV is continuous — intimacy (Studies 2a/2b)
is already handled the right way (a continuous latent over a 101-bin grid scored
with `compute_intimacy_nll`).

## The continuous formulation

Desire becomes a genuine continuous quantity `d` that **scales** a
desire-independent, LM-elicited action value. This is the standard
inverse-planning reward structure (reward magnitude × how much the action
attains the goal), with the magnitude made continuous instead of binary. The
reward term of the utility changes from `w_v · V(a|s,m)` to:

```
reward(a | s, d) = w_v · d · g(a|s)
```

- `d` — desire, on a [0,1] scale internally (read out to the 0–100 human rating as `100·d`).
- `g(a|s) ∈ [0,1]` — the **goal-satisfaction** of the action: how much this
  action results in the two people getting/eating the food. Desire-free: no
  sharing ≈ 0; low-risk and high-risk sharing both ≈ high (both deliver the
  food — the choice *between* them is governed by the risk and effort terms,
  not by desire).

The full utility is otherwise unchanged:

```
U(a | s, I, d) =  w_v · d · g(a|s)
               −  w_d · risk(a|s) · (1 − I)^γ
               −  w_e · effort(a|s)
```

`discomfort_only` (no reward term) is structurally unaffected; `base` becomes
`w_v · d · g − w_e · effort`.

### Why not interpolate between two LM V anchors

Linear interpolation between an LM-elicited `V_low` and `V_high` forces the
continuum to be a blend of two arbitrary endpoints. The multiplicative
`d · g(a|s)` instead treats desire as a real continuous intensity parameter
scaling a stable per-action value, which is a genuine generative story rather
than a blend, and gives the monotone behaviour we want (more desire → more
willing to pay an risk/effort cost to share). The one modelling commitment is
that desire scales a fixed action-value profile and cannot re-rank which action
is most goal-serving; with the literal "not at all → extremely" anchors there
is no aversion region (`d = 0` means "no reward pull", not "repelled").

## Two regimes for `d`

Desire is **inferred** in some studies and **given** in others. The LM is
involved in `g` everywhere, and additionally supplies the `d` *scalar* only in
the given-desire studies.

| Studies | desire role | `d` source | reward term |
|---|---|---|---|
| 1a `food_inv_desire`, 1b `food_inv_joint_de` | inferred | continuous latent `DesireLevels` (101 bins, [0,1]) | `w_v · desire · g(slot,…)` |
| 2a `food_inv_intimacy`, 2b `food_inv_joint_ie` | given | LM scalar `desire_table[scenario, desire_condition]` | `w_v · desire_table[s,r] · g(slot,…)` |

In the inferred case `d` is never elicited — the observer recovers it, and the
low/high desire paragraphs only define the endpoints of the human's rating
scale. In the given case the LM reads the scenario + the shown desire paragraph
and rates how much the two people would like the food, on the same scale; that scalar
plugs into the actor utility as a constant. Putting both on the same scale gives
a cross-study coherence check: the `d` 1a recovers from "the action a high-desire
dyad would take" should land near the `d` the LM assigns the high-desire
condition in 2a.

## LM elicitation changes

- **`g(a|s)`** replaces V. One desire-free rating per (scenario, action),
  0–6 → normalized [0,1]: "how much does each action result in the two people
  getting/eating the food." Shared across all four studies. Drops V's
  per-desire axis (cheaper: one prompt per scenario instead of two).
- **`desire` scalar** (given-desire studies only). One rating per
  (scenario, desire condition) = 16 × 2 = 32: "given this state, how much would
  the two people like the food," on the 0–100 scale (stored as [0,1] = rating/100).
- Alternative *generation* is unchanged (it never used V).
- Output CSV changes: `lm_scenario_v.csv → lm_scenario_g.csv`
  (drop `desire`); `lm_alternatives_v_<slug>.csv →
  lm_alternatives_g_<slug>.csv` (drop `desire_query`); add
  `lm_scenario_desire.csv` (scenario_label, desire_condition, desire).

## Likelihood

> **Superseded (June 2026).** The full-posterior categorical NLL described below
> has been replaced by the manuscript's belief-update Gaussian-mixture likelihood
> (`mixture_nll_1d`/`mixture_nll_2d` in `inverse/_helpers.py`): the DV is the
> belief update `posterior − prior`, each elicitation run k contributes a model
> update `δ_k = posterior mean − prior mean`, and a participant's update is scored
> under `(1/K) Σ_k N(u | δ_k, σ²)` with a fitted response-noise `σ` (isotropic
> bivariate for the joint studies). The posterior-mean summary noted parenthetically
> below is what the current procedure uses. See `rules/model.md` "DV likelihoods".

Historical (pre-mixture): `compute_desire_nll(posterior, response)` — full-posterior
NLL at the response bin, mapping the 0–100 rating directly onto the 101-bin [0,1]
grid (`idx = round(response)`), an exact parallel of `compute_intimacy_nll`. (A
posterior-mean + squared-error variant was considered as a smaller change but
discards the posterior shape — the mixture procedure above ultimately adopted the
posterior mean.)

## Implementation map

> **Partially superseded (June–July 2026).** This was the original build plan.
> Since then: the LM tables moved from the `lm_scenario_*.csv` / `lm_alternatives_*.csv`
> files named below to `lm_runs.jsonl` + `lm_alternatives.jsonl` (the CSVs remain only as a
> K=1 fallback); the categorical `compute_desire_nll` was replaced by the belief-update
> Gaussian mixture (see the Likelihood note above); and the separate in-sample predict stage
> was dropped entirely — CV is the sole prediction source, so the `predict_*.py` references
> below no longer apply.

- **`tables.py`** — add `DesireLevels`; add `load_lm_g`,
  `load_lm_scenario_desire`; rewrite the four
  `load_padded_lm_tables_{desire,joint_de,intimacy,joint_ie}` to emit a `g`
  table (drop the trailing desire axis of the old `v` table) and read the
  `_g_` alternatives CSVs.
- **`utility.py`** — `get_utility_{full,base}_padded_{desire,joint_de}`
  take a continuous `desire` scalar and a `g_padded_table`, reward term
  `w_v · desire · g`. `…_{intimacy,joint_ie}` keep `desire_condition`, take a
  `desire_table` and `g_padded_table`, reward term
  `w_v · desire_table[s,r] · g`. Rename the `get_lm_v_padded_*` helpers to
  `…_g_…` (drop the desire arg).
- **`actors.py`** — `actor_discrete_*_padded_{desire,joint_de}`: latent
  dim `desire_condition → desire: DesireLevels`.
  `actor_continuous_*_padded_{intimacy,joint_ie}`: keep `desire_condition`,
  thread `desire_table`.
- **`observers.py`** — `observer_reward_* → observer_desire_*` (infer
  `desire in DesireLevels`); `observer_joint_de_*` joint over
  `(desire in DesireLevels, effort_condition)`; `observer_intimacy_*` and
  `observer_joint_ie_*` keep `desire_condition` known, thread `desire_table`.
- **`inverse/_helpers.py`** — add `compute_desire_nll`; drop
  `compute_desire_likert_se`; rewrite the desire half of
  `fit_desire_observer_joint` / `fit_joint_de_observer_joint` to slice the
  101-bin desire posterior; `*_table_kwargs` load `g` (+ `desire_table` for
  2a/2b); `uses_v → uses_g`.
- **`lm/prompts.py`, `lm/_features_dispatcher.py`, `lm/score_merged.py`** —
  add the `g` and `desire` rating prompts/normalizers; swap the V pass for the
  `g` pass; add the per-condition `desire` pass for the given-desire studies.
- **`inverse/{fit,predict}_*.py`, `cv/_inverse_dispatcher.py`** — point at the
  renamed observers; desire CV/predict use the 101-bin posterior,
  `compute_desire_nll`, and `100·E[desire]`.

## Manuscript discrepancy (do not silently reconcile)

`SIP_journal/main.tex` describes the old binary-V model in three places that
would need rewriting to the `w_v·d·g` form:

- `:490` — signed V on a −3..+3 scale.
- `:525–529` — desire "enters through the signed-valence rating V(a|s,m)".
- `:515` — the observer "marginalizes over each candidate desire **state**"
  (discrete) — should describe a continuous desire grid, like intimacy.

The manuscript is the source of truth, so these are left for a deliberate edit
rather than changed as part of the code refactor.

## Status / what's left

- Model code: implemented and compiles; Study 1a was initially smoke-tested
  with a provisional `g` derived from the existing V CSVs (posterior normalizes
  over the 101-bin desire grid; more sharing observed → higher `E[desire]`).
- **Done — LM elicitation:** the K=20 elicitation (`generate_alternatives.py` +
  `score_merged.py`) has been run for all four studies; each study's
  `lm_runs.jsonl` is committed under `model/outputs/lm/<slug>/`.
- **In progress — data and fits:** Study 1a has roughly half its target sample
  collected, with fits and LOSO CV run on it. The other three studies have
  pilot-scale data only; their final samples have not yet been collected.
- **Pending — manuscript:** update the three passages above.
