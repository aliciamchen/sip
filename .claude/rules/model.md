---
paths:
  - "model/**/*"
---

# Model structure

Models are built using the `memo` DSL with JAX backend. Both actor (forward planning) and observer (inverse planning) models use the canonical access utility with a uniform action prior.

## Canonical actor

```
P(a | s, I) ∝ exp( U(a|s, I) )

U(a|s, I) = w_v · V(a|s)
          − w_d · access(a) · (1 − I)^γ
          − w_e · effort(a)
```

Intimacy `I` scales the access-discomfort term (bodily/spatial/informational exposure) through a power-law modulator `(1 − I)^γ`: at high intimacy the penalty shrinks toward zero, so higher-access actions become relatively more attractive. The exponent γ is a free parameter (initialized at 1.0; γ = 1 reproduces the linear-intimacy special case). Empirically food prefers γ < 1 (late relaxation) and non-food prefers γ > 1 (early relaxation). `V(a|s, m)` is the signed valence of the action with respect to the actor's motivational state (in [-1, +1]; positive = serves the state, negative = actively counterproductive). Three ablations are fit and compared:

- **full** — the full utility above (the main Full model)
- **discomfort_only** — only the access-discomfort term (`−w_d · access · (1 − I)^γ`); drops V and effort (Discomfort-only)
- **base** — `w_v · V − w_e · effort`; no relational structure (Base model). Has no intimacy term, so γ does not apply.

Parameters: `w_v` (V weight), `w_d` (access-discomfort weight), `w_e` (effort weight), `gamma` (intimacy power-law exponent, free, init 1.0, clipped ≥ 1e-6 by the optimizer's clip), plus `alpha` (actor softmax temperature, fixed to 1) and `alpha_observer` (observer softmax temperature). Each ablation uses only the subset of weights its utility requires; full and discomfort_only fit γ, base does not.

## Where the utility values come from

All three components — V, access, effort — are LLM-elicited per scenario by `model/lm_scenario_params.py` (Llama-3.3-70B via Together AI, 10 runs averaged). The Together calls themselves go through `model/lm_client.py`, which fans NUM_RUNS calls across a thread pool, constrains output to a JSON schema via `response_format`, retries transient errors at the SDK layer, and checkpoints per-scenario; new LM call sites should reuse `get_ratings_concurrent` + the schema helpers (`numeric_action_schema`, `alternatives_array_schema`) rather than calling Together directly. CSV outputs include both `n_runs_*` and `n_failures_*` columns.

- **V**: `--feature v` mode produces `lm_scenario_v.csv` (16 × 4 × 2 — scenario × action × motivation, signed [-1, +1]).
- **access**, **effort**: default mode produces `lm_scenario_params.csv` (16 × 4 each, normalized [0, 2] and [0, 1]).
- **alternatives V** (no-alt observers only): `--feature v_alternatives` mode produces `lm_alternatives_v.csv`, scoring V for each LM-generated alternative under both motivation states.

`tables.py` loads these into `LLM_TABLES` (access/effort canonical) at import; `load_lm_v(domain)` lazily loads the V table; `load_padded_lm_tables()` builds the (16, 4, 2, MAX_ACTIONS) padded tables (access, effort, v, prior) used by the no-alt observers. If any required CSV is missing, the loader raises FileNotFoundError or returns None.

Every memo model takes the scenario tables as arguments (`access_table: ...`, `effort_table: ...`, and `v_table: ...` for the full and base variants) and has `scenario_idx: Scenarios` as a dimension, so predictions vary by scenario. The `discomfort_only` ablation is V-independent and doesn't take `v_table`.

## Core files

- `tables.py` — `Scenarios` / `RewardConditions` / `RelationshipConditions` / `EffortConditions` / `PaddedActionSlots` enums, `SCENARIO_LABELS`, `LLM_TABLES`, `LLM_TABLES_EFFORT`, padded-table loaders (`load_padded_lm_tables`, `load_padded_lm_tables_relationship`), domain-asset loader, V table loader
- `utility.py` — jit-compiled utility functions: `get_utility_full / discomfort_only / base` (with `_disc`, `_padded`, `_padded_rel` siblings) plus the effort-experiment counterparts. Dimension-agnostic — used by both canonical 4-action and effort 2-action actors.
- `actors.py` — actor memo models: `actor_forw_*` (forward), `actor_discrete_*` and `actor_continuous_*` (inverse), `actor_continuous_*_padded` and `_padded_rel` (no-alt), plus `actor_forw_effort_*` and `actor_continuous_effort_*` (effort experiment).
- `observers.py` — observer memo models: `observer_intimacy_*` and `observer_reward_*` (alt-shown), `observer_intimacy_*_padded` and `observer_reward_*_padded_rel` (no-alt), `observer_intimacy_effort_*` and `observer_effort_intimacy_*` (effort experiment).
- `lm_scenario_params.py` — LLM-calls Together AI to generate per-scenario access and effort
- `lm_client.py` — shared LM-call infrastructure: `get_ratings_concurrent` (thread-pooled fan-out + SDK retries), schema helpers (`numeric_action_schema`, `alternatives_array_schema`), JSON parsing helpers, and `load_api_key`. Used by `lm_scenario_params{,_effort}.py` and `lm_generate_alternatives.py`.
- `fit_forward_planning.py` — fits the three actor ablations to `data/food_forw_intimacy_desire/` (output: `forward_planning_fit_results.csv`, `forward_planning_fits.csv`)
- `fit_inverse_planning_alt.py` — alt-shown observers; fits only `alpha_observer` with frozen actor params (output: `inverse_planning_fit_results.csv`)
- `fit_inverse_planning_intimacy_noalt.py` — intimacy no-alt observer; **jointly fits all actor weights + α_observer** on no-alt data (not frozen from Exp 1, because the padded observer's variable-length action space differs from Exp 1's fixed 4-action space). Output: `inverse_planning_intimacy_noalt_fit_results.csv`
- `fit_inverse_planning_desire_noalt.py` — desire no-alt observer; same joint-fit structure as the intimacy no-alt fit but uses the **relationship-keyed** padded reward observers (`observer_reward_*_padded_rel`) and BCE NLL on P(reward=HIGH). Action space is keyed on relationship (not motivation) since the observer sees relationship and infers motivation. Loads tables via `load_padded_lm_tables_relationship`, which expects `lm_alternatives_relationship{,_features,_v}.csv`. Output: `inverse_planning_desire_noalt_fit_results.csv`
- `generate_inverse_planning_alt_preds.py` — emits per-scenario posterior predictions for alt-shown (`inv_plan_{intimacy,desire}_preds_{full,summary}.csv`)
- `generate_inverse_planning_intimacy_noalt_preds.py` — same for intimacy no-alt, using the joint-fit weights from `inverse_planning_intimacy_noalt_fit_results.csv`
- `generate_inverse_planning_desire_noalt_preds.py` — same for desire no-alt; emits `food_inv-desire_intimacy_noalt_preds_{full,summary}.csv` (`p_high` is what the slider response 0-100 encodes)
- `test_model_compliance.py` — validation tests

### Effort-experiment parallel pipeline

A second, parallel pipeline mirrors the canonical scripts on the effort stimulus set (`scenarios_effort.csv`): 16 scenarios × 2 actions × 2 effort conditions (low / high), with reward held fixed at HIGH so V is constant across actions and `w_v` is non-identified under the softmax (it's kept in the utility for parallelism with the canonical pipeline but stays near initialization). Scenario labels are shared with the canonical 16, so `Scenarios` / `SCENARIO_TO_IDX` are reused; effort adds a separate `EffortConditions` IntEnum and `EFFORT_CONDITION_TO_IDX` map.

- Effort utility functions and memo models live alongside the canonical ones in `utility.py`, `actors.py`, and `observers.py` (look for the `_effort` suffix). `tables.py` loads `LLM_TABLES_EFFORT` (`access`, `effort`, both shape 16×2×2; plus `access_marg` shape 16×2×2 — the effort-marginal access broadcast across the effort_condition dimension) at import.
- `lm_scenario_params_effort.py` — produces two CSVs: (1) `lm_scenario_params_effort.csv` (64 rows: 16 scenarios × 2 effort × 2 actions) — effort-conditional access + effort, where the LM is prompted with the full vignette + effort paragraph so the manipulation lands in the ratings; (2) `lm_scenario_params_effort_marginal.csv` (32 rows: 16 scenarios × 2 actions) — effort-marginal access only, where the LM is prompted with just the base vignette. The marginal pass is needed because the food_inv-effort_intimacy_alt observer does not see the effort paragraph and so must reason about access from the base vignette alone.
- `fit_forward_planning_effort.py` — fits the three actor ablations to `data/food_forw_intimacy_effort/`. Outputs: `forward_planning_effort_fit_results.csv`, `forward_planning_effort_fits.csv`.
- `fit_inverse_planning_intimacy_effort.py` — fits only `alpha_observer` for `food_inv-intimacy_effort_alt`, with actor weights frozen from `forward_planning_effort_fit_results.csv` (NOT the canonical `food_forw_intimacy_desire` fit, because the effort actor's 2-action softmax doesn't transplant).
- `generate_inverse_planning_intimacy_effort_preds.py` — emits `food_inv-intimacy_effort_alt_preds_{full,summary}.csv`.
- `fit_inverse_planning_effort_intimacy.py` — flips the inference target: observer infers effort condition (latent) given observed action × intimacy. Uses `observer_effort_intimacy_*` from `observers.py` and binary cross-entropy NLL (slider 0–100 = P(effort_high)·100). Actor weights frozen from `forward_planning_effort_fit_results.csv`, but the actor's utility is evaluated with **effort-marginal access** (`LLM_TABLES_EFFORT['access_marg']`) instead of the effort-conditional table — because the observer in this experiment does not see the effort paragraph and so cannot perceive any effort-induced setting differences in the access of an action. The effort term itself remains effort-conditional (the observer does compute likelihoods under each candidate effort condition). Output: `inverse_planning_effort_intimacy_fit_results.csv`.
- `generate_inverse_planning_effort_intimacy_preds.py` — emits `food_inv-effort_intimacy_alt_preds_{full,summary}.csv`. The `summary` CSV's column `p_effort_high` is what the slider response 0-100 encodes.

## Cross-validation

All model-vs-human correlations reported in the analysis qmds are out-of-sample, from leave-one-scenario-out (LOSO) CV. The analysis qmds load CV-prediction CSVs (`cv_loso_*`) as the source for all plots.

- `cv/loso_forward.py` — Exp 1; refits $w_v, w_d, w_e$ per fold. Outputs: `cv_loso_forward.csv`, `cv_loso_preds.csv`
- `cv/loso_inverse_alt.py` — Exp 2a intimacy + 2b desire; refits only $\alpha_{\mathrm{obs}}$ per fold (actor frozen from all-data Exp 1 fit, same 4-action space). Outputs: `cv_loso_inv_plan_{intimacy,desire}_alt_preds_summary.csv`, `cv_loso_inverse_alt_folds.csv`
- `cv/loso_inverse_intimacy_noalt.py` — Exp 2c no-alt; joint LOSO refit of all actor weights + $\alpha_{\mathrm{obs}}$ per fold. Outputs: `cv_loso_food_inv-intimacy_desire_noalt_preds_summary.csv`, `cv_loso_inverse_intimacy_noalt_folds.csv`
- `cv/loso_forward_effort.py` — `food_forw_intimacy_effort`; refits $w_v, w_d, w_e$ per fold (note `w_v` is non-identified). Outputs: `cv_loso_forward_effort.csv`, `cv_loso_preds_effort.csv`
- `cv/loso_inverse_intimacy_effort.py` — `food_inv-intimacy_effort_alt`; refits only $\alpha_{\mathrm{obs}}$ per fold (actor frozen from the effort all-data forward fit). Outputs: `cv_loso_food_inv-intimacy_effort_alt_preds_summary.csv`, `cv_loso_inverse_intimacy_effort_folds.csv`
- `cv/loso_inverse_intimacy_effort_intimacy.py` — `food_inv-effort_intimacy_alt`; refits only $\alpha_{\mathrm{obs}}$ per fold (actor frozen from the effort all-data forward fit). Outputs: `cv_loso_food_inv-effort_intimacy_alt_preds_summary.csv`, `cv_loso_inverse_intimacy_effort_intimacy_folds.csv`

The non-CV `fit_*` / `generate_*` pipelines still produce all-data fits; AIC and fitted-parameter tables in the qmds use the all-data fit, but all model-vs-human displays use the CV predictions.

Model outputs are saved to `model/outputs/`. Preregistration documents are in `model/preregs/`. Legacy/experimental code is in `model/sandbox/`.

## Commands

LLM-derived scenario parameters (prerequisite for all fits; requires `TOGETHER_API_KEY` in `.env`; Llama-3.3-70B via Together AI, 10 runs averaged):

```bash
uv run python model/lm_scenario_params.py                                   # canonical food access+effort: 16×4 → lm_scenario_params.csv
uv run python model/lm_scenario_params.py --domain nonfood                  # nonfood access+effort: → lm_scenario_params_nonfood.csv
uv run python model/lm_scenario_params.py --feature v                       # food signed-valence V: → lm_scenario_v.csv (16×4×2: scenario × action × motivation, values in [-1, +1])
uv run python model/lm_scenario_params.py --feature v --domain nonfood      # nonfood signed-valence V: → lm_scenario_v_nonfood.csv
uv run python model/lm_scenario_params.py --feature v_alternatives          # food V for motivation-conditioned LM alternatives: → lm_alternatives_v.csv
uv run python model/lm_generate_alternatives.py --conditioning relationship                     # food relationship-conditioned alternatives: → lm_alternatives_relationship.csv (256 cells: 16 × 4 × 4)
uv run python model/lm_scenario_params.py --feature access_effort_alternatives_relationship     # access/effort for those alternatives: → lm_alternatives_relationship_features.csv
uv run python model/lm_scenario_params.py --feature v_alternatives_relationship                 # V for those alternatives (under both motivation states): → lm_alternatives_relationship_v.csv
uv run python model/lm_scenario_params_effort.py                            # effort: 64-row conditional + 32-row marginal
```

Forward-planning fits (3 ablations: Base / Discomfort-only / Full). All variants now use LM-elicited V; `discomfort_only` is V-independent.

```bash
uv run python model/fit_forward_planning.py                                 # food
uv run python model/fit_forward_planning.py --domain nonfood                # nonfood (writes *_nonfood.csv outputs)
uv run python model/fit_forward_planning_effort.py                          # effort (V hardcoded constant, since reward fixed HIGH)
```

Inverse-planning fits + prediction generators:

```bash
uv run python model/fit_inverse_planning_alt.py                            # alt-shown (intimacy + desire), α_observer only
uv run python model/generate_inverse_planning_alt_preds.py
uv run python model/fit_inverse_planning_intimacy_noalt.py                      # intimacy no-alt, joint fit (all weights + α_observer)
uv run python model/generate_inverse_planning_intimacy_noalt_preds.py
uv run python model/fit_inverse_planning_desire_noalt.py               # desire no-alt, joint fit (all weights + α_observer)
uv run python model/generate_inverse_planning_desire_noalt_preds.py
uv run python model/fit_inverse_planning_intimacy_effort.py                     # food_inv-intimacy_effort_alt, α_observer only
uv run python model/generate_inverse_planning_intimacy_effort_preds.py
uv run python model/fit_inverse_planning_intimacy_effort_intimacy.py            # food_inv-effort_intimacy_alt, α_observer only
uv run python model/generate_inverse_planning_effort_intimacy_preds.py
```

LOSO cross-validation (16 folds × 3 variants per experiment; the analysis qmds plot from these CSVs):

```bash
uv run python model/cv/loso_forward.py                     # refits w_v, w_d, w_e, β per fold (food)
uv run python model/cv/loso_forward.py --domain nonfood    # nonfood (writes *_nonfood.csv outputs)
uv run python model/cv/loso_inverse_alt.py                 # refits only α_observer per fold
uv run python model/cv/loso_inverse_intimacy_noalt.py               # intimacy no-alt, joint fit per fold
uv run python model/cv/loso_inverse_desire_noalt.py        # desire no-alt, joint fit per fold (relationship-keyed)
uv run python model/cv/loso_forward_effort.py              # refits w_d, w_e, β per fold (w_v non-identified)
uv run python model/cv/loso_inverse_intimacy_effort.py              # refits only α_observer per fold
uv run python model/cv/loso_inverse_intimacy_effort_intimacy.py     # refits only α_observer per fold
```

`fit_forward_planning.py` and `cv/loso_forward.py` accept `--domain food|nonfood`. Food is the default and writes the canonical filenames (`forward_planning_*.csv`, `cv_loso_forward.csv`, `cv_loso_preds.csv`); nonfood writes `*_nonfood.csv` siblings. Both branches share the same memo models in `actors.py` / `observers.py` — only the scenario-label↔index map and the LLM tables differ (see `tables.load_domain_assets`).

CV outputs (in `model/outputs/`):
- `cv_loso_forward.csv` / `cv_loso_preds.csv` — per-fold fits + per-trial held-out forward predictions
- `cv_loso_forward_nonfood.csv` / `cv_loso_preds_nonfood.csv` — same, nonfood
- `cv_loso_food_inv-intimacy_desire_alt_preds_summary.csv` / `cv_loso_food_inv-desire_intimacy_alt_preds_summary.csv` / `cv_loso_inverse_alt_folds.csv`
- `cv_loso_food_inv-intimacy_desire_noalt_preds_summary.csv` / `cv_loso_inverse_intimacy_noalt_folds.csv`
- `cv_loso_food_inv-desire_intimacy_noalt_preds_summary.csv` / `cv_loso_inverse_desire_noalt_folds.csv`
- `cv_loso_forward_effort.csv` / `cv_loso_preds_effort.csv`
- `cv_loso_food_inv-intimacy_effort_alt_preds_summary.csv` / `cv_loso_inverse_intimacy_effort_folds.csv`
- `cv_loso_food_inv-effort_intimacy_alt_preds_summary.csv` / `cv_loso_inverse_intimacy_effort_intimacy_folds.csv`

The non-CV `fit_*` / `generate_*` pipelines still produce all-data fits — AIC and fitted-parameter tables in the qmds use the all-data fit, but all model-vs-human displays use the CV predictions.

Tests:

```bash
uv run python model/test_model_compliance.py
```
