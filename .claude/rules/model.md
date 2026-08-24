---
paths:
  - "model/**/*"
---

# Model structure

Production models use plain JAX. `model/memo_spec.py` is the executable memo-language specification, and compliance tests compare it with the production implementations. Any semantic model change must update both representations and pass `make test`.

The manuscript and `model/utility.py` define the utility. Preserve these implementation invariants:

- The reward term is `w_v · desire · g`; risk is weighted by `w_d`, and `gamma` controls intimacy's effect on risk.
- Actor `alpha` is fixed. Fits estimate the applicable utility parameters, `alpha_observer`, and response-noise `sigma`.
- Desire is inferred in Studies 1a, 1b, and 3a and given in Studies 2a, 2b, and 3b. Intimacy has the reverse status.
- LM tables carry a leading elicitation-run axis. Alternative slot 0 is always the participant-observed action; fit and CV likelihoods select that slot.
- Per-run predictions form the mixture-likelihood components. Do not average runs before evaluating the likelihood.

Use code docstrings and tests for array layouts and optimizer details rather than duplicating them here.

## Variants and tables

All active studies use `full`, `discomfort_only`, and `base`; given-relationship studies also use `base_shared`. Studies 1a, 1b, and 3a route `base` to relationship-free alternatives in `lm_runs_base.jsonl`. Their paper-facing "Base" is `base_shared`, resolved only through `study_registry.reported_base(slug)`; raw output keys remain unchanged. `base_prereg` is a reporting-layer label, not a fitted on-disk variant.

Canonical LM inputs are in `model/outputs/lm/<slug>/`:

- `lm_alternatives.jsonl` and `lm_runs.jsonl` for the reported variants.
- `lm_alternatives_base.jsonl` and `lm_runs_base.jsonl` for the relationship-free base where applicable.
- `*-diag.jsonl` diagnostic vintages, which must never overwrite canonical tables.

Loaders must validate required records and values before fitting. New Together AI call sites should use `model/lm/client.py` rather than calling the provider directly.

## Fit, CV, and outputs

Per-study `fit_<slug>.py` and `cv_<slug>.py` files are thin wrappers. Shared behavior belongs in `_fit_dispatcher.py` and `_inverse_dispatcher.py`, keyed by their family registries. Do not add per-study logic to wrappers.

Leave-one-scenario-out CV is the sole source of model predictions. Completed variant-fold refits are checkpointed with input and code fingerprints, so interrupted runs can resume safely. A standard study has 48 CV jobs; Studies 1a, 1b, and 3a have 64 because they include `base_shared`.

The reported run writes to `model/outputs/<slug>/` and uses comparison-set reweighting where configured. `--no-reweighting` is the preregistered configuration; it omits `eta` and writes under `alt/uniform-noreweight/`. Use `bin/prereg-eta0.sh` for the all-study preregistered run instead of relying on the Makefile's reported-output file targets.

Read `model/outputs/README.md` for output schemas. Fitted fields use `param_<name>` except the intentionally bare `alpha_observer`; reweighted records also include `param_eta` and `reweighting_targets`.

## Output vintages

Changes to compiled model graphs, observer or actor semantics, likelihoods, table routing, or fit/CV protocols require regenerating fits and CV for all affected studies as one coherent vintage. Use `bin/regenerate-vintage.sh`, then `bin/compare_vintage.py`, and review the numerical differences before replacing reported artifacts. Do not mix fit, CV, LM, or summary files from different vintages.

Exploratory transfer, pooled, and generalization analyses have dedicated modules and Make targets. Their module docstrings and manifests are the source of truth for assumptions, outputs, and expected runtime.

## Commands

Use `make help` for the current target list. Common entry points are:

```bash
make test
make check-reported
make lm-diag K_RUNS=1
make lm-base-diag K_RUNS=1
make fit
make cv
make model-comparison
```

LM commands spend API money. Follow `.claude/skills/rerun-lm-elicitation/SKILL.md`, estimate cost first, and obtain explicit approval before a paid full run.
