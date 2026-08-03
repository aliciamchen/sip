"""Surprise-weighted comparison-set reweighting — the likelihood-layer module.

The reported observer reweights its comparison set toward alternatives that are
informative about the CONTRASTIVE-ONLY questions it faces (the variables that do
not price the observed action first-order):

    p_eta(a | a_obs, tau)  proportional to  p(a) * exp(eta * S * v(a))

with S the surprise of the observed action under the baseline (ungated) actor
policy, v(a) a per-question sensitivity score, and eta a single fitted gain per
study, nested at eta = 0. The manuscript states the rule in the methods and
develops the formal detail in the SI: the forgone-value reading of S, the
criterion that decides which questions are reweighted, and the specification
checks.

Design, deliberately mirroring `_priors.py`: the reweighting lives entirely at
the likelihood layer. It produces a replacement `prior_table` for the observer's
table kwargs, so `observers.py` and `actors.py` are untouched and the
preregistered path (`config=None`) stays byte-identical. The only
thing the fit helpers must do is thread eta through the parameter vector and
build their observer tables through `gated_table_kwargs`.

Per-study specification (one rule, instantiated by which questions are
contrastive-only in each study):

    1a  food_inv_desire       none (desire prices the action)      -> NO reweighting
    1b  food_inv_joint_de     physical state                       -> swing
    2a  food_inv_intimacy     intimacy                             -> |d risk|
    2b  food_inv_joint_ie     intimacy + physical state            -> |d risk| + swing
    3a  nonfood_inv_joint_de  physical state                       -> swing
    3b  nonfood_inv_joint_ie  intimacy + physical state            -> |d risk| + swing

Ablations carry the reweighting only for the contrastive-only questions THEIR
OWN utility prices, so that
`full - ablation` isolates the utility term rather than mixing in the mechanism:
`base` has no discomfort term (no intimacy channel) and `discomfort_only` has no
effort term (no world-state channel). `variant_targets` encodes this; a variant
with no channel gets no reweighting and no extra parameter.
"""

import jax
import jax.numpy as jnp

# A slot is in the comparison set when its base prior is above this; padded null
# slots carry ~1e-8. Matches the diagnostics-layer value the development used.
ACTIVE_EPS = 1e-6
# Floor on E[pi_0(a_obs)] before the log, so a numerically impossible observed
# action gives a large-but-finite surprise instead of inf.
SURPRISE_CLIP = 1e-9

# Which questions each study's observer is asked, and which of those are
# contrastive-only (hence reweighted). "world" = the physical world state.
STUDY_CONTRASTIVE = {
    "food_inv_desire": (),
    "food_inv_joint_de": ("world",),
    "food_inv_intimacy": ("intimacy",),
    "food_inv_joint_ie": ("intimacy", "world"),
    "nonfood_inv_joint_de": ("world",),
    "nonfood_inv_joint_ie": ("intimacy", "world"),
}
# A target needs a utility channel to act through: the world-state score works
# through the effort term, the intimacy score through the discomfort term.
TARGET_REQUIRES_WEIGHT = {"world": "w_e", "intimacy": "w_d"}


def variant_targets(slug, utility_param_names, enabled=True):
    """The reweighted targets for one (study, variant): the study's
    contrastive-only questions, minus those whose utility channel this variant
    lacks. Empty tuple means no reweighting and no eta for this variant.

    `enabled=False` (the run config's `--no-reweighting`) empties the targets for
    every (study, variant), which is how the PREREGISTERED model is reached: one
    switch, applied at the single place the scope rule is evaluated, so the fit,
    the CV fold refits, and the warm-start vector's extras cannot disagree about
    whether eta exists."""
    if not enabled:
        return ()
    have = set(utility_param_names)
    return tuple(
        t for t in STUDY_CONTRASTIVE[slug] if TARGET_REQUIRES_WEIGHT[t] in have
    )


def uses_reweighting(slug, utility_param_names, enabled=True):
    return len(variant_targets(slug, utility_param_names, enabled)) > 0


def sensitivity(target, table_kwargs):
    """Per-slot sensitivity score v(a) for one question.

    Each latent enters the utility through exactly one term, so the score is the
    leading-order feature contrast (SI: si:reweighting-detail):
      - "world":    |effort(a | high) - effort(a | low)|, the effort swing. The
                    effort table carries an effort_condition axis whenever the
                    world state is inferred, which is exactly when this target
                    is used.
      - "intimacy": |risk(a) - risk(a_obs)|, with the observed action in slot 0.
    Returns an array shaped like the base prior table (cell grid + slot).
    """
    if target == "world":
        effort = table_kwargs["effort_table"]
        return jnp.abs(effort[..., 1, :] - effort[..., 0, :])
    if target == "intimacy":
        risk = table_kwargs["risk_table"]
        return jnp.abs(risk - risk[..., 0:1])
    raise ValueError(f"unknown reweighting target {target!r}")


def combined_sensitivity(targets, table_kwargs):
    """Sum of the per-question scores. Parameter-free: every score is a contrast
    of features on the shared normalized [0, 1] elicitation scale, so the sum
    needs no relative weight. One score per cell x slot serves all of a study's
    questions, giving ONE reweighted comparison set and ONE joint posterior per
    cell (the coherence property; the per-question-posterior alternative was
    tested and is equivalent-or-worse, RESULTS.md 13)."""
    if not targets:
        raise ValueError("combined_sensitivity called with no targets")
    v = sensitivity(targets[0], table_kwargs)
    for t in targets[1:]:
        v = v + sensitivity(t, table_kwargs)
    return v


def action_surprise(actor_fn, core_params, utility_param_names, table_kwargs):
    """S = -log E_z[pi_0(a_obs | z)], one scalar per (run, cell).

    Computed from the BASELINE (ungated) actor policy at the current utility
    parameters: the reweighting is never used to compute its own surprise, so
    there is no fixed point to solve — but S is recomputed every optimizer step,
    which is what makes eta and the utility weights jointly estimable.

    `actor_fn` is the variant's actor memo; `table_kwargs` carries a leading run
    axis, so the actor is vmapped over runs. The expectation over latents is the
    mean over every axis the policy carries beyond the cell grid.
    """
    kwargs = {"alpha": 1.0}
    for i, name in enumerate(utility_param_names):
        kwargs[name] = core_params[i]

    def _one(run_slice):
        return actor_fn(**kwargs, **run_slice)

    policy = jax.vmap(_one)({k: jnp.asarray(v) for k, v in table_kwargs.items()})
    # policy: (run, slot, *cell, *latents); slot 0 is the observed action.
    observed = policy[:, 0]
    n_cell_axes = table_kwargs["prior_table"].ndim - 1  # minus the slot axis
    latent_axes = tuple(range(n_cell_axes, observed.ndim))
    p_obs = observed.mean(axis=latent_axes) if latent_axes else observed
    return -jnp.log(jnp.clip(p_obs, SURPRISE_CLIP, None))


def gated_prior(prior, surprise, v, eta, floor=0.0):
    """Masked softmax: logits = log p(a) + eta * S * v(a) on active slots.

    `floor` places a tiny mass on inactive slots instead of exact zero, for
    consumers whose gradient w.r.t. the prior is only finite where it is
    strictly positive. The plain-JAX observers guard zeros explicitly and use
    floor = 0.
    """
    active = prior > ACTIVE_EPS
    log_prior = jnp.log(jnp.clip(prior, 1e-12, None))
    logits = jnp.where(active, log_prior + eta * surprise[..., None] * v, -jnp.inf)
    m = jnp.max(logits, axis=-1, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    w = jnp.where(active, jnp.exp(logits - m), floor)
    denom = w.sum(-1, keepdims=True)
    # An all-inactive row (possible only with floor = 0) would give 0/0; the
    # observers guard their renormalization the same way.
    return w / jnp.where(denom > 0.0, denom, 1.0)


def gated_table_kwargs(
    slug, utility_param_names, table_kwargs, core_params, eta, actor_fn, floor=0.0
):
    """Table kwargs with `prior_table` replaced by the reweighted comparison
    weights. Returns the input unchanged when this (study, variant) has no
    contrastive-only question its utility can act on — so the unreweighted path is
    reached by construction rather than by a flag."""
    targets = variant_targets(slug, utility_param_names)
    if not targets:
        return table_kwargs
    tk = {k: jnp.asarray(v) for k, v in table_kwargs.items()}
    s = action_surprise(actor_fn, core_params, utility_param_names, tk)
    v = combined_sensitivity(targets, tk)
    return dict(tk, prior_table=gated_prior(tk["prior_table"], s, v, eta, floor=floor))


# ==============================================================================
# Per-(study, variant) configuration for the fit helpers
# ==============================================================================
# The surprise term needs the variant's ACTOR (the reweighting is computed from
# the baseline actor policy), which the observer registries don't carry — so the
# mapping lives here, next to the rule it serves.

FAMILY_BY_SLUG = {
    "food_inv_desire": "desire",
    "food_inv_joint_de": "joint_de",
    "food_inv_intimacy": "intimacy",
    "food_inv_joint_ie": "joint_ie",
    "nonfood_inv_joint_de": "joint_de",
    "nonfood_inv_joint_ie": "joint_ie",
}


#: Variants that reuse another variant's actor. `base_shared` is `base`'s utility
#: scored against a different comparison set, so it shares `base`'s actor exactly;
#: the actor name is derived from the variant string, which would otherwise look
#: for a function that does not exist.
_ACTOR_ALIAS = {"base_shared": "base"}


def _actor(family, variant):
    """The actor memo for one (family, variant), imported lazily so this module
    stays importable in contexts that never build an actor."""
    import actors

    prefix = (
        "actor_discrete" if family in ("desire", "joint_de") else "actor_continuous"
    )
    name = _ACTOR_ALIAS.get(variant, variant)
    return getattr(actors, f"{prefix}_{name}_padded_{family}")


def config_for(slug, variant, utility_param_names, enabled=True):
    """Reweighting config for one (study, variant), or None when the rule grants
    none (in which case the fit is the preregistered one and gains no parameter).

    Returned dict is what the `fit_*_observer_joint` helpers accept as
    `reweighting=`: the slug (for the scope rule), the variant's actor, and the
    resolved targets for logging/provenance.

    `enabled=False` returns None for every (study, variant) — see
    `variant_targets`. Note the None-means-preregistered invariant then does
    double duty: it already covered the studies and ablations the scope rule
    exempts, and now covers the whole-run switch too, so no caller needs a
    second code path.
    """
    targets = variant_targets(slug, utility_param_names, enabled)
    if not targets:
        return None
    return {
        "slug": slug,
        "variant": variant,
        "targets": targets,
        "utility_param_names": tuple(utility_param_names),
        "actor_fn": _actor(FAMILY_BY_SLUG[slug], variant),
    }


def apply(reweighting, table_kwargs, core_params, eta, floor=0.0):
    """Thin wrapper the fit helpers call inside their loss: returns the table
    kwargs with the reweighted `prior_table`, or the input unchanged when
    `reweighting` is None (the preregistered path)."""
    if reweighting is None:
        return table_kwargs
    return gated_table_kwargs(
        reweighting["slug"],
        list(reweighting["utility_param_names"]),
        table_kwargs,
        core_params,
        eta,
        reweighting["actor_fn"],
        floor=floor,
    )
