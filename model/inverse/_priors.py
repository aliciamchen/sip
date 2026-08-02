"""Informative-prior machinery for the inverse fits.

STATUS: evaluated, not adopted as the reported model. A full K=20 evaluation
found informative priors suppress the formal>intimate effort gradient -- they
hand the fit a competing explanation for the belief-update data, driving
alpha_observer up or collapsing gamma -- while only improving the low-risk-dip
magnitude. Every preregistration specifies uniform priors, and the reported
fits use them. This module stays as a switchable configuration so the question
can be answered with a run rather than an argument; it is not on the reported
path.

Every observer in observers.py is a Bayes inversion of the actor policy under
a UNIFORM latent prior, so an informative-prior posterior is exactly the
uniform-prior posterior reweighted by the prior and renormalized:
    post_inf(z | a) ∝ prior(z) · post_unif(z | a).
That identity lets the prior live entirely at the likelihood layer -- the
observers (fast and memo reference) are untouched, and the uniform path stays
byte-identical (see test_model_compliance nesting tests).

Grid latents (desire / intimacy) get a discretized Beta(mean m, concentration
nu) prior with a single fitted nu per study (param_prior_nu; uniform is nested
at m = 0.5, nu = 2). The 2-state effort latent's prior is the elicited scalar
P(effort = high) directly -- no shape parameter.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tables import DesireLevels

GRID = DesireLevels  # == IntimacyLevels; the 101-bin [0, 1] latent grid
# Beta pdf evaluation clamp: half a grid bin, so the exact-0/1 endpoints get
# the density of the nearest half-bin instead of ±inf when a or b < 1.
GRID_EPS = 0.005


def beta_prior_on_grid(m, nu, grid=None):
    """Discretized Beta prior over the latent grid.

    m: prior mean(s), shape (...,) in [0, 1]; nu: concentration (scalar,
    differentiable -- the fitted param). Returns normalized weights
    (..., n_grid). Exactly uniform at (m=0.5, nu=2), which nests the
    preregistered uniform prior.
    """
    g = GRID if grid is None else grid
    x = jnp.clip(g, GRID_EPS, 1.0 - GRID_EPS)
    m = jnp.asarray(m)[..., None]
    log_pdf = (nu * m - 1.0) * jnp.log(x) + (nu * (1.0 - m) - 1.0) * jnp.log1p(-x)
    return jax.nn.softmax(log_pdf, axis=-1)


def reweight_grid(post, w):
    """post (..., n_grid) uniform-prior posterior × w (..., n_grid) prior
    weights → renormalized informative-prior posterior."""
    p = post * w
    return p / p.sum(axis=-1, keepdims=True)


def reweight_joint(joint, w_latent=None, p_high=None):
    """(..., n_grid, 2) joint posterior reweighted by a grid-latent prior
    and/or a 2-state effort prior; None leaves that axis at the observer's
    uniform prior. Both None returns the input unchanged (the preregistered path)."""
    if w_latent is None and p_high is None:
        return joint
    p = joint
    if w_latent is not None:
        p = p * w_latent[..., :, None]
    if p_high is not None:
        w_eff = jnp.stack([1.0 - p_high, p_high], axis=-1)
        p = p * w_eff[..., None, :]
    return p / p.sum(axis=(-2, -1), keepdims=True)


# The given-relationship studies (1a/1b/3a) are the only ones with a
# relationship-free *base* priors vintage (`lm_priors_base.jsonl`, elicited
# without the relationship paragraph); `elicit_priors.py --base` refuses the
# given-desire studies (2a/2b/3b), which never show a relationship paragraph and
# whose base ablation shares the standard priors file. Single source of truth for
# the fit wrappers and the CV dispatcher.
GIVEN_RELATIONSHIP_SLUGS = frozenset(
    {"food_inv_desire", "food_inv_joint_de", "nonfood_inv_joint_de"}
)


def priors_base_variant(slug, variant, priors_file=None):
    """Whether to load the relationship-free *base* priors vintage for this
    (slug, variant) — the single decision both the fit wrappers and the CV
    dispatcher use so they never disagree.

    True only when all three hold:
      - `variant == "base"` (the relationship-invariant ablation);
      - `slug` is a given-relationship study (the only ones with a base priors
        vintage; the given-desire studies would route to a nonexistent file);
      - `priors_file is None` (no explicit `--priors-file`). An explicit priors
        file is a single materialized vintage (e.g. the human-ceiling
        `lm_priors_human.jsonl`, which is full-shaped, one row per relationship
        level) used as-is for every variant, so the loader's base collapse —
        which drops `intimacy_condition` — must not run on it.

    The `variant == "base"` test is an exact match, deliberately: the
    exploratory `base_shared` variant is base's *utility* scored against full's
    relationship-conditioned comparison set, so its cells carry a relationship
    axis and it needs the full-shaped priors. Widening this to
    `variant.startswith("base")` would hand it the collapsed vintage and
    misalign the prior with the cell grid.
    """
    return (
        variant == "base" and slug in GIVEN_RELATIONSHIP_SLUGS and priors_file is None
    )


def build_priors_kwarg(slug, config, base=False):
    """Assemble the fit helpers' `priors=` dict for one study/variant from the
    elicited prior tables and the RunConfig.

    Returns None in uniform mode (the preregistered byte-identical path). In
    informative mode it loads the study's per-run, per-cell prior scalars and
    keeps only the latents the study actually infers (mapping the grid latents
    desire/intimacy → `m_latent` and the 2-state effort latent → `p_effort`),
    producing the {"m_latent": ..., "p_effort": ...} dict the fit helpers
    reweight the observer posterior with. K-alignment against the alternatives
    tables is asserted by the caller (fit wrapper / CV dispatcher), which knows
    the feature tables' run count.

    Guards two silent-misconfiguration traps:
      - informative priors requested but the elicited file is missing →
        FileNotFoundError with the `make lm-priors-<slug>` hint (fail fast
        instead of silently falling back to uniform);
      - `--priors informative:<latent>` naming only latent(s) this study does
        NOT infer (so `active` is empty) → ValueError naming the study's
        inferred latents, rather than returning None and silently running the
        preregistered uniform fit while the user believes it is informative.
    """
    active = config.active_latents(slug)
    if not active:
        if config.priors_mode == "informative":
            from run_config import INFERRED_LATENTS

            raise ValueError(
                f"--priors informative:{','.join(config.priors_latents)} names no "
                f"latent that {slug} infers (it infers "
                f"{', '.join(INFERRED_LATENTS[slug])}); the requested prior would "
                "silently run the preregistered uniform fit. Name one of the study's "
                "inferred latents or drop the :latent qualifier."
            )
        return None
    from tables import load_lm_priors

    tables = load_lm_priors(slug, base=base, filename=config.priors_filename(base))
    if tables is None:
        raise FileNotFoundError(
            f"informative priors requested but "
            f"{config.priors_filename(base)} not found for {slug} — run "
            f"`make lm-priors-{'base-' if base else ''}{slug}` first."
        )
    out = {"m_latent": None, "p_effort": None}
    for lat in active:
        if lat in ("desire", "intimacy"):
            out["m_latent"] = tables[f"{lat}_m"]
        elif lat == "effort":
            out["p_effort"] = tables["effort_p"]
    return out
