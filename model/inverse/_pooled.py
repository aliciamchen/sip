"""Pooled cross-experiment fits: one utility, several experiments.

Every reported fit estimates one experiment's parameters from that experiment's
own data, so six experiments give six utilities. This module fits a SHARED set
of utility weights across a group of experiments, while each experiment keeps
its own response parameters. It is the direct form of the question the
cross-study transfer analysis (`model/cv/transfer.py`) answers indirectly: not
"do experiment A's weights happen to work on B" but "is one utility enough for
both".

WHAT POOLS AND WHAT DOES NOT. The utility weights (`w_v`, `w_d`, `w_e`, gamma)
are shared; `alpha_observer`, `sigma` and `eta` stay per experiment. That split
is not a convenience -- it is what the transfer analysis measured. Freezing a
donor's whole vector cost 0.005-0.18 nats/trial; freezing only the utility and
re-estimating the response layer cost 0.000-0.035. `alpha_observer` indexes a
posterior over a different latent space in each experiment and trades off
against the overall weight scale, `sigma` is on a 1-D DV in some experiments
and a 2-D one in others, and `eta`'s very scope is defined per experiment by
`_reweighting.STUDY_CONTRASTIVE`. None of the three is a claim anyone makes
about the actor, so forcing them equal would test something no one believes.

The utility, by contrast, is a claim about the ACTOR, and which latent the
observer infers is a property of the OBSERVER's task. The same actor is being
reasoned about in every experiment, so a shared utility is a coherence
requirement rather than a hopeful hypothesis -- with one caveat worth keeping:
the LM-elicited comparison sets differ across experiments (they are conditioned
on whatever the observer can see), so a pooled fit that loses badly implicates
either the utility or those alternative sets.

PARAMETER VECTOR

    [*utility, *(alpha_observer_s, sigma_s, eta_s?) for s in group]

The utility block is shared; each experiment then contributes its own response
block, in group order, carrying an `eta` only where the reweighting scope rule
grants it one. `study_slice` rebuilds a single experiment's ordinary fit vector
(`[*utility, alpha_observer, sigma, eta?]`) out of the pooled one, which is what
lets the existing per-study scoring code do the scoring unchanged.

Uniform priors only: an informative-prior configuration would add a `prior_nu`
per experiment, and nothing here has been checked against that.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _helpers import (  # noqa: E402
    ALPHA_OBS_MAX,
    _desire_loss,
    _intimacy_loss,
    _joint_de_loss,
    _joint_ie_loss,
    fit_masked,
)

#: The loss factory for each observer family, keyed as in the fit and CV
#: dispatchers' `_FAMILIES` registries.
LOSS_FACTORY = {
    "desire": _desire_loss,
    "intimacy": _intimacy_loss,
    "joint_de": _joint_de_loss,
    "joint_ie": _joint_ie_loss,
}


@dataclass(frozen=True)
class PooledLayout:
    """Where each experiment's parameters sit in the pooled vector.

    `n_utility` shared slots first, then one response block per experiment in
    `slugs` order. `blocks[i] = (start, has_eta)` locates experiment i's
    (alpha_observer, sigma, eta?) run.
    """

    slugs: tuple
    n_utility: int
    blocks: tuple  # ((start, has_eta), ...) parallel to slugs

    @property
    def n_params(self):
        start, has_eta = self.blocks[-1]
        return start + 2 + int(has_eta)

    def study_slice(self, params, i):
        """Experiment i's ordinary fit vector, carved out of the pooled one:
        the shared utility followed by that experiment's own response block."""
        start, has_eta = self.blocks[i]
        end = start + 2 + int(has_eta)
        return jnp.concatenate([params[: self.n_utility], params[start:end]])

    def param_names(self, utility_param_names):
        """Human-readable name per pooled slot, for provenance and diagnostics."""
        names = list(utility_param_names)
        for slug, (_, has_eta) in zip(self.slugs, self.blocks):
            names += [f"{slug}:alpha_observer", f"{slug}:sigma"]
            if has_eta:
                names.append(f"{slug}:eta")
        return names


def build_layout(slugs, n_utility, has_eta):
    """Lay out the pooled vector for a group. `has_eta` is a per-slug bool --
    whether the reweighting scope rule grants that (experiment, variant) an
    eta; it is decided per experiment, so the blocks are not uniform."""
    blocks, cursor = [], n_utility
    for slug in slugs:
        blocks.append((cursor, bool(has_eta[slug])))
        cursor += 2 + int(has_eta[slug])
    return PooledLayout(tuple(slugs), int(n_utility), tuple(blocks))


def pooled_loss(layout, study_losses):
    """Sum the group's per-experiment NLLs under one shared utility.

    `study_losses` is one loss closure per slug, in `layout.slugs` order, each
    built by that experiment's `LOSS_FACTORY` entry over whichever trials this
    fit is training on. Each is called on its OWN fit vector, rebuilt from the
    pooled one -- so the pooled objective is exactly the sum of the same
    likelihoods the per-experiment fits maximize, with the utility tied.

    The sum weights experiments by trial count and by DV dimensionality (a
    2-slider experiment's log densities are systematically larger in magnitude
    than a 1-slider experiment's). That is the honest joint likelihood, but it
    does mean the fit leans toward the larger experiments -- which is why every
    caller reports per-experiment held-out likelihood rather than this total.
    """

    def loss(params):
        total = 0.0
        for i, study_loss in enumerate(study_losses):
            total = total + study_loss(layout.study_slice(params, i))
        return total

    return loss


def pooled_init(layout, per_study_params):
    """A starting vector from the group's existing per-experiment fits: the
    trial-count-free mean of their utility weights, and each experiment's own
    response parameters kept as they are.

    The mean is a neutral starting point rather than a claim -- the fit moves
    off it -- but starting from one experiment's weights would make the result
    depend on which experiment was listed first.
    """
    utils = np.mean(
        [p[: layout.n_utility] for p in per_study_params], axis=0
    )  # (n_utility,)
    out = list(utils)
    for i, (_, has_eta) in enumerate(layout.blocks):
        p = per_study_params[i]
        out += [p[layout.n_utility], p[layout.n_utility + 1]]
        if has_eta:
            out.append(p[-1])
    return np.asarray(out, dtype=float)


def pooled_upper(layout):
    """Elementwise upper bounds for the pooled vector, or None when no bound is
    active (the default -- see `ALPHA_OBS_MAX`). The single-slot
    `param_upper_bounds` does not apply here: the pooled vector carries one
    alpha_observer per experiment, so the cap goes on every response block."""
    if ALPHA_OBS_MAX is None:
        return None
    upper = np.full(layout.n_params, np.inf)
    for start, _ in layout.blocks:
        upper[start] = float(ALPHA_OBS_MAX)
    return jnp.asarray(upper)


def fit_pooled(
    layout,
    study_losses,
    init_params,
    free_mask=None,
    n_restarts=1,
    lr=0.1,
    max_steps=1000,
    patience=100,
    verbose=False,
    seed_key=None,
):
    """Fit the pooled vector. Returns (params, nll, restarts).

    No `alpha_obs_index` is passed: the pooled vector has one alpha_observer per
    experiment, so `_fit_multistart`'s single-slot basin seeding does not apply.
    Callers that want basin coverage vary the init explicitly (see
    `model/cv/pooled.py`), which is also what the transfer analysis does.
    """
    if len(init_params) != layout.n_params:
        # `fit_masked` only length-checks its masked path, and `study_slice`
        # would hand the trailing experiment a short vector that JAX then
        # indexes with clamping -- a plausible-looking but wrong fit.
        raise ValueError(
            f"init_params has length {len(init_params)} but this group's pooled "
            f"vector has {layout.n_params} slots "
            f"({layout.n_utility} shared + {len(layout.slugs)} response blocks)"
        )
    return fit_masked(
        loss_fn=pooled_loss(layout, study_losses),
        n_params=layout.n_params,
        free_mask=free_mask,
        init_params=init_params,
        n_restarts=n_restarts,
        lr=lr,
        max_steps=max_steps,
        patience=patience,
        verbose=verbose,
        seed_key=seed_key,
        label="pooled",
        upper=pooled_upper(layout),
    )
