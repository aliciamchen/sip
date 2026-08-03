"""Fit/CV run configuration: which model the fit is, and where its outputs go.

The default config is the reported one: uniform priors, plus the
surprise-weighted comparison-set reweighting that `_reweighting.py` layers on
wherever its scope rule applies. It writes outputs/<slug>/. Every non-default
config writes outputs/<slug>/alt/<tag>/ instead, so an exploratory or
comparison run can never overwrite the reported baseline.

Two axes move off the default:

  - `priors_mode="informative"` — the "build both" priors comparison, evaluated
    and not adopted (see the rules file); the machinery stays as tooling.
  - `no_reweighting=True` — drop the reweighting, giving the **preregistered**
    model (eta = 0, and no eta parameter at all). The paper reports the
    reweighted fits and declares the reweighting a deviation, so the
    preregistered model's held-out numbers have to be reportable alongside them.

Note on naming: the default used to be called "canonical" and then
"preregistered". Neither was right — it is the *reported* config, and the
reported model is not the preregistered one, which is exactly what
`no_reweighting` now names. `is_default` therefore means "writes to the reported
output directory", nothing more.
"""

from dataclasses import dataclass
from pathlib import Path

from utils import get_project_root

# The latent(s) each study infers (grid latent first). "effort" is the 2-state
# world latent of the joint studies; desire/intimacy are 101-bin grid latents.
INFERRED_LATENTS = {
    "food_inv_desire": ("desire",),
    "food_inv_joint_de": ("desire", "effort"),
    "food_inv_intimacy": ("intimacy",),
    "food_inv_joint_ie": ("intimacy", "effort"),
    "nonfood_inv_joint_de": ("desire", "effort"),
    "nonfood_inv_joint_ie": ("intimacy", "effort"),
}
GRID_LATENTS = ("desire", "intimacy")


@dataclass(frozen=True)
class RunConfig:
    priors_mode: str = "uniform"  # "uniform" | "informative"
    priors_latents: tuple = ()  # () = all of the study's inferred latents
    priors_file: str | None = None  # None = lm_priors.jsonl
    #: Drop the comparison-set reweighting → the preregistered model (eta = 0,
    #: with no eta parameter). The reweighting is otherwise applied by
    #: `_reweighting.config_for`'s scope rule, independently of this config.
    no_reweighting: bool = False

    @classmethod
    def parse(cls, priors, priors_file, no_reweighting=False):
        mode, _, latents = (priors or "uniform").partition(":")
        if mode not in ("uniform", "informative"):
            raise ValueError(
                f"--priors must be uniform|informative[...], got {priors!r}"
            )
        latents = tuple(s for s in latents.split(",") if s) if latents else ()
        valid = {lat for lats in INFERRED_LATENTS.values() for lat in lats}
        unknown = set(latents) - valid
        if unknown:
            raise ValueError(f"unknown latent(s) in --priors: {sorted(unknown)}")
        if mode == "uniform" and latents:
            raise ValueError("--priors uniform takes no :latents suffix")
        return cls(mode, latents, priors_file or None, bool(no_reweighting))

    @property
    def is_default(self):
        """True for the default config — the reported one, which writes straight
        to outputs/<slug>/. Everything else writes under alt/<tag>/. This is a
        statement about the output location, not about the preregistration: the
        reported model layers the comparison-set reweighting on top of the
        preregistered specification (see `no_reweighting`)."""
        return self == RunConfig()

    def active_latents(self, slug):
        if self.priors_mode == "uniform":
            return ()
        inferred = INFERRED_LATENTS[slug]
        if not self.priors_latents:
            return inferred
        return tuple(lat for lat in inferred if lat in self.priors_latents)

    def tag(self):
        tag = self.priors_mode
        if self.priors_latents:
            tag += "-" + "-".join(self.priors_latents)
        if self.no_reweighting:
            tag += "-noreweight"
        if self.priors_file:
            tag += "_" + Path(self.priors_file).stem
        return tag

    def outputs_dir(self, slug):
        root = get_project_root() / "model" / "outputs" / slug
        return root if self.is_default else root / "alt" / self.tag()

    def priors_filename(self, base=False):
        if self.priors_file is not None:
            return self.priors_file
        return f"lm_priors{'_base' if base else ''}.jsonl"
