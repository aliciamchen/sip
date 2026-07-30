"""Fit/CV run configuration: which priors the observer uses, and where outputs
go.

The default config (uniform priors) reproduces the **preregistered** pipeline
byte-identically and keeps writing outputs/<slug>/. The informative-prior
configs (the "build both" priors comparison) write outputs/<slug>/alt/<tag>/ so
they never overwrite the preregistered baseline.

Note on naming: this used to be called the "canonical" config, which read as
"the authoritative one". It isn't — the fits reported in the paper add the
surprise-weighted comparison-set reweighting on top of this config, so
"canonical" pointed at the wrong model. It means *preregistered*: uniform
priors, as every prereg specifies.
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

    @classmethod
    def parse(cls, priors, priors_file):
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
        return cls(mode, latents, priors_file or None)

    @property
    def is_preregistered(self):
        """True for the default (uniform-prior) config — the one the preregs
        specify. Not "the reported model": the paper's fits layer the
        comparison-set reweighting on top of this."""
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
        if self.priors_file:
            tag += "_" + Path(self.priors_file).stem
        return tag

    def outputs_dir(self, slug):
        root = get_project_root() / "model" / "outputs" / slug
        return root if self.is_preregistered else root / "alt" / self.tag()

    def priors_filename(self, base=False):
        if self.priors_file is not None:
            return self.priors_file
        return f"lm_priors{'_base' if base else ''}.jsonl"
