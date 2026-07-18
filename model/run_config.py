"""Fit/CV run configuration: which priors the observer uses, which
alternatives vintage the LM tables come from, and where outputs go.

The canonical config (uniform priors, unsuffixed lm_runs.jsonl) reproduces the
preregistered pipeline byte-identically and keeps writing outputs/<slug>/.
Every other combination writes outputs/<slug>/alt/<tag>/ so vintages never
overwrite each other (spec: notes/2026-07-18-informative-priors-refusal-alts-design.md).
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

_ALTS_TAG = {"": "current"}  # any other suffix names itself, minus leading "_"


@dataclass(frozen=True)
class RunConfig:
    priors_mode: str = "uniform"  # "uniform" | "informative"
    priors_latents: tuple = ()  # () = all of the study's inferred latents
    alts_suffix: str = ""  # "" | "_refusal_hint" | "_refusal_hint_hyp"
    priors_file: str | None = None  # None = lm_priors.jsonl

    @classmethod
    def parse(cls, priors, alts_suffix, priors_file):
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
        return cls(mode, latents, alts_suffix or "", priors_file or None)

    @property
    def is_canonical(self):
        return self == RunConfig()

    def active_latents(self, slug):
        if self.priors_mode == "uniform":
            return ()
        inferred = INFERRED_LATENTS[slug]
        if not self.priors_latents:
            return inferred
        return tuple(lat for lat in inferred if lat in self.priors_latents)

    def tag(self):
        pr = self.priors_mode
        if self.priors_latents:
            pr += "-" + "-".join(self.priors_latents)
        alts = _ALTS_TAG.get(self.alts_suffix, self.alts_suffix.lstrip("_"))
        tag = f"{pr}_{alts}"
        if self.priors_file:
            tag += "_" + Path(self.priors_file).stem
        return tag

    def outputs_dir(self, slug):
        root = get_project_root() / "model" / "outputs" / slug
        return root if self.is_canonical else root / "alt" / self.tag()

    def runs_filename(self, base=False):
        return f"lm_runs{'_base' if base else ''}{self.alts_suffix}.jsonl"

    def priors_filename(self, base=False):
        if self.priors_file is not None:
            return self.priors_file
        return f"lm_priors{'_base' if base else ''}.jsonl"
