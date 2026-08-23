"""Fit/CV run configuration: which model the fit is, and where its outputs go.

The default config is the reported one: the surprise-weighted comparison-set
reweighting that `_reweighting.py` layers on wherever its scope rule applies.
It writes outputs/<slug>/. Every non-default config writes
outputs/<slug>/alt/<tag>/ instead, so an exploratory or comparison run can
never overwrite the reported baseline.

One axis moves off the default:

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

from utils import get_project_root


@dataclass(frozen=True)
class RunConfig:
    #: Drop the comparison-set reweighting → the preregistered model (eta = 0,
    #: with no eta parameter). The reweighting is otherwise applied by
    #: `_reweighting.config_for`'s scope rule, independently of this config.
    no_reweighting: bool = False

    @classmethod
    def parse(cls, no_reweighting=False):
        return cls(bool(no_reweighting))

    @property
    def is_default(self):
        """True for the default config — the reported one, which writes straight
        to outputs/<slug>/. Everything else writes under alt/<tag>/. This is a
        statement about the output location, not about the preregistration: the
        reported model layers the comparison-set reweighting on top of the
        preregistered specification (see `no_reweighting`)."""
        return self == RunConfig()

    def tag(self):
        # The preregistered config's tag keeps the historical "uniform-" prefix
        # (from when a priors axis existed): the committed
        # outputs/<slug>/alt/uniform-noreweight/ directories and the
        # --compare-configs consumers are keyed to it.
        return "uniform-noreweight" if self.no_reweighting else "reported"

    def outputs_dir(self, slug):
        root = get_project_root() / "model" / "outputs" / slug
        return root if self.is_default else root / "alt" / self.tag()
