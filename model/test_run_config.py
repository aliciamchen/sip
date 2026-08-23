"""Unit tests for RunConfig (fit/CV configuration: output layout)."""

from model.run_config import RunConfig
from utils import get_project_root


def test_default_config_layout():
    c = RunConfig()
    assert c.is_default
    assert not c.no_reweighting
    assert c.outputs_dir("food_inv_desire") == (
        get_project_root() / "model" / "outputs" / "food_inv_desire"
    )
    print("✓ default config keeps the reported output layout")


def test_no_reweighting_is_named_and_routed_off_the_root():
    """The preregistered model (no comparison-set reweighting) must never write
    where the reported fits live, and its tag must keep the name the committed
    alt/ directories and --compare-configs consumers are keyed to."""
    c = RunConfig.parse(True)
    assert c.no_reweighting
    assert not c.is_default, "the preregistered run must not claim the study root"
    assert c.tag() == "uniform-noreweight"
    assert c.outputs_dir("food_inv_joint_de") == (
        get_project_root()
        / "model"
        / "outputs"
        / "food_inv_joint_de"
        / "alt"
        / "uniform-noreweight"
    )
    print("✓ --no-reweighting names itself in the tag and writes under alt/")


if __name__ == "__main__":
    print("=" * 60)
    print("RunConfig tests")
    print("=" * 60)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
