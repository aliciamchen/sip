"""Unit tests for RunConfig (fit/CV configuration: priors mode, output layout)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_config import INFERRED_LATENTS, RunConfig
from utils import get_project_root


def test_canonical_default():
    c = RunConfig()
    assert c.is_canonical
    assert c.outputs_dir("food_inv_desire") == (
        get_project_root() / "model" / "outputs" / "food_inv_desire"
    )
    assert c.priors_filename(base=False) == "lm_priors.jsonl"
    assert c.priors_filename(base=True) == "lm_priors_base.jsonl"
    assert c.active_latents("food_inv_joint_de") == ()
    print("✓ canonical default config keeps the preregistered layout")


def test_parse_informative_all_latents():
    c = RunConfig.parse("informative", None)
    assert not c.is_canonical
    assert c.tag() == "informative"
    assert c.active_latents("food_inv_joint_de") == ("desire", "effort")
    assert c.active_latents("food_inv_intimacy") == ("intimacy",)
    assert c.priors_filename(base=False) == "lm_priors.jsonl"
    assert c.priors_filename(base=True) == "lm_priors_base.jsonl"
    assert c.outputs_dir("food_inv_joint_de") == (
        get_project_root()
        / "model"
        / "outputs"
        / "food_inv_joint_de"
        / "alt"
        / "informative"
    )
    print("✓ informative priors infer all latents and write to alt/<tag>/")


def test_parse_latent_subset():
    c = RunConfig.parse("informative:desire", None)
    assert c.tag() == "informative-desire"
    assert c.active_latents("food_inv_joint_de") == ("desire",)
    print("✓ latent subset composes into the tag")


def test_custom_priors_file():
    c = RunConfig.parse("informative", "lm_priors_human.jsonl")
    assert c.priors_filename(base=False) == "lm_priors_human.jsonl"
    assert c.tag() == "informative_lm_priors_human"
    print("✓ custom priors file names itself in filename and tag")


def test_parse_rejects_unknown_mode_and_latent():
    try:
        RunConfig.parse("bogus", None)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown priors mode was not rejected")
    try:
        RunConfig.parse("informative:sharing", None)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown latent was not rejected")
    print("✓ parse rejects unknown priors mode and unknown latent")


def test_inferred_latents_cover_all_six_studies():
    assert set(INFERRED_LATENTS) == {
        "food_inv_desire",
        "food_inv_joint_de",
        "food_inv_intimacy",
        "food_inv_joint_ie",
        "nonfood_inv_joint_de",
        "nonfood_inv_joint_ie",
    }
    print("✓ INFERRED_LATENTS covers all six studies")


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
