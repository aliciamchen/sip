"""Reading the per-run held-out belief updates of a CV run.

`cv_preds_summary.json` stores each held-out cell's predicted belief update as a
mean over the K elicitation runs, and (since 2026-08-03) the K per-run values
`delta_*_runs` behind it. Those per-run values are the elicitation mixture's own
components, so anything that reasons about the mixture's spread needs them.

CV runs written before that date carry only the means and instead have a
`cv_run_deltas.json` sidecar holding the recovered per-run values (committed
next to those outputs; a fresh CV run writes the per-run values natively).
Consumers therefore have to try two sources in a fixed order and check the
sidecar's provenance, which is the logic this module exists to hold once — a
second copy of a staleness check is a second place for it to be subtly wrong.

Deliberately stdlib + numpy only -- no jax, no `_helpers` -- exactly as
`_checkpoint.py` is, so a figure script can import it without paying for a JAX
import or coupling itself to the model code.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

#: The per-run-delta sidecar kept next to a pre-2026-08-03 study's CV outputs.
OUTPUT_NAME = "cv_run_deltas.json"

#: The inferred variable (`Study.dvs[...].name`) that is the two-state physical
#: world state rather than a continuous latent on the 101-bin grid. Its belief
#: update is a probability difference on a two-point support, so its per-run
#: spread behaves differently and is reported apart from the continuous latents'.
WORLD_STATE_DV = "effort"


class RunDeltasUnavailable(Exception):
    """A study's per-run deltas cannot be read. Carries a message naming the
    command that would produce them. Raised rather than returning None so a
    caller chooses the policy: the SI figure skips the study, while the
    results-LaTeX exporter lets it propagate (a silently missing macro would
    surface as an undefined control sequence far from the cause)."""


def sha256_file(path):
    """Hex SHA-256 of a file's bytes. Local rather than imported from
    `_helpers` to keep this module JAX-free."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_per_run_deltas(outputs_dir, variant="full"):
    """The per-run held-out belief updates for one (study, variant).

    Returns `({delta_col: (n_cells, K) array}, sigma)`, where sigma is the
    variant's fitted response noise from `fit_results.json` -- the scale the
    spread is only interpretable against.

    Two sources, in order of authority:

      1. `cv_preds_summary.json`, when its rows carry `<delta>_runs`. Those are
         the CV run's own per-run values, written by the fold bodies.
      2. `cv_run_deltas.json`, the recomputed sidecar. It records the SHA-256 of
         the `cv_preds_summary.json` it was gated against; a mismatch means CV
         has been re-run since, so this raises instead of returning one vintage's
         spread to be read against another vintage's sigma.

    Raises `RunDeltasUnavailable` when neither source is usable.
    """
    outputs_dir = Path(outputs_dir)
    preds_path = outputs_dir / "cv_preds_summary.json"
    fit_path = outputs_dir / "fit_results.json"
    for path in (preds_path, fit_path):
        if not path.exists():
            raise RunDeltasUnavailable(f"{path} missing -- run the study's fit and CV")

    fits = json.loads(fit_path.read_text())
    fit = next((r for r in fits if r["model"] == variant), None)
    if fit is None:
        raise RunDeltasUnavailable(f"{fit_path} has no `{variant}` fit")
    sigma = float(fit["param_sigma"])

    rows = [r for r in json.loads(preds_path.read_text()) if r["model"] == variant]
    if not rows:
        raise RunDeltasUnavailable(f"{preds_path} has no `{variant}` rows")
    run_keys = [k for k in rows[0] if k.startswith("delta_") and k.endswith("_runs")]
    if run_keys:
        return {
            k[: -len("_runs")]: np.array([r[k] for r in rows]) for k in run_keys
        }, sigma

    side_path = outputs_dir / OUTPUT_NAME
    if not side_path.exists():
        raise RunDeltasUnavailable(
            f"{preds_path.name} has no per-run deltas and there is no "
            f"{OUTPUT_NAME} sidecar -- re-run the study's CV"
        )
    side = json.loads(side_path.read_text())
    if side.get("variant") != variant:
        raise RunDeltasUnavailable(
            f"{side_path} holds variant `{side.get('variant')}`, not `{variant}`"
        )
    if side.get("source", {}).get("cv_preds_summary.json") != sha256_file(preds_path):
        raise RunDeltasUnavailable(
            f"{side_path.name} was gated against a different "
            f"{preds_path.name} than the one on disk -- CV has been re-run since, "
            f"so this sidecar is stale. Regenerate with `make run-deltas`"
        )
    cells = side["cells"]
    return {
        k: np.array([c[f"{k}_runs"] for c in cells]) for k in side["delta_keys"]
    }, sigma
