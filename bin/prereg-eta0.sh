#!/usr/bin/env bash
#
# prereg-eta0.sh — fit + LOSO CV the PREREGISTERED model (eta = 0, no
# comparison-set reweighting) for all six inverse-planning studies.
#
# Why: the paper reports the reweighted fits and declares the reweighting a
# deviation from the preregistered specification, so the preregistered model's
# held-out numbers have to be reportable beside them (main.tex: the
# prereg-deviations section), computed by the same production fit and CV code
# as every other reported number.
#
# Every invocation passes --no-reweighting, which routes ALL outputs to
# model/outputs/<slug>/alt/uniform-noreweight/. The reported outputs under
# model/outputs/<slug>/ are never written by this script.
#
# Resumability: each study's fit and CV are skipped when their outputs are
# already present, and an interrupted CV resumes from its own fingerprint-guarded
# cv_checkpoint.jsonl inside the alt dir. So re-running after an interruption
# (or after a machine restart) continues rather than restarting.
#
# Usage:
#   bin/prereg-eta0.sh              # run it (backgroundable with nohup)
#   bin/prereg-eta0.sh --dry-run    # print the commands, run nothing
#
# Tunables (env): POOL CV_WORKERS OVERNIGHT_DIR
#   POOL       concurrent per-study chains (default 2)
#   CV_WORKERS CV worker processes per study (default 6; POOL x CV_WORKERS
#              should stay <= the core count, leaving a little headroom)
set -uo pipefail

# Re-exec under caffeinate so idle-sleep can't pause a multi-hour run.
if [ "${_CAFFEINATED:-}" != "1" ] && command -v caffeinate >/dev/null 2>&1; then
  case " $* " in
    *--chain*|*--dry-run*) : ;;
    *) export _CAFFEINATED=1; exec caffeinate -i "$0" "$@" ;;
  esac
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/bin/$(basename "$0")"
cd "$ROOT"

# Line-buffer the children's stdout. Without this a multi-hour run's per-fold
# progress sits in a block buffer and the logs stay empty until a study finishes,
# which makes the run unmonitorable exactly when monitoring matters.
export PYTHONUNBUFFERED=1

POOL="${POOL:-2}"
CV_WORKERS="${CV_WORKERS:-6}"
TAG="uniform-noreweight"
STUDIES=(
  food_inv_desire
  food_inv_joint_de
  food_inv_intimacy
  food_inv_joint_ie
  nonfood_inv_joint_de
  nonfood_inv_joint_ie
)

DRY=0
CHAIN=""
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --chain=*) CHAIN="${a#--chain=}" ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

RUN_DIR="${OVERNIGHT_DIR:-$ROOT/notes/prereg-eta0-$(date +%Y-%m-%d)}"
mkdir -p "$RUN_DIR"

run() {
  if [ "$DRY" = "1" ]; then echo "+ $*"; return 0; fi
  echo "+ $*"
  "$@"
}

out_dir() { echo "model/outputs/$1/alt/$TAG"; }

# --- one study's chain: fit then CV, each skipped when already complete --------
# A subshell, not a brace group: `exit 1` must end only the chain and hand its
# status to the OK/FAIL line below — inside a brace group it would terminate
# the whole script before that line, making the FAIL branch unreachable.
if [ -n "$CHAIN" ]; then
  s="$CHAIN"
  d="$(out_dir "$s")"
  log="$RUN_DIR/$s.log"
  (
    echo "=== $s :: preregistered (eta = 0) fit + CV — $(date) ==="
    if [ -f "$d/fit_results.json" ]; then
      echo "-- fit already present in $d — skipping"
    else
      run uv run python "model/inverse/fit_$s.py" --no-reweighting || exit 1
    fi
    if [ -f "$d/cv_trial_ll.jsonl" ]; then
      echo "-- CV already present in $d — skipping"
    else
      CV_WORKERS="$CV_WORKERS" run uv run python "model/cv/cv_$s.py" \
        --no-reweighting || exit 1
    fi
    echo "=== $s done — $(date) ==="
  ) >>"$log" 2>&1
  status=$?
  [ "$status" = "0" ] && echo "OK   $s" || echo "FAIL $s (see $log)"
  exit "$status"
fi

# --- driver -------------------------------------------------------------------
echo "============================================================"
echo "Preregistered model (eta = 0): fit + LOSO CV, all six studies"
echo "  outputs -> model/outputs/<slug>/alt/$TAG/  (reported outputs untouched)"
echo "  POOL=$POOL  CV_WORKERS=$CV_WORKERS"
echo "  logs    -> $RUN_DIR/"
echo "  started -> $(date)"
echo "============================================================"

if [ "$DRY" = "1" ]; then
  for s in "${STUDIES[@]}"; do echo "+ $SCRIPT --chain=$s"; done
  exit 0
fi

printf '%s\n' "${STUDIES[@]}" \
  | POOL="$POOL" CV_WORKERS="$CV_WORKERS" OVERNIGHT_DIR="$RUN_DIR" \
    xargs -P "$POOL" -I{} "$SCRIPT" --chain={}
status=$?

echo "============================================================"
echo "finished -> $(date)  (xargs status $status)"
for s in "${STUDIES[@]}"; do
  d="$(out_dir "$s")"
  if [ -f "$d/cv_trial_ll.jsonl" ]; then echo "  ✓ $s"; else echo "  ✗ $s (incomplete)"; fi
done
echo "Next: uv run python model/cv/model_comparison.py --study <slug> \\"
echo "        --compare-configs $TAG reported     # per study"
echo "============================================================"
exit "$status"
