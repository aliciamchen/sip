#!/usr/bin/env bash
#
# regenerate-vintage.sh — re-run every fitted/derived output on the current
# code, producing ONE new self-consistent output vintage.
#
# Why: refactors that change the compiled computation graph (even when the
# math is identical) shift float32 results by ~1 ulp, and the repo's rule is
# that outputs are never mixed across such changes — so after a batch of
# graph-touching refactors, everything downstream of the fits is regenerated
# together: reported fits, LOSO CV, model comparison, the preregistered
# (uniform-noreweight) arm, the exploratory arms (transfer/pooled/
# generalization), and the LaTeX/figure exports.
#
# It never touches the LM elicitation tables (outputs/lm/**) — those are paid
# inputs, not derived outputs, and nothing here re-elicits.
#
# Everything deleted here is committed, so the pre-regeneration vintage stays
# recoverable from git until the new outputs are reviewed and committed.
# Acceptance gate: bin/compare_vintage.py diffs the new headline numbers
# against HEAD at reported precision.
#
# Usage:
#   bin/regenerate-vintage.sh                 # all stages, in order
#   bin/regenerate-vintage.sh cv comparison   # just these stages
#   bin/regenerate-vintage.sh --dry-run       # print the stage commands only
#
# Stages: clean fits cv comparison prereg arms export figures verify
#
# Interruptions are cheap: make's file targets skip completed studies, CV
# resumes from its fingerprint-guarded checkpoint, and prereg-eta0.sh skips
# studies whose alt outputs already exist — re-run the script (or the stage)
# to continue.

set -uo pipefail

# Re-exec under caffeinate so idle-sleep can't pause a multi-hour run.
if [ "${_CAFFEINATED:-}" != "1" ] && command -v caffeinate >/dev/null 2>&1; then
  case " $* " in
    *--dry-run*) : ;;
    *) export _CAFFEINATED=1; exec caffeinate -i "$0" "$@" ;;
  esac
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DRY=0
STAGES=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    *) STAGES+=("$a") ;;
  esac
done
[ ${#STAGES[@]} -eq 0 ] && STAGES=(clean fits cv comparison prereg arms export figures verify)

LOGDIR="notes/regenerate-vintage-logs"
mkdir -p "$LOGDIR"

# Fatal: a failed command aborts its stage immediately. Stage functions run
# on the left side of stage()'s tee pipeline (a subshell), so `exit 1` here
# fails that pipeline (pipefail) and stage() reports the failure — an
# intermediate command's failure is never swallowed by a later one succeeding.
run() {
  echo "+ $*"
  [ "$DRY" = 1 ] && return 0
  "$@" || { echo "FAILED: $*" >&2; exit 1; }
}

stage() {
  local name="$1"; shift
  echo "============================================================"
  echo "stage: $name  ($(date))"
  echo "============================================================"
  if [ "$DRY" = 1 ]; then "$@"; return; fi
  local log="$LOGDIR/$name.log"
  if "$@" 2>&1 | tee "$log"; then
    echo "stage $name OK"
  else
    echo "stage $name FAILED — see $log" >&2
    exit 1
  fi
}

stage_clean() {
  run make clean
  # Derived/exploratory outputs of the old vintage (all committed; the fresh CV
  # writes delta_*_runs natively, so the cv_run_deltas.json compatibility
  # sidecars are retired rather than regenerated).
  run rm -rf model/outputs/*/alt
  run rm -rf model/outputs/transfer model/outputs/pooled
  run rm -f model/outputs/generalization_primary.json model/outputs/group_correlations.json
  run rm -f model/outputs/*/cv_run_deltas.json
}

stage_prereg() {
  run bin/prereg-eta0.sh
  # Roster from study_registry (the single source of truth), not a hand copy.
  for s in $(uv run python -c "from study_registry import SLUGS; print(' '.join(SLUGS))"); do
    run uv run python model/cv/model_comparison.py --study "$s" \
      --compare-configs uniform-noreweight reported
  done
}

stage_arms() {
  # transfer and pooled are independent (each depends only on the per-study CV
  # outputs); run them concurrently — everything below needs both.
  run make -j2 transfer pooled
  # The pooled-food-to-nonfood transfer arm (alt/transfer-pooled-food-refit) is
  # a separate transfer.py mode, not one of the eight designed pairs — the
  # manuscript's \rNonfoodFoodFit and generalization_primary's `food` arm read
  # it, so the vintage is incomplete without it.
  run uv run python model/cv/transfer.py --from-pooled food --to 3a --to 3b
  run make generalization-primary
}

stage_figures() {
  run make figures-panels
  run make figures-lm-si
  run make figures-si-prior-posterior
  run make figures-si-prereg-predictions
  run make figures-nonfood-domains
  run make sync-journal-figures
}

for s in "${STAGES[@]}"; do
  case "$s" in
    clean)      stage clean       stage_clean ;;
    fits)       stage fits        run make fit ;;
    cv)         stage cv          run make cv ;;
    comparison) stage comparison  run make model-comparison ;;
    prereg)     stage prereg      stage_prereg ;;
    arms)       stage arms        stage_arms ;;
    export)     stage export      run make results-latex ;;
    figures)    stage figures     stage_figures ;;
    verify)     stage verify      run uv run python bin/compare_vintage.py ;;
    *) echo "unknown stage: $s (stages: clean fits cv comparison prereg arms export figures verify)" >&2; exit 2 ;;
  esac
done

echo "============================================================"
echo "regeneration complete ($(date)) — review bin/compare_vintage.py's report,"
echo "then commit the new outputs as one vintage."
echo "============================================================"
