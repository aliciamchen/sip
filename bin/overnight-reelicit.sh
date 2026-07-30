#!/usr/bin/env bash
#
# overnight-reelicit.sh — one-command overnight re-elicitation + refit of all six
# inverse-planning studies on the current prompt source, then fit + LOSO CV.
#
# Why: the committed lm_runs.jsonl tables predate the task-aware alternatives
# prompt and the total-dyadic-cost effort rubric, so every study's LM tables are
# stale relative to model/lm/prompts.py. This regenerates them at K=20 (main +
# base ablations), refits, and re-runs CV.
#
# What it does, in order:
#   0. preflight  — TOGETHER_API_KEY present, nothing else already running
#   1. smoke      — clear the gitignored diagnostic vintage, run K=1 elicitation
#                   of all 9 units there, then validate. HARD GATE: canonical
#                   tables remain untouched unless this passes
#   2. backup     — copy every canonical LM JSONL and provenance manifest to the
#                   run dir
#   3. delete     — remove the stale canonical generate/score artifact sets (a
#                   prompt edit makes the generator REFUSE to resume)
#   4. full       — K=20, pipelined: POOL concurrent per-study chains, each doing
#                   elicit(main[+base]) -> fit -> cv, so network-bound elicitation
#                   overlaps CPU-bound fit/CV instead of running after it
#   5. downstream — model_comparison.py + figures-results
#   6. validate   — final structural check of the K=20 tables
#
# Cost ~$51-73 (smoke ~$3 + full ~$48-70). Wall-clock ~4-5 h (the smoke calibrates
# it — multiply the smoke's elicitation time by ~15-20). Re-execs under
# `caffeinate -i` so idle-sleep can't pause it.
#
# Usage:
#   bin/overnight-reelicit.sh            # full run (prompts once for the $ gate)
#   bin/overnight-reelicit.sh --yes      # full run, no prompt (for nohup/background)
#   bin/overnight-reelicit.sh --dry-run  # print every command, spend/delete nothing
#   bin/overnight-reelicit.sh --smoke-only   # K=1 diagnostic smoke + validate
#   bin/overnight-reelicit.sh --resume-full  # continue a partial K=20 run in place
#
# Tunables (env): POOL SMOKE_JOBS CELL_WORKERS SCENARIO_WORKERS CV_WORKERS
#                 K_RUNS_FULL ALT_T OVERNIGHT_DIR
set -uo pipefail

# --- re-exec under caffeinate once (skip for chains / dry-run) -----------------
if [ "${_CAFFEINATED:-}" != "1" ] && command -v caffeinate >/dev/null 2>&1; then
  case " $* " in
    *--chain*|*--dry-run*) : ;;
    *) export _CAFFEINATED=1; exec caffeinate -i "$0" "$@" ;;
  esac
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/bin/$(basename "$0")"
cd "$ROOT"

# --- config (env-overridable) -------------------------------------------------
K_RUNS_FULL="${K_RUNS_FULL:-20}"
ALT_T="${ALT_T:-0.7}"
POOL="${POOL:-3}"                     # concurrent per-study chains (xargs -P)
SMOKE_JOBS="${SMOKE_JOBS:-3}"         # make -j for the K=1 smoke
CELL_WORKERS="${CELL_WORKERS:-12}"    # generation concurrency per study
SCENARIO_WORKERS="${SCENARIO_WORKERS:-3}"  # scoring concurrency per study (x~5 in flight)
CV_WORKERS="${CV_WORKERS:-5}"         # CV workers per study; x POOL(3) = 15 ~ 14 cores

MAIN_STUDIES="food_inv_desire food_inv_joint_de food_inv_intimacy food_inv_joint_ie nonfood_inv_joint_de nonfood_inv_joint_ie"
BASE_STUDIES="food_inv_desire food_inv_joint_de nonfood_inv_joint_de"
# Pipeline order: longest (and base-bearing) studies first, so their long CV
# overlaps the shorter studies' elicitation.
CHAIN_ORDER="food_inv_desire food_inv_joint_de nonfood_inv_joint_de food_inv_intimacy food_inv_joint_ie nonfood_inv_joint_ie"

# --- arg parsing --------------------------------------------------------------
MODE=full
DRY_RUN=false
ASSUME_YES=false
CHAIN_STUDY=""
usage() { sed -n '2,40p' "$SCRIPT" | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)     DRY_RUN=true ;;
    --yes|-y)      ASSUME_YES=true ;;
    --smoke-only)  MODE=smoke-only ;;
    --resume-full) MODE=resume-full ;;
    --chain)       MODE=chain; CHAIN_STUDY="${2:-}"; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

# --- helpers ------------------------------------------------------------------
is_base() { case " $BASE_STUDIES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
run()  { if $DRY_RUN; then printf 'DRY  %s\n' "$*"; else "$@"; fi; }
log()  { printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$RUN_DIR/main.log"; }

backup() {
  log "Backup lm/ tables -> $RUN_DIR/backup-lm/"
  for s in $MAIN_STUDIES; do
    if $DRY_RUN; then printf 'DRY  cp model/outputs/lm/%s/*.{jsonl,manifest.json} -> backup\n' "$s"; continue; fi
    mkdir -p "$RUN_DIR/backup-lm/$s"
    cp -p model/outputs/lm/"$s"/*.jsonl \
      model/outputs/lm/"$s"/*.manifest.json \
      "$RUN_DIR/backup-lm/$s/" 2>/dev/null || true
  done
}

# Remove ONLY the generate/score outputs — never the embedding artifacts
# (lm_alternatives_projection/semantic.jsonl) or the priors files.
delete_stale_files() {
  s="$1"
  for f in lm_alternatives.jsonl lm_alternatives.empty_units.jsonl \
             lm_alternatives.rationale.jsonl lm_alternatives.reasoning.jsonl \
             lm_alternatives.manifest.json lm_runs.jsonl \
             lm_runs.manifest.json lm_alternatives_base.jsonl \
             lm_alternatives_base.empty_units.jsonl \
             lm_alternatives_base.rationale.jsonl \
             lm_alternatives_base.reasoning.jsonl \
             lm_alternatives_base.manifest.json lm_runs_base.jsonl \
             lm_runs_base.manifest.json; do
    run rm -f "model/outputs/lm/$s/$f"
  done
}

delete_diag_files() {
  s="$1"
  for f in lm_alternatives_diag.jsonl lm_alternatives_diag.empty_units.jsonl \
             lm_alternatives_diag.rationale.jsonl \
             lm_alternatives_diag.reasoning.jsonl \
             lm_alternatives_diag.manifest.json lm_runs_diag.jsonl \
             lm_runs_diag.manifest.json lm_alternatives_base_diag.jsonl \
             lm_alternatives_base_diag.empty_units.jsonl \
             lm_alternatives_base_diag.rationale.jsonl \
             lm_alternatives_base_diag.reasoning.jsonl \
             lm_alternatives_base_diag.manifest.json \
             lm_runs_base_diag.jsonl lm_runs_base_diag.manifest.json; do
    run rm -f "model/outputs/lm/$s/$f"
  done
}

delete_diag() {
  log "Clear stale diagnostic generate/score artifact sets"
  for s in $MAIN_STUDIES; do
    delete_diag_files "$s"
  done
}

delete_stale() {
  log "Delete stale generate/score artifact sets (backup already taken)"
  for s in $MAIN_STUDIES; do
    delete_stale_files "$s"
  done
}

preflight() {
  log "Preflight"
  if $DRY_RUN; then log "DRY: skip key/process checks"; return 0; fi
  if ! grep -q TOGETHER_API_KEY .env 2>/dev/null; then
    echo "ERROR: TOGETHER_API_KEY not found in .env" >&2; exit 1
  fi
  if ps aux | grep -E "generate_alternatives|score_merged|fit_[a-z]|cv_[a-z]" | grep -v grep | grep -q .; then
    echo "ERROR: LM/fit/cv processes already running — aborting to avoid contention:" >&2
    ps aux | grep -E "generate_alternatives|score_merged|fit_[a-z]|cv_[a-z]" | grep -v grep >&2
    exit 1
  fi
}

# One study's full chain: elicit (main [+ base]) -> fit -> cv. Run by xargs, one
# per process, output redirected to its own study-<slug>.log.
chain_one() {
  s="$1"
  echo "$(date '+%H:%M:%S')  [$s] elicit main (K=$K_RUNS_FULL)"
  run make "lm-$s" K_RUNS="$K_RUNS_FULL" ALT_T="$ALT_T" CELL_WORKERS="$CELL_WORKERS" SCENARIO_WORKERS="$SCENARIO_WORKERS" \
    || { echo "[$s] elicit main FAILED"; return 1; }
  if is_base "$s"; then
    echo "$(date '+%H:%M:%S')  [$s] elicit base"
    run make "lm-base-$s" K_RUNS="$K_RUNS_FULL" ALT_T="$ALT_T" CELL_WORKERS="$CELL_WORKERS" SCENARIO_WORKERS="$SCENARIO_WORKERS" \
      || { echo "[$s] elicit base FAILED"; return 1; }
  fi
  echo "$(date '+%H:%M:%S')  [$s] fit"
  run make "fit-$s" || { echo "[$s] fit FAILED"; return 1; }
  echo "$(date '+%H:%M:%S')  [$s] cv (CV_WORKERS=$CV_WORKERS)"
  run make "cv-$s" CV_WORKERS="$CV_WORKERS" || { echo "[$s] cv FAILED"; return 1; }
  echo "$(date '+%H:%M:%S')  [$s] DONE"
}

phase_smoke() {
  log "K=1 diagnostic smoke: make -j$SMOKE_JOBS over all 9 elicitation units"
  targets=""
  for s in $MAIN_STUDIES; do targets="$targets lm-diag-$s"; done
  for s in $BASE_STUDIES; do targets="$targets lm-base-diag-$s"; done
  if $DRY_RUN; then
    printf 'DRY  make -j%s%s K_RUNS=1 ALT_T=%s CELL_WORKERS=%s SCENARIO_WORKERS=%s\n' \
      "$SMOKE_JOBS" "$targets" "$ALT_T" "$CELL_WORKERS" "$SCENARIO_WORKERS"
    log "DRY: skip smoke validation"
    return 0
  fi
  make -j"$SMOKE_JOBS" $targets K_RUNS=1 ALT_T="$ALT_T" CELL_WORKERS="$CELL_WORKERS" SCENARIO_WORKERS="$SCENARIO_WORKERS" \
    2>&1 | tee -a "$RUN_DIR/smoke.log"
  make_status="${PIPESTATUS[0]}"
  if [ "$make_status" -ne 0 ]; then
    log "SMOKE ELICITATION FAILED"
    return "$make_status"
  fi
  log "Validating diagnostic smoke output (K=1)"
  uv run python bin/_overnight_validate.py --k 1 --diag --studies $MAIN_STUDIES 2>&1 | tee -a "$RUN_DIR/smoke.log"
  validate_status="${PIPESTATUS[0]}"
  if [ "$validate_status" -ne 0 ]; then
    log "SMOKE VALIDATION FAILED"
    return "$validate_status"
  fi
  return 0
}

phase_full() {
  log "Full K=$K_RUNS_FULL run: $POOL concurrent study chains (elicit->fit->cv)"
  if $DRY_RUN; then
    for s in $CHAIN_ORDER; do
      printf 'DRY  chain %-24s make lm-%s' "$s" "$s"
      is_base "$s" && printf ' + make lm-base-%s' "$s"
      printf ' -> make fit-%s -> make cv-%s (CV_WORKERS=%s)\n' "$s" "$s" "$CV_WORKERS"
    done
    log "DRY: would run the above, $POOL at a time via xargs -P$POOL"
    return 0
  fi
  export RUN_DIR K_RUNS_FULL ALT_T CELL_WORKERS SCENARIO_WORKERS CV_WORKERS BASE_STUDIES _CAFFEINATED
  if ! printf '%s\n' $CHAIN_ORDER | xargs -P "$POOL" -I{} bash -c \
    '"$0" --chain "$1" >>"$RUN_DIR/study-$1.log" 2>&1
     status=$?
     if [ "$status" -ne 0 ]; then
       echo "CHAIN FAILED: $1" | tee -a "$RUN_DIR/main.log"
     fi
     exit "$status"' \
    "$SCRIPT" {}; then
    log "FULL RUN FAILED — partial outputs retained for --resume-full"
    return 1
  fi
  log "All study chains returned (per-study detail in study-*.log)"
}

phase_downstream() {
  log "Downstream: model_comparison + figures-results"
  if ! run uv run python model/cv/model_comparison.py; then
    log "DOWNSTREAM FAILED — model_comparison"
    return 1
  fi
  if ! run make figures-results; then
    log "DOWNSTREAM FAILED — figures-results"
    return 1
  fi
}

phase_finalcheck() {
  log "Final validation (K=$K_RUNS_FULL)"
  $DRY_RUN && return 0
  uv run python bin/_overnight_validate.py --k "$K_RUNS_FULL" --studies $MAIN_STUDIES 2>&1 | tee -a "$RUN_DIR/main.log"
  validate_status="${PIPESTATUS[0]}"
  if [ "$validate_status" -ne 0 ]; then
    log "FINAL VALIDATION FAILED"
    return "$validate_status"
  fi
  return 0
}

confirm() {
  cat <<EOF

============================================================
  OVERNIGHT RE-ELICITATION + REFIT  (all six studies, tight prompt)
  Mode:            $MODE$($DRY_RUN && echo '  [DRY-RUN]')
  Est. cost:       ~\$51-73   (K=1 smoke ~\$3 + K=20 full ~\$48-70)
  Est. wall-clock: ~4-5 h     (pipelined; smoke calibrates it)
  Logs + backup:   $RUN_DIR
============================================================
EOF
  { $ASSUME_YES || $DRY_RUN; } && return 0
  printf "Proceed with the PAID run? [y/N] "
  read -r ans
  case "$ans" in [yY]*) ;; *) echo "Aborted."; exit 1 ;; esac
}

# --- chain subcommand (invoked by xargs; RUN_DIR etc. inherited) --------------
if [ "$MODE" = chain ]; then
  [ -n "$CHAIN_STUDY" ] || { echo "--chain needs a study"; exit 2; }
  chain_one "$CHAIN_STUDY"
  exit $?
fi

# --- top-level flow -----------------------------------------------------------
if $DRY_RUN; then
  RUN_DIR="${TMPDIR:-/tmp}/overnight-dryrun-$$"   # keep dry-runs out of $HOME
else
  RUN_DIR="${OVERNIGHT_DIR:-$HOME/sip-overnight-runs}/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$RUN_DIR"
export RUN_DIR
log "run dir: $RUN_DIR   (mode=$MODE dry=$DRY_RUN)"
log "config: POOL=$POOL CELL_WORKERS=$CELL_WORKERS SCENARIO_WORKERS=$SCENARIO_WORKERS CV_WORKERS=$CV_WORKERS K=$K_RUNS_FULL"

case "$MODE" in
  full)
    confirm
    preflight
    delete_diag
    if phase_smoke; then log "SMOKE PASSED"; else
      log "SMOKE FAILED — canonical tables remain untouched"
      exit 1
    fi
    backup
    delete_stale
    if ! phase_full; then
      log "ABORTED after a study-chain failure; use --resume-full after diagnosis"
      exit 1
    fi
    if ! phase_downstream; then
      log "ABORTED after a downstream failure"
      exit 1
    fi
    if ! phase_finalcheck; then
      log "ABORTED after final validation"
      exit 1
    fi
    log "COMPLETE. Review $RUN_DIR and commit the LM tables + model outputs when satisfied."
    ;;
  smoke-only)
    confirm
    preflight
    delete_diag
    if phase_smoke; then
      log "SMOKE PASSED; diagnostic tables retained for inspection"
    else
      log "SMOKE FAILED; canonical tables remain untouched"
      exit 1
    fi
    ;;
  resume-full)
    preflight
    log "RESUME: skipping backup/smoke/delete — continuing the partial K=$K_RUNS_FULL run"
    log "(elicitation resumes via the scripts' own logic; CV resumes from its checkpoint)"
    if ! phase_full; then
      log "RESUME FAILED during a study chain"
      exit 1
    fi
    if ! phase_downstream; then
      log "RESUME FAILED during downstream outputs"
      exit 1
    fi
    if ! phase_finalcheck; then
      log "RESUME FAILED final validation"
      exit 1
    fi
    log "RESUME COMPLETE. Review $RUN_DIR."
    ;;
  *) echo "bad mode: $MODE" >&2; exit 2 ;;
esac
