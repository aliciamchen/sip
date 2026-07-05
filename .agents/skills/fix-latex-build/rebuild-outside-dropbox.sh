#!/usr/bin/env bash
# rebuild-outside-dropbox.sh — repair a corrupted LaTeX build in a Dropbox-synced
# project. It (1) diagnoses the corruption, (2) clears the biber PAR cache, and
# (3) rebuilds in a scratch dir OUTSIDE Dropbox and copies the good artifacts
# back, sidestepping the local-latexmk-vs-Dropbox race that keeps breaking the
# in-place build. General over LaTeX projects; defaults to SIP_journal.
#
# Usage:
#   rebuild-outside-dropbox.sh [--check] [--force] [PROJECT_DIR] [MAIN_TEX]
#     --check      diagnose only; do not clear the cache, rebuild, or copy back
#     --force      rebuild even if a local latex/biber process is running
#     PROJECT_DIR  LaTeX project dir  (default: SIP_journal)
#     MAIN_TEX     main .tex within it (default: main.tex)
#
# Exit codes: 0 ok / clean, 2 bad args, 3 live build running, 4 build failed.

set -o pipefail

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }

CHECK_ONLY=0; FORCE=0; POS1=""; POS2=""
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    --force) FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "unknown flag: $arg" >&2; usage >&2; exit 2 ;;
    *) if [ -z "$POS1" ]; then POS1="$arg"; elif [ -z "$POS2" ]; then POS2="$arg"; fi ;;
  esac
done

PROJECT_ARG="${POS1:-SIP_journal}"
MAIN_TEX="${POS2:-main.tex}"
BASE="${MAIN_TEX%.tex}"

PROJECT_DIR="$(cd "$PROJECT_ARG" 2>/dev/null && pwd)"
if [ -z "$PROJECT_DIR" ] || [ ! -f "$PROJECT_DIR/$MAIN_TEX" ]; then
  echo "ERROR: '$PROJECT_ARG/$MAIN_TEX' not found. Run from the repo root or pass PROJECT_DIR." >&2
  exit 2
fi

echo "project : $PROJECT_DIR"
echo "main    : $MAIN_TEX"
echo

# ---- 1. diagnose (in place) -----------------------------------------------
echo "=== diagnosis ==="
python3 - "$PROJECT_DIR" "$BASE" <<'PY'
import os, sys, glob
proj, base = sys.argv[1], sys.argv[2]
p = lambda ext: os.path.join(proj, base + ext)
issues = []
bbl = p(".bbl")
if os.path.isfile(bbl) and os.path.getsize(bbl) == 0:
    issues.append((f"empty {base}.bbl (0 bytes)", "stale biber PAR cache"))
pdf = p(".pdf")
if os.path.isfile(pdf):
    d = open(pdf, "rb").read()
    if d[:5] != b"%PDF-":
        issues.append((f"{base}.pdf has a bad header", "Dropbox sync race (mid-write)"))
    elif b"%%EOF" not in d[-3000:]:
        issues.append((f"{base}.pdf is truncated (no EOF marker)", "Dropbox sync race (mid-write)"))
for ext in (".aux", ".bcf", ".toc", ".out", ".run.xml"):
    q = p(ext)
    if os.path.isfile(q):
        d = open(q, "rb").read()
        if len(d) == 0:
            issues.append((f"{base}{ext} is empty", "Dropbox sync race (mid-write)"))
        elif b"\x00" in d:
            issues.append((f"{base}{ext} contains NUL bytes", "Dropbox sync race (mid-write)"))
for f in glob.glob(os.path.join(proj, "*.synctex(busy)")):
    issues.append((os.path.basename(f) + " present", "build interrupted / being raced by sync"))
for f in sorted(set(glob.glob(os.path.join(proj, "*conflicted copy*"))
                    + glob.glob(os.path.join(proj, "*conflict*")))):
    issues.append((os.path.basename(f), "Dropbox conflict copy — resolve by hand"))
if issues:
    for what, cause in issues:
        print(f"  [!] {what:<44} -> {cause}")
else:
    print("  no corruption signatures right now — this may have been a transient")
    print("  mid-sync state that already settled. Rebuilding still gives a PDF")
    print("  that is guaranteed to match the current source.")
PY
echo

if [ "$CHECK_ONLY" = 1 ]; then
  echo "(--check: diagnosis only; nothing modified)"
  exit 0
fi

# ---- 2. guard: don't race a live local build ------------------------------
if [ "$FORCE" != 1 ] && pgrep -fl 'latexmk|pdflatex|xelatex|lualatex|biber' >/dev/null 2>&1; then
  echo "ERROR: a local latex/biber process is running — refusing to rebuild (would race it)." >&2
  echo "       Wait for it to finish, or pass --force:" >&2
  pgrep -fl 'latexmk|pdflatex|xelatex|lualatex|biber' >&2
  exit 3
fi

# ---- 3. clear the biber PAR cache (fixes the empty-bbl failure mode) -------
PARROOT="${TMPDIR:-/tmp}"
CACHES="$(find "$PARROOT" -maxdepth 1 -type d -name 'par-*' 2>/dev/null)"
if [ -n "$CACHES" ]; then
  echo "clearing biber PAR cache:"; echo "$CACHES" | sed 's/^/  /'
  find "$PARROOT" -maxdepth 1 -type d -name 'par-*' -exec rm -rf {} + 2>/dev/null
else
  echo "biber PAR cache: none to clear"
fi
echo

# ---- 4. rebuild in a scratch dir OUTSIDE Dropbox --------------------------
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/latexfix.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
echo "scratch : $SCRATCH"

# copy the whole project except .git (keeps figures/, .sty, .cls, all \input'd files)
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude='.git' "$PROJECT_DIR"/ "$SCRATCH"/
else
  cp -R "$PROJECT_DIR"/. "$SCRATCH"/ && rm -rf "$SCRATCH/.git"
fi
# remove ROOT-level build artifacts so latexmk builds fresh (figures/*.pdf stay)
( cd "$SCRATCH" && rm -f \
    "$BASE.aux" "$BASE.bbl" "$BASE.bcf" "$BASE.blg" "$BASE.fdb_latexmk" \
    "$BASE.fls" "$BASE.log" "$BASE.out" "$BASE.run.xml" "$BASE.toc" \
    "$BASE.pdf" "$BASE.synctex.gz" "$BASE.synctex(busy)" )

echo "building: latexmk -pdf $MAIN_TEX"
( cd "$SCRATCH" && latexmk -pdf -interaction=nonstopmode "$MAIN_TEX" ) >"$SCRATCH/_build.log" 2>&1
RC=$?
echo "latexmk exit: $RC"

# ---- 5. verify ------------------------------------------------------------
OK=1
NEWPDF="$SCRATCH/$BASE.pdf"
if [ ! -f "$NEWPDF" ]; then
  echo "  FAIL: no $BASE.pdf produced"; OK=0
else
  python3 - "$NEWPDF" <<'PY' || OK=0
import sys
d = open(sys.argv[1], "rb").read()
valid = d[:5] == b"%PDF-" and b"%%EOF" in d[-3000:]
print("  pdf valid:", valid, "| bytes:", len(d))
sys.exit(0 if valid else 1)
PY
fi
count() { n="$(grep -c "$1" "$2" 2>/dev/null)"; echo "${n:-0}"; }
LOG="$SCRATCH/$BASE.log"
echo "  undefined citations: $(count 'Citation.*undefined' "$LOG") | undefined refs: $(count 'LaTeX Warning: Reference.*undefined' "$LOG") | overfull: $(count 'Overfull' "$LOG")"
PAGES="$(pdfinfo "$NEWPDF" 2>/dev/null | awk '/^Pages:/{print $2}')"
[ -n "$PAGES" ] && echo "  pages: $PAGES"

if [ "$RC" != 0 ] || [ "$OK" != 1 ]; then
  echo
  echo "BUILD FAILED. Real errors from the log:"
  grep -nE '^!|Fatal error|Emergency stop|! LaTeX Error' "$LOG" 2>/dev/null | head -15
  echo "Scratch kept for inspection: $SCRATCH"
  trap - EXIT
  exit 4
fi

# ---- 6. copy good artifacts back ------------------------------------------
echo
echo "copying artifacts back to $PROJECT_DIR:"
for ext in .pdf .bbl .aux .bcf .blg .log .out .run.xml .toc .lof .lot; do
  SRC="$SCRATCH/$BASE$ext"
  [ -f "$SRC" ] && cp "$SRC" "$PROJECT_DIR/$BASE$ext" && echo "  $BASE$ext"
done
# drop the stale latexmk db + any in-progress file so the next local build re-inits
rm -f "$PROJECT_DIR/$BASE.fdb_latexmk" "$PROJECT_DIR/$BASE.fls" "$PROJECT_DIR/$BASE.synctex(busy)"

echo
echo "DONE — $PROJECT_DIR/$BASE.pdf is a clean ${PAGES:-?}-page build of the current source."
