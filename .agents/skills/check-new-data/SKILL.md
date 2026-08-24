---
name: check-new-data
description: Use when the user says she added participant data ("i added more data, preprocess again"), when new raw JSON lands in data/<slug>/raw_data/ during live collection, or when she asks how collection is going.
allowed-tools: Bash, Read, Grep, Glob
---

# Process and summarize a new data batch

During live collection she drops new raw JSON in repeatedly (7+ rounds in one collection window) and wants the same check-in each time. Don't re-derive the analysis — run the standard report.

## Procedure

1. Count the batch: `ls data/<slug>/raw_data/*.json | wc -l` (compare to the last known N).
2. Process: `uv run python data_prep/json_to_csv.py <slug>` (equivalently `make data-<slug>`). It applies the per-study exclusion rule (1a lax, all later studies strict) and fails fast on bad raw files — report a failure as-is, don't work around it.
3. Standard report, computed from `main_trials_long.csv` + `exit_survey.csv`:
   - N total / N retained after exclusions, with attention-vs-memory failure tallies and comprehension-attempt distribution.
   - Per-condition belief-update cell means for the study's latents (belief update = posterior − prior rating; effort split shown only within `low_risk_share`).
   - Counterbalancing balance (condition/sequence counts).
   - **Collected N against this study's own preregistered target** (`preregs/<slug>*`), never a sibling's — a launch once reused 3a's N=240 target for 3b and silently collected twice 3b's preregistered 120.
   - The standard human-data plots alongside the numbers ("preprocess and plot" is the default ask): the per-condition belief-update panels via the committed figure machinery where the study has CV-independent panels, else a quick scratch plot.
   - Anything different about this batch vs the running sample (drift in demographics, exclusion rate, or effect direction).
4. On request: the strict-memory robustness variant (retain only `memory_correct_count` ≥ 2 or 3) — she asks for this recurringly; compute it from the CSVs, don't re-implement exclusion logic.
5. If the same inline analysis is being rewritten a third time in a collection window, offer to commit it as a small script in `data_prep/` so subsequent check-ins are one command.

Keep the report compact — a handful of numbers she can scan, not a notebook. Flag, don't interpret away, anything that looks like a data-quality problem.
