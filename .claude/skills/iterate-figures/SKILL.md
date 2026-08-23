---
name: iterate-figures
description: Use when creating or tweaking any generated figure — the SI LM figures, schematic panels, or panels destined for Illustrator — and when committing regenerated figure files.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
---

# Iterate on figures

She converges on figures by looking and reacting, in many small steps. The job is to make each step cheap and visible.

## The loop

1. **Which script?** Every figure comes from `figures/scripts/`, and most families have their own make target:
   - **Results panels + legends** (the Illustrator components, where most iteration happens) ← `figure_paper_panels.py` via `make figures-panels`. Its parts: `_points.py` (the points design), `_agg.py` (pooled model-vs-humans panel), `_data.py` (data prep, reusing `model/cv/model_comparison.py`'s cell specs), `_panels.py` (scaffolding). Study 3's domain-split human panels ← `figure_nonfood_domains.py` via `make figures-nonfood-domains`.
   - **SI figures** ← `figure_si_scenarios.py`, `figure_si_prior_posterior.py`, `figure_si_prereg_predictions.py`, each with a matching `make figures-si-<name>` (`figures-si-scenarios`, `figures-si-prior-posterior`, `figures-si-prereg-predictions`).
   - **SI LM-elicitation figures** (`si_lm_*`) ← `figure_si_lm_validation.py` and `figure_si_consolidated.py`, both via `make figures-lm-si`.
   - **Schematic panels** ← `figure_schematic_plots.py` via `make figures-schematic`.
   Colors/fonts/palettes live **only** in `plot_style.py`; never inline styling. Per-study metadata (given conditions, inferred latents, labels, domain) comes from `study_registry.py`, not a local dict.
2. **Regenerate only the affected script**, filtering warning noise from output. The targets are witness-driven — `make figures-panels` rebuilds only if `figures/panels/results/panel_study1a.pdf` is older than its prereqs, and one witness stands in for every file its script writes. `FIG_SHARED` (`_data.py`, `_panels.py`, `_points.py`, `_agg.py`, `figures/scripts/plot_style.py`, `study_registry.py`, `model/cv/model_comparison.py`) is a prereq of every figure target, so editing any shared module correctly rebuilds them all. If a target is a surprising no-op after an edit, check that the file you edited is actually in the witness's prereq list (`make -pn <target> | grep '^<witness>:'`) before concluding the script is at fault.
3. **Render and show every iteration.** Figures are vector PDFs — convert (`pdftoppm -png -r 100`) or use the `savefig()` PNG preview, and Read/open the image. Never describe a change without showing it.
4. **Ambiguous visual request → one mockup first.** When her ask could mean two different things visually, render one candidate and confirm direction before multi-turn polishing (a misparsed legend request once burned 5 turns and was reverted). For a **brand-new panel composition** (a schematic or summary panel, not a variant of an existing data plot), sketch candidates as HTML/CSS mockups first — she iterates on those fastest ("give me html mockups") — and port the accepted design into matplotlib after.
5. **Prototype variants as scratchpad `mock_*.py` scripts, not as edits to the real script.** A standalone mock that imports `plot_style` and reads the same JSON inputs renders in seconds and costs nothing to throw away; only the accepted version gets folded into the committed script. The schematic-panel sessions went through ~20 rejected variants this way (`mock_utility`, `mock_region`, `mock_joint`, `mock_2d`, …) and `figure_schematic_plots.py` stayed clean.
6. Manuscript-bound figures: `make sync-journal-figures`, then verify the rendered page (build-manuscript skill) — including `grep "Float too large"` in the build log: a figure taller than `\textheight` can't be placed as *any* float, so reduce its `\includegraphics` width. After swapping or restructuring figures, also re-check the in-text figure/panel callouts and captions against the new content — "i changed the main text figures, could you resolve their references in the text" is the standard follow-up.
7. **Quick model-vs-human diagnostics are scratch, not figures.** When she wants "just qualitative stuff" on a model variant mid-development, build the plot from `model/cv/model_comparison.py`'s cell specs, render a PNG into the scratchpad, and never commit it — it is a look, not a paper figure.

## House style (conventions that survived commits)

- Zero values in bar plots get a small visible stub, symmetric around zero — never a literal zero-height bar.
- No white edges around scatter points; small jitter + alpha ≈ 0.8 against overplotting; a sequential palette must not make overlap read as higher value.
- X tick labels horizontal with `\n` line breaks — never 45° rotation, staggering, or abbreviation. On stacked/faceted panels don't repeat the axis *label* on every row — set it on the bottom row only (never drop it entirely).
- Legends below the plot, entries stacked and aligned across legends, preserving the panel's aspect ratio; in-plot legends get a lightly translucent background. Build a below-plot legend as one centered `fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0))` with a thin bottom strip reserved via `tight_layout(rect=[0, gap/fig_h, 1, 1])` so it hugs the plot; `fig.legend` fills cells **column-major**, so to get grouped columns (e.g. actions | effort) concatenate the per-group handle lists — don't interleave.
- Panel letters and panels must align across a multi-panel figure: letters on one line at shared figure coordinates (`plot_style.panel_label_at`, not per-axes offsets), no panel shifted with stray margin ("the a/b/c/d aren't aligned", "panel b is moved to the right ... white space on the left").
- SI figures get a deliberate parity pass against the main-text aesthetic before they count as done: point sizes, font sizes, aspect (SI panels default too small and too horizontal), and legends where the main figures have them ("make the SI plots follow the aesthetics in the main text more").
- Sentence case for all labels; math in `$...$`; label the conditions "Relationship" (not "Intimacy") and "Humans" (not "Human").
- Show the effort split only where effort is manipulated (the `low_risk_share` action) — don't cross all factors mechanically.
- SI figures span **all six studies** per figure (suffix `_all`); never mix food and nonfood within a panel. Bigger fonts, compact dimensions, prefer vertical stacking — but a reduced `\includegraphics` width shrinks fonts, so width-constrained SI figures use the `SI_LARGE_RC` profile (applied via `plt.rc_context`), while full-width or dense figures (e.g. the feature-structure grid) stay on base `si`.
- Illustrator-bound panels: **PDF only** (plus the gitignored PNG preview) — `savefig()` defaults to `formats=("pdf",)`, and a parallel SVG of every panel is repo bloat. Text stays editable in Illustrator through `pdf.fonttype: 42`. Thicker axis/data lines, no titles, minimal chrome, consistent (square-ish) per-panel aspect ratios; panels meant to be stacked by page origin pass `tight=False` so their axes land in the same place on every page.
- Layout choices (faceting, point layers, which panels) are **per-plot decisions — confirm, don't assume**; she has reversed each of these at least once. Encode a new aesthetic as a convention only after it survives to a commit.

## Committing figure files

Regeneration byte-churns PDFs that are visually identical. Before committing: compare each PDF to HEAD with `/CreationDate` and `/ModDate` stripped from both byte strings — exact and faster than rasterizing; identical → `git restore`. Rasterize and eyeball only when the stripped bytes differ and you need to judge whether the change is visual. Then stage an explicit file list. Delete orphaned figure PDFs when a naming scheme changes, and check every `\includegraphics` still resolves.
