#!/usr/bin/env python3
"""Export the project's LM prompts to LaTeX for the paper's Supplementary Material.

This reads the production prompt templates straight from ``prompts.py`` and
renders each one inside a titled, monospaced box, so the prompts reproduced in
the paper cannot drift from the prompts the pipeline actually sends. Re-run it
whenever ``prompts.py`` changes.

The prompts come in four groups, matching the elicitation stages in the
manuscript:

  1. **Counterfactual action generation** (the generator ``G_LM``): one system
     prompt and one worked example of the user prompt. Which condition
     paragraphs each experiment reveals is the experiment design, described in
     the main text, so it is not restated here.
  2. **Utility-feature scoring** (the feature map ``phi_tau``): a system prompt
     each for goal-satisfaction ``g``, effort, and interpersonal risk,
     plus the shared user-prompt template and the one rating-instruction line
     that varies across the three features.
  3. **Given-magnitude ratings**: the scalar desire and relationship-intimacy
     ratings used in the studies where those variables are given rather than
     inferred.

Each prompt is rendered verbatim (no paraphrase). Variable content that the
pipeline fills in per trial -- the scenario vignette, the motivational-state
and effort paragraphs, the action texts -- is shown as ``<angle-bracket>``
placeholders; the actual text of those paragraphs lives in the scenario tables.

Output is a single ``si_prompts.tex`` meant to be ``\\input`` from the
manuscript. Pass ``--standalone`` to instead wrap it in a minimal document you
can compile on its own to preview the boxes.

Usage:
    uv run python model/lm/export_prompts_latex.py            # write si_prompts.tex
    uv run python model/lm/export_prompts_latex.py --out path.tex
    uv run python model/lm/export_prompts_latex.py --standalone --out preview.tex
"""

import argparse
import os
import sys
from pathlib import Path

# prompts.py lives next to this file; import it the same way the LM scripts do.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- placeholders for per-trial variable content -----------------------------
VIGNETTE = "<scenario vignette>"
STATE = "<motivational-state paragraph>"
EFFORT_PARA = "<physical-effort paragraph>"
OBSERVED = "<observed action>"
DESIRE_OBJECT = "<the resource at stake>"
RELATIONSHIP = "<relationship descriptor>"
FEATURE_INSTR = "<feature-specific rating instruction (see below)>"
ACTIONS = ["<observed action>", "<alternative action 1>", "<... alternative action k>"]

# --- LaTeX preamble the generated file depends on -----------------------------
# Kept in one place so both the header comment (non-standalone) and the
# standalone wrapper stay in sync.
PREAMBLE = r"""\usepackage{fvextra}  % extends fancyvrb; provides Verbatim line-breaking (breaklines)
\usepackage[skins,breakable]{tcolorbox}
% promptbox: a simple titled, monospaced box for reproducing an LM prompt.
\newtcolorbox{promptbox}[1]{%
  breakable, enhanced,
  colback=gray!3, colframe=gray!50, boxrule=0.5pt, arc=1.5pt,
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  fonttitle=\bfseries\footnotesize, coltitle=black, colbacktitle=gray!15,
  title={#1},
}"""


def vsub(text):
    """Replace the handful of non-ASCII characters in the prompts with
    pdfLaTeX-safe ASCII so the boxes compile under any engine/font."""
    return text.replace("—", "--").replace("∈", "in")


def box(title, body):
    """A titled promptbox wrapping `body` in a line-wrapping verbatim block."""
    body = vsub(body).rstrip("\n")
    return (
        f"\\begin{{promptbox}}{{{title}}}\n"
        "\\begin{Verbatim}[breaklines=true,breakanywhere=false,"
        "breaksymbolleft={},breaksymbolright={},fontsize=\\footnotesize]\n"
        f"{body}\n"
        "\\end{Verbatim}\n"
        "\\end{promptbox}\n\n"
    )


def subsection(title):
    """A numbered subsection (S-prefixed in the manuscript appendix)."""
    return f"\\subsection{{{title}}}\n\n"


def build_content():
    """Render every prompt, grouped, returning the body LaTeX (no preamble)."""
    out = []

    # ------------------------------------------------------------------ group 1
    out.append(subsection("Counterfactual action generation"))
    out.append(
        box(
            "System prompt --- alternative action generation ($G_{\\mathrm{LM}}$)",
            prompts.ALTERNATIVES_SYSTEM_PROMPT,
        )
    )
    # ONE worked example rather than one box per experiment. The six live
    # branches differ only in which given-condition paragraphs are revealed and
    # which latents are flagged unknown -- which is the experiment design, stated
    # in the main text, so restating it here was pure repetition. The example is
    # the 1a branch (inferring desire): a revealed effort paragraph and
    # relationship sentence, with the desire object flagged unknown-magnitude.
    prompts.RELATIONSHIP_DESCRIPTORS["__tmpl__"] = RELATIONSHIP
    try:
        alt_user_example = prompts.alternatives_user_prompt(
            VIGNETTE,
            OBSERVED,
            effort_text=EFFORT_PARA,
            intimacy_level="__tmpl__",
            unknown_desire_object=DESIRE_OBJECT,
        )
    finally:
        del prompts.RELATIONSHIP_DESCRIPTORS["__tmpl__"]
    out.append(
        box(
            "User prompt (Example) --- alternative action generation",
            alt_user_example,
        )
    )

    # ------------------------------------------------------------------ group 2
    out.append(subsection("Utility-feature scoring"))
    out.append(
        box(
            "System prompt --- goal-satisfaction $g_{\\tau}(a)$",
            prompts.system_prompt("g"),
        )
    )
    out.append(
        box(
            "System prompt --- effort $\\mathrm{effort}_{\\tau}(a)$",
            prompts.system_prompt("effort"),
        )
    )
    out.append(
        box(
            "System prompt --- interpersonal risk $\\mathrm{risk}_{\\tau}(a)$",
            prompts.system_prompt("risk"),
        )
    )
    # The three feature user prompts are identical except for one instruction
    # line; show the shared template with that line as a placeholder.
    shared_user = prompts.user_prompt("risk", VIGNETTE, ACTIONS)
    shared_user = shared_user.replace(prompts._USER_INSTRUCTIONS["risk"], FEATURE_INSTR)
    out.append(
        box(
            "User prompt (template) --- feature scoring (shared across $g$, effort, risk)",
            shared_user,
        )
    )

    # The actual per-feature instruction lines. The g instruction is rendered
    # through the real formatter (prompts.user_prompt) so the SI cannot drift
    # from the code; its second paragraph is the instruction line. Desire
    # objects phrased as an infinitive outcome (some non-food scenarios, e.g.
    # "to try the harmonica") drop "or consuming", so both renderings are shown.
    def g_instr_for(obj):
        return prompts.user_prompt("g", VIGNETTE, ACTIONS, desire_object=obj).split(
            "\n\n"
        )[1]

    instr_body = (
        f"goal-satisfaction g:  {g_instr_for(DESIRE_OBJECT)}\n"
        "  (for desire objects phrased as an infinitive outcome in some\n"
        '   non-food scenarios, "getting or consuming" becomes "getting",\n'
        f'   e.g.: "{g_instr_for("to try the harmonica")}")\n\n'
        f"effort:               {prompts._USER_INSTRUCTIONS['effort']}\n\n"
        f"interpersonal risk:   {prompts._USER_INSTRUCTIONS['risk']}"
    )
    out.append(
        box(
            "Per-feature rating instruction (the one line that varies)",
            instr_body,
        )
    )

    # ------------------------------------------------------------------ group 3
    out.append(subsection("Given-magnitude ratings"))
    out.append(
        box(
            "System prompt --- desire $d$",
            prompts.DESIRE_SYSTEM_PROMPT,
        )
    )
    out.append(
        box(
            "User prompt (template) --- desire $d$",
            prompts.desire_user_prompt(VIGNETTE, STATE, DESIRE_OBJECT),
        )
    )
    out.append(
        box(
            "System prompt --- relationship intimacy $I$",
            prompts.INTIMACY_SYSTEM_PROMPT,
        )
    )
    out.append(
        box(
            "User prompt (template) --- relationship intimacy $I$",
            prompts.relationship_user_prompt(RELATIONSHIP),
        )
    )

    return "".join(out).rstrip("\n") + "\n"


def header_comment():
    pre = "\n".join("% " + line for line in PREAMBLE.splitlines())
    return (
        "% !!! AUTO-GENERATED by model/lm/export_prompts_latex.py -- do not edit by hand.\n"
        "% Regenerate with:  uv run python model/lm/export_prompts_latex.py\n"
        "% Reproduces, verbatim, the LM prompts in model/lm/prompts.py.\n"
        "%\n"
        "% This file is meant to be \\input from the manuscript. It requires the\n"
        "% following in your preamble (the promptbox environment + its packages):\n"
        "%\n"
        f"{pre}\n"
        "%\n"
    )


def standalone(content):
    return (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{xcolor}\n"
        "\\usepackage{amsmath}\n"
        f"{PREAMBLE}\n"
        "\\begin{document}\n"
        "\\setcounter{tocdepth}{2}\n"
        "\\tableofcontents\n"
        "\\newpage\n"
        "\\section*{LM prompts}\n\n"
        f"{content}"
        "\\end{document}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = PROJECT_ROOT / "SIP_journal" / "si_prompts.tex"
    if not default_out.parent.exists():
        default_out = PROJECT_ROOT / "si_prompts.tex"
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"output .tex path (default: {default_out})",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="wrap the boxes in a minimal compilable document for preview",
    )
    args = parser.parse_args()

    content = build_content()
    if args.standalone:
        text = standalone(content)
    else:
        text = header_comment() + "\n" + content

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
