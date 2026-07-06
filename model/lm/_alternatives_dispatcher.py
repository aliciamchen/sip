#!/usr/bin/env python3
"""
LM elicitation of counterfactual alternative action sets, one cell at a time.

`elicit_alternatives(client, user_prompt, temperature)` prompts
Llama-3.3-70B-Instruct-Turbo to list the plausible alternative actions the actor
could have taken instead of the observed action (the LM decides set size; no
fixed quota). The caller (`generate_alternatives.py`) builds the per-cell prompt and runs cells
through a thread pool; this module handles one cell's call + parse + dedup.

Requires TOGETHER_API_KEY (in .env) and the `together` package.
"""

import json
import sys
from pathlib import Path

# Shared LM-call infrastructure (JSON helpers, retries) + prompt template.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import (
    MAX_RETRIES,
    MODEL_ID,
    find_json,
    find_json_array,
)
from prompts import ALTERNATIVES_SYSTEM_PROMPT

# NOTE: this call deliberately does NOT pass a strict json_schema
# `response_format`. Grammar-constrained decoding against the alternatives
# schema (which permits an empty array) collapsed to `{"alternatives": []}` on
# hard-to-counterfactual cells — most of the disclosure/privacy nonfood
# scenarios' share actions — even when free generation reliably produces good
# alternatives for the same prompt and seed. We instead let the model generate
# freely and extract the JSON with the tolerant parser below (parse_alternatives
# → find_json / find_json_array), backed by MAX_PARSE_RETRIES. The prompt still
# asks for the exact `[{"action": ...}]` shape, so parseable output is the norm.

MAX_TOKENS = 800
MAX_PARSE_RETRIES = 5

# How many cells to elicit concurrently. One LM call per cell (with up to
# MAX_PARSE_RETRIES parse-retries inside), so this is the per-call concurrency.
MAX_CELL_WORKERS = 8

# How often to flush partial results to disk while the thread pool runs.
CHECKPOINT_EVERY = 16


def parse_alternatives(response_text):
    if response_text is None:
        return None
    # Free generation follows the prompt and emits a bare top-level array
    # [{"action": ...}, ...], which the fallback below reads. We still try a
    # wrapped {"alternatives": [...]} object first for robustness — that was the
    # shape the retired json_schema response_format produced, and older data or
    # a stray wrapped response should still parse.
    arr = None
    js = find_json(response_text)
    if js is not None:
        try:
            obj = json.loads(js)
            if isinstance(obj, dict):
                arr = obj.get("alternatives")
        except (json.JSONDecodeError, ValueError):
            arr = None
    if not isinstance(arr, list):
        js = find_json_array(response_text)
        if js is None:
            return None
        try:
            arr = json.loads(js)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Failed to parse JSON: {e}")
            return None
    if not isinstance(arr, list):
        return None
    out = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if not isinstance(action, str):
            continue
        out.append({"action": action.strip()})
    # An empty array is a VALID "no alternatives" response — return [] with no
    # retries. Only a non-empty array with no parseable item is a parse failure.
    return out if (out or not arr) else None


def _dedup_alternatives(alts):
    seen = set()
    out = []
    for a in alts:
        key = a["action"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out


def elicit_alternatives(client, user_prompt, temperature, seed=None):
    """Elicit alternatives for one (cell, run). Up to MAX_PARSE_RETRIES tries to
    land a parseable response; transient errors inside each call are retried by
    the SDK via ``max_retries=MAX_RETRIES``. Returns the (possibly empty) list
    of alternatives, or None when all parse retries are exhausted (rather than
    raising) so a thread-pool batch can continue. The distinction matters for
    resume: [] is a VALID "no alternatives" elicitation the caller records as
    done, while None means the unit failed and must stay pending.

    For the K-run pipeline the caller passes a nonzero ``temperature`` (so
    independent runs explore genuinely different alternative sets — the
    run-to-run spread is the point) plus a deterministic per-(cell, run)
    ``seed`` for reproducibility; each parse retry offsets the seed by the
    attempt index (masked to Together's non-negative 31-bit seed range) so a
    deterministically unparseable response is not refetched identically. Dedup
    stays WITHIN a run; cross-run repetition is preserved.
    """
    messages = [
        {"role": "system", "content": ALTERNATIVES_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    retrying_client = client.with_options(max_retries=MAX_RETRIES)
    create_kwargs = dict(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
    )
    for attempt in range(MAX_PARSE_RETRIES):
        if seed is not None:
            create_kwargs["seed"] = (seed + attempt) & 0x7FFFFFFF
        try:
            response = retrying_client.chat.completions.create(
                messages=messages, **create_kwargs
            )
            choice = response.choices[0]
            finish = getattr(choice, "finish_reason", None)
            finish = getattr(finish, "value", finish)  # enum in the SDK
            if finish == "length":
                # Truncated at max_tokens — incomplete JSON must not be handed
                # to the parser as if it were a complete response.
                print(
                    f"  Attempt {attempt + 1}: response truncated at max_tokens "
                    "(finish_reason=length); retrying",
                    flush=True,
                )
                continue
            parsed = parse_alternatives(choice.message.content)
            if parsed is not None:
                return _dedup_alternatives(parsed)
        except Exception as e:
            print(f"  Attempt {attempt + 1} error: {e}", flush=True)
    print(
        "  All parse retries exhausted; leaving this (cell, run) unit pending "
        "for a future invocation.",
        flush=True,
    )
    return None
