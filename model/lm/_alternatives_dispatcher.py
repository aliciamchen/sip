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
    alternatives_array_schema,
    find_json_array,
)
from prompts import ALTERNATIVES_SYSTEM_PROMPT

# Built once and reused — schema construction is pure.
_ALTERNATIVES_RESPONSE_FORMAT = alternatives_array_schema()

TEMPERATURE = 1.0
MAX_TOKENS = 800
MAX_PARSE_RETRIES = 5

# How many cells to elicit concurrently. One LM call per cell (with up to
# MAX_PARSE_RETRIES parse-retries inside), so this is the per-call concurrency.
MAX_CELL_WORKERS = 8

# How often to flush partial results to disk while the thread pool runs.
CHECKPOINT_EVERY = 16


def parse_alternatives(response_text):
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
    return out if out else None


def _dedup_alternatives(alts):
    seen = set()
    out = []
    for a in alts:
        key = a["action"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out


def elicit_alternatives(client, user_prompt, temperature=TEMPERATURE, seed=None):
    """Elicit alternatives for one (cell, run). Up to MAX_PARSE_RETRIES tries to
    land a parseable response; transient errors inside each call are retried by
    the SDK via ``max_retries=MAX_RETRIES``. Returns [] when all parse retries are
    exhausted (rather than raising) so a thread-pool batch can continue.

    ``temperature`` defaults to the module-level TEMPERATURE (1.0). For the K-run
    pipeline the caller passes a higher T (so independent runs explore genuinely
    different alternative sets — the run-to-run spread is the point) plus a
    deterministic per-(cell, run) ``seed`` for reproducibility. Dedup stays WITHIN
    a run; cross-run repetition is preserved.
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
        response_format=_ALTERNATIVES_RESPONSE_FORMAT,
    )
    if seed is not None:
        create_kwargs["seed"] = seed
    for attempt in range(MAX_PARSE_RETRIES):
        try:
            response = retrying_client.chat.completions.create(
                messages=messages, **create_kwargs
            )
            parsed = parse_alternatives(response.choices[0].message.content)
            if parsed:
                return _dedup_alternatives(parsed)
        except Exception as e:
            print(f"  Attempt {attempt + 1} error: {e}", flush=True)
    print(
        "  All parse retries exhausted; returning empty alternative set for this cell.",
        flush=True,
    )
    return []
