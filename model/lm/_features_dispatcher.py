#!/usr/bin/env python3
"""
Shared LM-scoring helpers for the merged feature / goal-satisfaction scorer
(`score_merged.py`).

Pure functions only: prompt formatters for the variable-length (LM-alternatives)
scoring calls, 0-6 → model-scale normalizers (risk, effort, g all → [0, 1]),
response parsers, and small schema/token helpers. The scripts that
import these own the actual LM calls and CSV writing.
"""

import json
import sys
from pathlib import Path

# Shared LM-call infrastructure + prompt templates (sibling modules).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import find_json, strip_leading_plus
from prompts import user_prompt as build_user_prompt


# ==============================================================================
# Prompt formatters (variable-length / LM-alternatives calls)
# ==============================================================================


def format_risk_prompt_variable(vignette, action_texts):
    return build_user_prompt("risk", vignette, action_texts)


def format_effort_prompt_variable(vignette, action_texts):
    return build_user_prompt("effort", vignette, action_texts)


def format_g_prompt_variable(vignette, action_texts, desire_object=None):
    """Goal-satisfaction g prompt: how fully each action delivers the outcome.
    Desire-free (no state paragraph) — desire enters the utility separately as
    the multiplier in w_v · desire · g. `desire_object` names the resource at
    stake so the instruction reads "getting or consuming the hot dog" rather
    than the generic "the thing at stake"."""
    return build_user_prompt("g", vignette, action_texts, desire_object=desire_object)


# ==============================================================================
# Normalization (0-6 LLM scale -> model-native scales)
# ==============================================================================


def normalize_risk(value, target_max=1.0):
    """0-6 -> [0, 1]. On the same scale as effort and g. The absolute scale is
    non-identifiable in the fit (the free, unregularized weight w_d absorbs any
    constant factor), so all three features share [0, 1] for comparability rather
    than carrying feature-specific ranges."""
    return value * (target_max / 6.0)


def normalize_effort(value, target_max=1.0):
    return value * (target_max / 6.0)


def normalize_g(value, target_max=1.0):
    """0-6 -> [0, 1]. Goal-satisfaction rating (how fully the action delivers
    the outcome)."""
    return value * (target_max / 6.0)


# ==============================================================================
# Schemas + response parsers
# ==============================================================================


def numeric_desire_schema(name="desire"):
    """response_format constraining the LM to emit ``{"desire": <number>}`` for
    the scenario-level desire rating in the given-desire studies."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": {
                "type": "object",
                "properties": {"desire": {"type": "number"}},
                "required": ["desire"],
                "additionalProperties": False,
            },
        },
    }


def parse_desire_response(response_text):
    """Parse a ``{"desire": <number>}`` scalar response (0-100)."""
    if response_text is None:
        return None
    js = find_json(response_text)
    if js is None:
        return None
    js = strip_leading_plus(js)
    try:
        d = json.loads(js)
        if "desire" in d:
            return float(d["desire"])
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"  Failed to parse desire JSON: {e}")
    return None


def numeric_intimacy_schema(name="intimacy"):
    """response_format constraining the LM to emit ``{"intimacy": <number>}`` for
    the per-level relationship-intimacy rating in the given-relationship studies."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": {
                "type": "object",
                "properties": {"intimacy": {"type": "number"}},
                "required": ["intimacy"],
                "additionalProperties": False,
            },
        },
    }


def parse_intimacy_response(response_text):
    """Parse a ``{"intimacy": <number>}`` scalar response (0-100)."""
    if response_text is None:
        return None
    js = find_json(response_text)
    if js is None:
        return None
    js = strip_leading_plus(js)
    try:
        d = json.loads(js)
        if "intimacy" in d:
            return float(d["intimacy"])
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"  Failed to parse intimacy JSON: {e}")
    return None


def parse_action_response_variable(response_text, n_actions):
    """Parse JSON with action_0..action_{n-1} keys."""
    if response_text is None:
        return None
    js = find_json(response_text)
    if js is None:
        return None
    js = strip_leading_plus(js)
    try:
        ratings = json.loads(js)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Failed to parse JSON: {e}")
        return None
    out = {}
    for i in range(n_actions):
        key = f"action_{i}"
        if key not in ratings:
            return None
        try:
            out[key] = float(ratings[key])
        except (TypeError, ValueError):
            return None
    return out


def _max_tokens_for(n_actions):
    """Token budget that scales with the number of actions in a variable-length call."""
    return max(200, 40 * n_actions)
