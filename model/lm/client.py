#!/usr/bin/env python3
"""Shared LM-call infrastructure for the Together AI rating + alternative
elicitation pipelines.

Centralizes what previously lived in three near-duplicate copies across
``lm/score_canonical_features.py``, ``lm/score_effort_features.py``, and
``lm/generate_alternatives_desire.py``:

- ``load_api_key`` — resolve ``TOGETHER_API_KEY`` from env or ``.env``.
- ``find_json`` / ``find_json_array`` — best-effort JSON extraction.
- ``strip_leading_plus`` — pre-process for the signed V scale, where Llama
  occasionally emits ``"action_0": +3`` (invalid JSON).
- ``get_ratings_concurrent`` — fan out N rating calls across a thread pool,
  letting the Together SDK retry transient errors via ``max_retries``.
  Returns ``(successes, n_failures)`` so callers can record both columns.
- ``aggregate_action_ratings`` — generic ``action_0..action_{n-1}`` aggregator.

A thread pool with the synchronous ``Together`` client is used in preference to
``AsyncTogether`` because the workload is modest (≤10 concurrent calls per
scenario) and a thread pool keeps callers as plain ``def`` rather than forcing
``async def`` propagation up every call stack.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from utils import get_project_root


# Together AI configuration shared across all rating call sites.
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
# Default repeat count for `get_ratings_concurrent`. The active K-run pipeline
# passes num_runs=1 (each elicitation run is scored once; the K runs are the
# variation axis), so this default is no longer exercised by score_merged.
NUM_RUNS = 10
TEMPERATURE = 0.2

# Together SDK default is 2; the workload is non-interactive so a slightly
# larger budget is cheaper than re-running an entire scenario.
MAX_RETRIES = 5

# Default thread-pool size for fanning out the NUM_RUNS-per-scenario calls.
MAX_WORKERS = 10


# ==============================================================================
# API key loading
# ==============================================================================


def load_api_key():
    """Resolve TOGETHER_API_KEY from the environment, falling back to .env."""
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        env_path = get_project_root() / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("TOGETHER_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip('"').strip("'")
                        break
    if not api_key:
        print("Error: set TOGETHER_API_KEY in env or .env")
        sys.exit(1)
    return api_key


# ==============================================================================
# JSON extraction helpers
# ==============================================================================


_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_LEADING_PLUS_RE = re.compile(r":\s*\+(\d)")


def find_json(text):
    """Return the substring from the first ``{`` to the last ``}``, or None."""
    if text is None:
        return None
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    return text[start:end] if start != -1 and end > start else None


def find_json_array(text):
    """Return the first ``[...]`` substring, or None."""
    if text is None:
        return None
    match = _JSON_ARRAY_RE.search(text)
    return match.group(0) if match else None


def strip_leading_plus(text):
    """Drop leading ``+`` from numeric JSON values (V's signed -3..+3 scale)."""
    return _LEADING_PLUS_RE.sub(r": \1", text)


# ==============================================================================
# Concurrent ratings runner
# ==============================================================================


def _one_call(
    client,
    system_prompt,
    user_prompt,
    model_id,
    max_tokens,
    temperature,
    max_retries,
    response_format=None,
    seed=None,
):
    """Issue one chat-completion call. Returns response text or None on failure.

    Transient errors (network, 429, 5xx) are retried by the SDK via
    ``max_retries``. Anything still failing after that is caught and translated
    to ``None`` so the surrounding batch can continue.

    ``response_format`` is forwarded to Together's structured-output API. Pass
    e.g. ``{"type": "json_schema", "json_schema": {"name": "ratings", "schema": ...}}``
    to constrain the output to a JSON schema.

    ``seed`` is forwarded for best-effort reproducibility (same seed + prompt +
    model + params → same output, per the Together docs)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = dict(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format
    if seed is not None:
        kwargs["seed"] = seed
    try:
        resp = client.with_options(max_retries=max_retries).chat.completions.create(
            **kwargs
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"  call error: {e}", flush=True)
        return None


def get_ratings_concurrent(
    client,
    system_prompt,
    user_prompt,
    parse_fn,
    num_runs=NUM_RUNS,
    *,
    model_id=MODEL_ID,
    max_tokens=200,
    temperature=TEMPERATURE,
    max_workers=MAX_WORKERS,
    max_retries=MAX_RETRIES,
    min_success_ratio=0.7,
    label=None,
    response_format=None,
    seed=None,
):
    """Fan out ``num_runs`` calls across a thread pool. Returns
    ``(successful_parses, n_failures)``.

    ``parse_fn`` takes the raw response text and returns the parsed dict (or
    ``None`` to mark the run as a failure). Failed runs do not appear in the
    success list but are counted; a warning prints when success rate drops
    below ``min_success_ratio``. ``label`` is shown in the warning so callers
    can identify which (scenario, rating-type) the warning belongs to.

    ``seed``, when given, is offset by the call index (``seed + i``) so a
    ``num_runs > 1`` fan-out stays varied while remaining reproducible; with the
    active ``num_runs=1`` pipeline this just pins the single call to ``seed``."""
    workers = min(max_workers, num_runs)
    successes = []
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _one_call,
                client,
                system_prompt,
                user_prompt,
                model_id,
                max_tokens,
                temperature,
                max_retries,
                response_format,
                (seed + i) if seed is not None else None,
            )
            for i in range(num_runs)
        ]
        for fut in as_completed(futures):
            text = fut.result()
            parsed = parse_fn(text) if text is not None else None
            if parsed is not None:
                successes.append(parsed)
            else:
                failures += 1
    if failures > 0 and len(successes) < num_runs * min_success_ratio:
        tag = f" [{label}]" if label else ""
        print(
            f"  WARNING{tag}: only {len(successes)}/{num_runs} runs returned "
            f"parseable output ({failures} failures)",
            flush=True,
        )
    return successes, failures


# ==============================================================================
# Schema builders for Together's structured-output mode
# ==============================================================================


def numeric_action_schema(n_actions, name="ratings"):
    """Build a response_format object that constrains the LM to emit
    ``{"action_0": <number>, ..., "action_{n-1}": <number>}``.

    Used by risk/effort/V rating calls in both the canonical 4-action and
    variable-length paths. ``strict: true`` makes the server enforce exact schema
    adherence (Together accepts it for Llama-3.3-70B-Turbo); the schema is
    strict-compatible — every property required, ``additionalProperties`` closed."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    f"action_{i}": {"type": "number"} for i in range(n_actions)
                },
                "required": [f"action_{i}" for i in range(n_actions)],
                "additionalProperties": False,
            },
        },
    }


def alternatives_array_schema(name="alternatives"):
    """response_format for the alternative-generation calls.

    Constrains the LM to emit ``{"alternatives": [{"action": <string>}, ...]}``.
    The array is wrapped in an object (rather than a bare top-level array) because
    ``strict: true`` requires an object root; the wrapper also lets the parser read
    a single object instead of regex-extracting an array."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "alternatives": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                            },
                            "required": ["action"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["alternatives"],
                "additionalProperties": False,
            },
        },
    }


# ==============================================================================
# Aggregator
# ==============================================================================


def aggregate_action_ratings(ratings_list, n_actions):
    """Aggregate ``action_0..action_{n_actions-1}`` ratings to (mean, std)
    tuples per key. Empty input or all-missing keys yield (NaN, NaN)."""
    if not ratings_list:
        return {f"action_{i}": (np.nan, np.nan) for i in range(n_actions)}
    result = {}
    for i in range(n_actions):
        key = f"action_{i}"
        values = [r[key] for r in ratings_list if key in r]
        result[key] = (np.mean(values), np.std(values)) if values else (np.nan, np.nan)
    return result


# ==============================================================================
# Run-provenance manifest
# ==============================================================================


def _git_sha():
    """Repo HEAD short SHA, or None outside a git checkout / on any error."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(get_project_root()),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _prompts_sha():
    """Short SHA-256 of prompts.py, so a tweaked prompt yields a different
    manifest even when the model and config are unchanged."""
    try:
        prompts_path = Path(__file__).resolve().parent / "prompts.py"
        return hashlib.sha256(prompts_path.read_bytes()).hexdigest()[:12]
    except Exception:
        return None


def write_run_manifest(output_path, stage, study, extra=None):
    """Write a small provenance sidecar next to a stage's JSONL output.

    The whole point of the elicitation artifacts is that the values are
    LM-generated, so two regenerations (a different model, a tweaked prompt, more
    runs) must be distinguishable. This records how the file was produced — model,
    prompt + code version, timestamp — next to the data file it describes
    (``lm_runs.jsonl`` → ``lm_runs.manifest.json``). ``extra`` carries
    stage-specific config (K runs, temperature, record counts). Mirrors the
    plain-JSON style of embed_alternatives.py's lm_clusters.json."""
    manifest = {
        "stage": stage,
        "study": study,
        "model": MODEL_ID,
        "prompts_sha256": _prompts_sha(),
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        manifest.update(extra)
    output_path = Path(output_path)
    manifest_path = output_path.with_name(output_path.stem + ".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
