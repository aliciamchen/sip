#!/usr/bin/env python3
"""Shared LM-call infrastructure for the Together AI rating + alternative
elicitation pipelines.

Centralizes what previously lived in near-duplicate copies across the earlier
per-feature scoring and alternative-generation scripts:

- ``load_api_key`` — resolve ``TOGETHER_API_KEY`` from env or ``.env``.
- ``find_json`` / ``find_json_array`` — best-effort JSON extraction.
- ``strip_leading_plus`` — drop the leading ``+`` Llama occasionally emits
  before numeric JSON values (invalid JSON).
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
TEMPERATURE = 0.2

# `get_ratings_concurrent` deliberately has NO default for `num_runs`. It used to
# default to 10, from the pre-K-run design where a rating was the mean of ten
# calls. Every live caller now passes num_runs=1 — each elicitation run is scored
# once, and the K runs are the variation axis — so the default was dead, but it
# was dead in the expensive direction: a new call site that forgot the argument
# would silently make 10x the API calls. Requiring it turns that into a
# TypeError at the call site.
#
# Together SDK default is 2; the workload is non-interactive so a slightly
# larger budget is cheaper than re-running an entire scenario.
MAX_RETRIES = 5

# Thread-pool size when a caller does fan out more than one run. With num_runs=1
# throughout the active pipeline the pool holds a single task, so this bounds
# nothing in practice; it stays because the fan-out mechanism is still correct
# and cheap, and a diagnostic may legitimately want repeats.
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
    """Drop the leading ``+`` Llama occasionally emits before numeric JSON
    values (e.g. ``"action_0": +3``), which is invalid JSON."""
    return _LEADING_PLUS_RE.sub(r": \1", text)


# ==============================================================================
# JSONL checkpoint I/O
# ==============================================================================


def write_jsonl_atomic(path, rows):
    """Rewrite a JSONL checkpoint via a same-directory temp file + os.replace,
    so a kill mid-write can't destroy already-paid-for records. (The main repo
    lives in Dropbox, which makes in-place truncate-and-rewrite extra risky.)"""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_jsonl_checked(path):
    """Read a JSONL file into a list of records, failing with a recovery
    message (file, line, byte offset) on an unparseable line instead of a bare
    JSONDecodeError. A truncated final line means an interrupted write; resume
    must surface it, not silently skip or crash on it."""
    path = Path(path)
    records = []
    offset = 0
    raw_lines = path.read_bytes().split(b"\n")
    for lineno, raw in enumerate(raw_lines, start=1):
        stripped = raw.strip()
        if stripped:
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                if all(not rest.strip() for rest in raw_lines[lineno:]):
                    raise SystemExit(
                        f"{path} ends with a truncated record at line {lineno} "
                        f"(byte offset {offset}) — likely an interrupted "
                        f"write. Recover by truncating the file to that offset "
                        f"(`truncate -s {offset} '{path}'`) to drop the "
                        "partial record, then re-run to resume."
                    )
                raise SystemExit(
                    f"{path} has an unparseable record at line {lineno} (byte "
                    f"offset {offset}), before the end of the file — the "
                    "corruption is not just a truncated tail. Restore the "
                    "file from a backup or delete it to re-elicit."
                )
        offset += len(raw) + 1
    return records


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
        choice = resp.choices[0]
        finish = getattr(choice, "finish_reason", None)
        finish = getattr(finish, "value", finish)  # enum in the SDK, str in raw
        if finish == "length":
            # Truncated at max_tokens: the JSON is incomplete, so don't hand
            # it to the parser as if it were a complete response.
            print(
                "  call truncated at max_tokens (finish_reason=length); "
                "treating as failed",
                flush=True,
            )
            return None
        return choice.message.content
    except Exception as e:
        print(f"  call error: {e}", flush=True)
        return None


def get_ratings_concurrent(
    client,
    system_prompt,
    user_prompt,
    parse_fn,
    num_runs,
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

    ``seed``, when given, is offset by the call index (``seed + i``, masked to
    Together's non-negative 31-bit seed range) so a ``num_runs > 1`` fan-out
    stays varied while remaining reproducible; with the active ``num_runs=1``
    pipeline this just pins the single call to ``seed``."""
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
                ((seed + i) & 0x7FFFFFFF) if seed is not None else None,
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

    Used by the per-action risk/effort/g rating calls (``n_actions`` varies per
    scenario). ``strict: true`` makes the server enforce exact schema
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
    """Legacy short SHA-256 of the complete prompts.py source.

    Old manifests use this as their only prompt fingerprint. New manifests
    retain it for source-level traceability but use ``_prompt_sha(stage)`` for
    resume safety, so comments and unrelated stages no longer cause false
    mismatches.
    """
    try:
        prompts_path = Path(__file__).resolve().parent / "prompts.py"
        return hashlib.sha256(prompts_path.read_bytes()).hexdigest()[:12]
    except Exception:
        return None


def _prompt_sha(stage):
    """Short SHA-256 of prompt surfaces that determine ``stage``'s output."""
    try:
        try:
            from .prompts import prompt_fingerprint_payload
        except ImportError:
            from prompts import prompt_fingerprint_payload

        payload = prompt_fingerprint_payload(stage)
        return fingerprint_payload(payload)
    except (ImportError, OSError):
        return None


def fingerprint_payload(payload):
    """Short SHA-256 of a JSON-serializable, canonically encoded payload."""
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def read_run_manifest(output_path):
    """The provenance manifest previously written next to ``output_path`` by
    ``write_run_manifest`` (``lm_runs.jsonl`` → ``lm_runs.manifest.json``), or
    None if there isn't one."""
    output_path = Path(output_path)
    manifest_path = output_path.with_name(output_path.stem + ".manifest.json")
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


RESUME_PROMPT_MISMATCH_ENV = "LM_RESUME_PROMPT_MISMATCH"


def _manifest_prompt_hashes(manifest):
    """Return ``(recorded, current, field)`` for a manifest's prompt format."""
    if manifest.get("prompt_sha256") is not None:
        return (
            manifest["prompt_sha256"],
            _prompt_sha(manifest.get("stage")),
            "prompt_sha256",
        )
    return manifest.get("prompts_sha256"), _prompts_sha(), "prompts_sha256"


def manifest_prompt_matches(manifest):
    """Whether a manifest's recorded prompt fingerprint matches current code."""
    old, cur, _ = _manifest_prompt_hashes(manifest)
    return old is None or cur is None or old == cur


def guard_resume_fingerprint_mismatch(
    output_path, field, current, description, *, allow_legacy_missing=True
):
    """Refuse a resume when a recorded dynamic-input fingerprint changed.

    Stage hashes cover centralized prompt templates. Callers use this companion
    guard for exact rendered messages or artifact inputs assembled outside
    prompts.py. A missing field is allowed only for legacy manifests whose
    stage/whole-source prompt guard has already passed.
    """
    manifest = read_run_manifest(output_path)
    if manifest is None:
        return
    recorded = manifest.get(field)
    if recorded is None:
        if allow_legacy_missing:
            return
        raise SystemExit(
            f"The manifest for {Path(output_path).name} predates {description} "
            "fingerprinting, so a safe resume is impossible. Delete the output "
            "and re-run the stage from scratch."
        )
    if current is None or recorded == current:
        return
    raise SystemExit(
        f"The {description} changed after {Path(output_path).name} began "
        f"(manifest {field}={recorded}, current={current}). Resuming would "
        "silently mix incompatible records. Delete the output and re-run the "
        "stage from scratch."
    )


def guard_resume_prompt_mismatch(output_path):
    """Refuse to resume onto data elicited under a different stage prompt.

    Resume skips already-done units, so silently resuming after a prompt edit
    would mix records from two prompt versions in one output file. New
    manifests compare a stage-specific ``prompt_sha256`` built from rendered
    prompt surfaces; legacy manifests fall back to their whole-file
    ``prompts_sha256``. A mismatch hard-errors unless
    ``LM_RESUME_PROMPT_MISMATCH=allow`` is set, in which case the superseded
    hash is retained in the next manifest.
    """
    manifest = read_run_manifest(output_path)
    if manifest is None:
        return
    old, cur, field = _manifest_prompt_hashes(manifest)
    if old is None or cur is None or old == cur:
        return
    if os.environ.get(RESUME_PROMPT_MISMATCH_ENV, "").lower() == "allow":
        print(
            f"WARNING: resuming {Path(output_path).name} elicited under "
            f"{field}={old} with current stage prompt ({cur}); the mixed "
            f"provenance will be recorded in the manifest "
            f"({RESUME_PROMPT_MISMATCH_ENV}=allow).",
            flush=True,
        )
        return
    raise SystemExit(
        f"The prompt for stage {manifest.get('stage')!r} has changed since "
        f"{Path(output_path).name} was elicited (manifest {field}={old}, "
        f"current={cur}). Resuming would "
        "silently mix data from two prompt versions. Either delete the output "
        "file (and its manifest) to re-elicit from scratch, or set "
        f"{RESUME_PROMPT_MISMATCH_ENV}=allow to resume anyway (the mismatch "
        "is then recorded in the manifest)."
    )


def write_run_manifest(output_path, stage, study, extra=None):
    """Write a small provenance sidecar next to a stage's JSONL output.

    The whole point of the elicitation artifacts is that the values are
    LM-generated, so two regenerations (a different model, a tweaked prompt, more
    runs) must be distinguishable. This records how the file was produced — model,
    prompt + code version, timestamp — next to the data file it describes
    (``lm_runs.jsonl`` → ``lm_runs.manifest.json``). ``extra`` carries
    stage-specific config (K runs, temperature, record counts).

    ``prompt_sha256`` identifies only the rendered prompts used by this stage.
    The legacy whole-source ``prompts_sha256`` remains for traceability. If an
    existing manifest is replaced after an explicitly allowed prompt mismatch,
    superseded hashes are preserved in history lists."""
    manifest = {
        "stage": stage,
        "study": study,
        "model": MODEL_ID,
        "prompt_sha256": _prompt_sha(stage),
        "prompts_sha256": _prompts_sha(),
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    prior = read_run_manifest(output_path)
    if prior is not None:
        prior_sha = prior.get("prompt_sha256")
        # Before stage-specific hashes existed, ``prompt_sha_history`` held
        # whole-source prompts.py hashes. Only carry that field forward as
        # stage history when the prior manifest actually has a stage hash.
        history = (
            list(prior.get("prompt_sha_history", [])) if prior_sha is not None else []
        )
        if (
            prior_sha
            and prior_sha != manifest["prompt_sha256"]
            and prior_sha not in history
        ):
            history.append(prior_sha)
        if history:
            manifest["prompt_sha_history"] = history
        source_history = list(prior.get("prompts_sha_history", []))
        if prior_sha is None:
            for legacy_sha in prior.get("prompt_sha_history", []):
                if legacy_sha not in source_history:
                    source_history.append(legacy_sha)
        prior_source_sha = prior.get("prompts_sha256")
        if (
            prior_source_sha
            and prior_source_sha != manifest["prompts_sha256"]
            and prior_source_sha not in source_history
        ):
            source_history.append(prior_source_sha)
        if source_history:
            manifest["prompts_sha_history"] = source_history
    if extra:
        manifest.update(extra)
    output_path = Path(output_path)
    manifest_path = output_path.with_name(output_path.stem + ".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
