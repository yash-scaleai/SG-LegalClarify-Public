#!/usr/bin/env python3
"""
Preflight probe: send one minimal request to every model that a run would
exercise, using the SAME `_build_llm_request` code path (including
enable_reasoning=True) that run_trial will use. Any 400 from a provider is
surfaced here in ~60 seconds, before committing to a multi-hour run.

Exits 0 if every model succeeds. Exits non-zero and prints per-model errors
if anything fails, so it's safe to chain in front of the orchestrator:

  scripts/preflight_models.py && scripts/run_sharded_experiment.sh -- ...

Usage:
  scripts/preflight_models.py                                     # probes DEFAULT_MODELS
  scripts/preflight_models.py --models anthropic/claude-opus-4-8  # probes a subset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import run_eval  # noqa: E402


PROBE_SYSTEM = "You are concise."
PROBE_USER = "Reply with the single word: OK."


def probe_one(model_id: str) -> tuple[bool, str]:
    """Hit one model with a trivial reasoning-enabled prompt. Return (ok, note)."""
    start = time.time()
    try:
        text, _reasoning = run_eval.call_llm_with_reasoning(
            model_id, PROBE_SYSTEM,
            [{"role": "user", "content": PROBE_USER}],
            enable_reasoning=True,
        )
        dt = time.time() - start
        preview = (text or "").strip().replace("\n", " ")[:40]
        return True, f"{dt:.1f}s   reply={preview!r}"
    except Exception as exc:
        dt = time.time() - start
        msg = str(exc)
        # Extract the useful bit of LiteLLM's wall-of-text error
        for marker in ("does not support parameters",
                       "maximum tokens",
                       "BedrockException",
                       "invalid_request_error"):
            i = msg.find(marker)
            if i >= 0:
                msg = "…" + msg[i:i + 240]
                break
        return False, f"{dt:.1f}s   ERROR: {msg[:280]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=run_eval.DEFAULT_MODELS,
                        help="Model ids to probe (default: run_eval.DEFAULT_MODELS).")
    args = parser.parse_args()

    run_eval.load_dotenv_if_present()
    if not run_eval.os.environ.get("API_KEY"):
        print("[preflight] Warning: API_KEY not set; LiteLLM calls will 401.", file=sys.stderr)

    print(f"=== preflight probe: {len(args.models)} model(s), reasoning enabled ===")
    print(f"{'model':<52} {'ok?':<4}  detail")
    print("-" * 100)
    failures: list[tuple[str, str]] = []
    for m in args.models:
        ok, note = probe_one(m)
        tag = "OK " if ok else "FAIL"
        print(f"  {m:<50} {tag}  {note}")
        if not ok:
            failures.append((m, note))

    print()
    if failures:
        print(f"[preflight] {len(failures)} of {len(args.models)} model(s) FAILED. "
              f"Fix routing/params or drop these models before launching:")
        for m, note in failures:
            print(f"  - {m}: {note}")
        return 1
    print(f"[preflight] all {len(args.models)} model(s) accepted the probe. Safe to launch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
