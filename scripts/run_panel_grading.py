#!/usr/bin/env python3
"""Grade a results tree with the paper's 3-judge panel.

Panel: DeepSeek V4 Flash + GPT-5.6 Sol + Kimi K3, chosen by agreement (Cohen's
kappa) with human rubric labels. Every judge uses run_eval.JUDGE_SYSTEM_PROMPT.

The DeepSeek V4 Flash judge uses the azure_ai route, the exact configuration
validated against the human labels, independent of the response-model routes
in run_eval.MODELS.

Modes
-----
grade (default): run the 3 judges over every (task_dir x trial) in the shard,
  writing one resumable JSONL sidecar per task dir:
      <results_dir>/<task_dir>/<task_dir>_panel_votes.jsonl
  one record per (trial, judge). Rubrics come from the dataset (via
  run_eval.SCENARIOS), not from the rubric_results stored in the JSONs. The
  question shown to the judge is the trial's own stored question.

--apply: merge sidecars back into each <task_dir>_base.json:
  rubric_results rebuilt from the current rubric with
      passed   = panel majority (2/3; a missing/null vote falls back to
                 majority of non-null, ties -> False)
      votes    = {judge_label: verdict} per item
  mandatory_passed / mandatory_total / score recomputed; trial stamped with
  panel_grading provenance. Refuses to apply where sidecar coverage is
  incomplete unless --allow-partial.

--status: per-shard completion counts.

Usage
-----
  python scripts/run_panel_grading.py --results-dir results_paper_run_1 --shard 1/10 --workers 3
  python scripts/run_panel_grading.py --results-dir results_paper_run_1 --status
  python scripts/run_panel_grading.py --results-dir results_paper_run_1 --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
for p in (REPO_ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_eval  # noqa: E402

# The validated panel: (judge model id, judge label). Ids are pinned here on
# purpose — panel validity is tied to these exact routes.
PANEL = [
    ("azure_ai/DeepSeek-V4-Flash", "DeepSeek V4 Flash"),
    ("openai/gpt-5.6-sol", "GPT-5.6 Sol"),
    ("fireworks_ai/kimi-k3", "Kimi K3"),
]
JUDGE_PROMPT_SHA256 = hashlib.sha256(
    run_eval.JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def normalize_verdicts(raw: list[dict], rubric: list[dict]) -> tuple[list[dict], int]:
    """Map the judge's raw rows onto the rubric ids: exactly one verdict per
    rubric item, missing ids null-filled. Returns (verdicts, n_null)."""
    by_id = {}
    for row in raw:
        # Tolerate non-integer id spellings ("ID 1", "1.", "criterion 1", ...):
        # some judges echo the prompt's "ID n" label instead of the bare int.
        try:
            rid = int(row.get("id"))
        except (TypeError, ValueError):
            m = re.search(r"\d+", str(row.get("id")))
            if not m:
                continue
            rid = int(m.group())
        if rid not in by_id:  # first verdict per id wins
            by_id[rid] = row
    verdicts, n_null = [], 0
    for item in rubric:
        row = by_id.get(int(item["id"]))
        if row is None:
            verdicts.append({"id": item["id"], "passed": None,
                             "reasoning": "No verdict returned by judge"})
            n_null += 1
        else:
            passed = row.get("passed")
            if passed is None:
                n_null += 1
            verdicts.append({"id": item["id"], "passed": passed,
                             "reasoning": row.get("reasoning", "")})
    return verdicts, n_null


def scenario_key(task_dir: str) -> str:
    if task_dir.endswith("_fully_specified") and not task_dir.endswith("__fully_specified"):
        return task_dir[: -len("_fully_specified")] + "__fully_specified"
    return task_dir


def current_rubric(task_dir: str) -> list[dict] | None:
    scen = run_eval.SCENARIOS.get(scenario_key(task_dir))
    if not scen or not scen.get("rubric"):
        return None
    return scen["rubric"]


def load_dirs(results_dir: Path) -> list[str]:
    return sorted(d.name for d in results_dir.iterdir()
                  if d.is_dir() and (d / f"{d.name}_base.json").exists())


def shard_of(dirs: list[str], spec: str | None) -> list[str]:
    if not spec:
        return dirs
    i, n = (int(x) for x in spec.split("/"))
    if not (1 <= i <= n):
        raise SystemExit(f"bad --shard {spec}")
    return dirs[i - 1::n]


def sidecar_path(results_dir: Path, task_dir: str) -> Path:
    return results_dir / task_dir / f"{task_dir}_panel_votes.jsonl"


def load_done(path: Path) -> set[tuple]:
    done = set()
    if path.exists():
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("verdicts") and not all(
                    v.get("passed") is None for v in rec["verdicts"]):
                done.add((rec["model"], rec["judge_model"]))
    return done


# ---------------------------------------------------------------------------

def grade(args) -> None:
    dirs = shard_of(load_dirs(args.results_dir), args.shard)
    work = []          # (task_dir, trial_dict, judge_id, judge_label, rubric)
    n_skipped_trials = 0
    for td in dirs:
        rubric = current_rubric(td)
        if rubric is None:
            print(f"[warn] no scenario/rubric for {td}; skipped", flush=True)
            continue
        base = json.loads((args.results_dir / td / f"{td}_base.json").read_text())
        done = load_done(sidecar_path(args.results_dir, td))
        for t in base.get("trials", []):
            if not t.get("conversation"):
                n_skipped_trials += 1
                continue
            if args.models is not None and t.get("model") not in args.models:
                continue
            for jid, jlabel in PANEL:
                if (t["model"], jid) not in done:
                    work.append((td, t, jid, jlabel, rubric))
    print(f"shard {args.shard or 'all'}: {len(dirs)} dirs, {len(work)} judgments "
          f"to run, {n_skipped_trials} conversationless trials skipped "
          f"(prompt {run_eval.JUDGE_PROMPT_VERSION})", flush=True)
    if args.dry_run or not work:
        return

    locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()
    counter = {"done": 0}
    counter_lock = threading.Lock()

    def run_one(td: str, trial: dict, jid: str, jlabel: str, rubric: list[dict]) -> None:
        scenario = {"user_question": trial["scenario"], "rubric": rubric}
        best, best_null, attempts = None, None, 0
        t0 = time.time()
        for attempt in range(1, args.max_retries + 1):
            attempts = attempt
            raw = run_eval.evaluate_rubric_with_judge(
                trial["conversation"], scenario, judge_model=jid)
            verdicts, n_null = normalize_verdicts(raw, rubric)
            if best_null is None or n_null < best_null:
                best, best_null = verdicts, n_null
            if best_null == 0:
                break
            time.sleep(min(30.0, 2.0 * attempt))
        rec = {
            "task_dir": td, "model": trial["model"],
            "model_label": trial["model_label"],
            "judge_model": jid, "judge_label": jlabel,
            "verdicts": best, "n_items": len(best), "n_null": best_null,
            "attempts": attempts,
            "duration_seconds": round(time.time() - t0, 2),
            "judge_prompt_version": run_eval.JUDGE_PROMPT_VERSION,
            "judge_prompt_sha256": JUDGE_PROMPT_SHA256,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with locks_guard:
            lock = locks.setdefault(td, threading.Lock())
        with lock:
            with sidecar_path(args.results_dir, td).open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with counter_lock:
            counter["done"] += 1
            done = counter["done"]
        flag = "" if best_null == 0 else f"  [{best_null} null]"
        print(f"[{done}/{len(work)}] {jlabel:<18} {td} / {trial['model_label']}{flag}",
              flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *w): w for w in work}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                td, trial, jid, jlabel, _ = futures[fut]
                print(f"[FAILED] {jlabel} on {td}/{trial['model_label']}: {exc}",
                      flush=True)
    print(f"shard done in {(time.time() - t0) / 3600:.2f} h", flush=True)


# ---------------------------------------------------------------------------

def status(args) -> None:
    dirs = load_dirs(args.results_dir)
    n_expected = n_done = n_null = 0
    dirs_complete = 0
    for td in dirs:
        rubric = current_rubric(td)
        if rubric is None:
            continue
        base = json.loads((args.results_dir / td / f"{td}_base.json").read_text())
        trials = [t for t in base.get("trials", []) if t.get("conversation")]
        expected = len(trials) * len(PANEL)
        done = load_done(sidecar_path(args.results_dir, td))
        n_expected += expected
        n_done += len(done)
        if len(done) >= expected:
            dirs_complete += 1
    print(f"{args.results_dir}: {n_done}/{n_expected} judgments "
          f"({100 * n_done / max(n_expected, 1):.1f}%), "
          f"{dirs_complete}/{len(dirs)} dirs complete")


# ---------------------------------------------------------------------------

def apply(args) -> None:
    dirs = load_dirs(args.results_dir)
    stats = Counter()
    for td in dirs:
        rubric = current_rubric(td)
        if rubric is None:
            stats["no_rubric"] += 1
            continue
        # best-record-wins: prefer the record with the fewest null verdicts
        # (ties broken by recency) so a late garbage record can never shadow
        # an earlier clean one.
        best: dict[tuple, dict] = {}
        sp = sidecar_path(args.results_dir, td)
        if sp.exists():
            for line in sp.open():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["model"], rec["judge_label"])
                if key not in best or rec.get("n_null", 0) <= best[key].get("n_null", 0):
                    best[key] = rec
        votes = {k: {v["id"]: v["passed"] for v in r["verdicts"]}
                 for k, r in best.items()}
        base_path = args.results_dir / td / f"{td}_base.json"
        base = json.loads(base_path.read_text())
        changed = False
        for t in base.get("trials", []):
            if not t.get("conversation"):
                stats["trials_no_conversation"] += 1
                continue
            if args.models is not None and t.get("model") not in args.models:
                continue
            per_judge = {jl: votes.get((t["model"], jl)) for _, jl in PANEL}
            missing = [jl for jl, v in per_judge.items() if v is None]
            if missing:
                stats["trials_missing_votes"] += 1
                if not args.allow_partial:
                    continue
            new_rubric = []
            for item in rubric:
                vv = {jl: (per_judge[jl] or {}).get(item["id"]) for _, jl in PANEL}
                yes = sum(1 for v in vv.values() if v is True)
                no = sum(1 for v in vv.values() if v is False)
                passed = yes > no
                if yes == no:
                    stats["tie_votes"] += 1
                new_rubric.append({
                    "id": item["id"], "item": item["item"],
                    "dimension": item["dimension"],
                    "importance": item["importance"],
                    "passed": passed,
                    "reasoning": f"Panel majority ({yes}/{yes + no} Met)",
                    "votes": vv,
                })
            t["rubric_results"] = new_rubric
            mand = [r for r in new_rubric if r["importance"] == "Mandatory"]
            t["mandatory_total"] = len(mand)
            t["mandatory_passed"] = sum(1 for r in mand if r["passed"])
            t["score"] = round(100.0 * t["mandatory_passed"] / t["mandatory_total"], 1) \
                if mand else 0.0
            t["panel_grading"] = {
                "judges": [jl for _, jl in PANEL],
                "judge_prompt_version": run_eval.JUDGE_PROMPT_VERSION,
                "applied_at": datetime.now(timezone.utc).isoformat(),
            }
            stats["trials_graded"] += 1
            changed = True
        if changed:
            base_path.write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n")
            stats["files_written"] += 1
    print("apply stats:", dict(stats))
    if stats["trials_missing_votes"] and not args.allow_partial:
        print("NOTE: trials with missing votes were left ungraded "
              "(re-run grading or use --allow-partial).")


def heal(args) -> None:
    judge_id = {label: jid for jid, label in PANEL}
    n_targets = n_healed = 0
    for td in load_dirs(args.results_dir):
        sp = sidecar_path(args.results_dir, td)
        if not sp.exists():
            continue
        lines = [json.loads(l) for l in sp.open() if l.strip()]
        latest: dict[tuple, dict] = {}
        for rec in lines:
            key = (rec["model"], rec["judge_label"])
            if key not in latest or rec.get("n_null", 0) <= latest[key].get("n_null", 0):
                latest[key] = rec
        bad = [rec for rec in latest.values() if rec.get("n_null", 0) > 0]
        if not bad:
            continue
        rubric = current_rubric(td)
        base = json.loads((args.results_dir / td / f"{td}_base.json").read_text())
        changed = False
        for rec in bad:
            n_targets += 1
            if args.dry_run:
                print(f"[dry] {td} / {rec['model_label']} / {rec['judge_label']}: "
                      f"{rec['n_null']} null")
                continue
            trial = next(t for t in base["trials"] if t["model"] == rec["model"])
            verdicts, n_null = rec["verdicts"], rec["n_null"]
            for attempt in range(1, args.heal_retries + 1):
                raw = run_eval.evaluate_rubric_with_judge(
                    trial["conversation"],
                    {"user_question": trial["scenario"], "rubric": rubric},
                    judge_model=judge_id[rec["judge_label"]])
                v, nn = normalize_verdicts(raw, rubric)
                if nn < n_null:
                    verdicts, n_null = v, nn
                if n_null == 0:
                    break
                time.sleep(min(30.0, 2.0 * attempt))
            if n_null < rec["n_null"]:
                rec.update(verdicts=verdicts, n_null=n_null)
                latest[(rec["model"], rec["judge_label"])] = rec
                changed = True
                n_healed += 1
            print(f"{td} / {rec['model_label']} / {rec['judge_label']}: "
                  f"nulls -> {n_null}", flush=True)
        if changed:
            sp.write_text("\n".join(
                json.dumps(r, ensure_ascii=False) for r in latest.values()) + "\n")
    print(f"heal: {n_targets} records had nulls, {n_healed} improved", flush=True)



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--shard", help="i/n, e.g. 3/10 (grade mode)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--heal", action="store_true",
                    help="Re-ask each judge whose latest record still has null verdicts. "
                         "Run after grading and before --apply.")
    ap.add_argument("--heal-retries", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=None,
                    help="Restrict grading (and apply) to these model IDs. Useful for "
                         "closing a gap without re-running the full panel over already-"
                         "graded models. When set, unlisted models are skipped in both "
                         "the grade loop and the apply-panel-votes step.")
    args = ap.parse_args()
    if args.status:
        status(args)
    elif args.apply:
        apply(args)
    elif args.heal:
        heal(args)
    else:
        grade(args)


if __name__ == "__main__":
    main()
