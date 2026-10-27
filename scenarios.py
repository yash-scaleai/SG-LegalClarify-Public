from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests

DATASET_REPO_ID = "ScaleAI/SG-LegalClarify"
# Pinned so a later edit to the dataset cannot silently change what a run evaluates.
DATASET_REVISION = "c9c64a42aa9350eb9d5e67b51109b21b571e110a"
DATASET_FILE = "data/test.parquet"
CACHE_DIR = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "sg-legalclarify"


def _hf_token() -> str | None:
    """HF_TOKEN, else the token saved by `hf auth login`. Only needed while the dataset is private."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    token_file = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    return token_file.read_text().strip() if token_file.is_file() else None


def _download_dataset(repo_id: str, revision: str) -> Path:
    """Fetch the dataset parquet once per revision and reuse the cached copy afterwards."""
    path = CACHE_DIR / repo_id.replace("/", "__") / revision / DATASET_FILE
    if path.is_file():
        return path
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{DATASET_FILE}"
    token = _hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    return path


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return re.sub(r"_+", "_", slug).strip("_")


def _normalize_importance(raw: str) -> str:
    text = raw.strip().lower()
    if text in {"mandatory", "essential"}:
        return "Mandatory"
    return "Optional"


def _normalize_question_type(raw: str) -> str:
    text = raw.strip().lower()
    if text == "fully specified":
        return "Fully specified"
    if text == "underspecified":
        return "Underspecified"
    return raw.strip() or "Unknown"


def _parse_clarifying_variables(facts: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Turn the dataset's clarifying_facts list into {var_i: {question, answer}}."""
    variables: dict[str, dict[str, str]] = {}
    for i, fact in enumerate(facts):
        q = _to_str(fact.get("question"))
        a = _to_str(fact.get("answer"))
        if not q and not a:
            continue
        variables[f"var_{i}"] = {
            "question": q,
            "answer": a,
        }
    return variables


def _derive_title(question: str, fallback: str) -> str:
    cleaned = re.sub(r"\s+", " ", question).strip()
    if not cleaned:
        return fallback
    short = cleaned[:72].rstrip(" .,!?:;")
    return short


def _build_converted_fully_specified_scenario(
    base_scenario: dict[str, Any],
    fully_specified_question: str,
    scenarios: dict[str, dict],
) -> tuple[str, dict] | None:
    """
    Build a synthetic fully-specified scenario for underspecified rows
    that also define a fully specified question.
    """
    question = fully_specified_question.strip()
    if not question:
        return None

    base_task_id = str(base_scenario.get("task_id", "")).strip()
    if not base_task_id:
        return None

    candidate_key = f"{base_task_id}__fully_specified"
    if candidate_key in scenarios:
        suffix = 2
        while f"{candidate_key}_{suffix}" in scenarios:
            suffix += 1
        candidate_key = f"{candidate_key}_{suffix}"

    converted = dict(base_scenario)
    converted.update(
        {
            "task_id": candidate_key,
            "title": _derive_title(question, base_scenario.get("subtopic", "scenario")),
            "output_slug": _slugify(candidate_key),
            "question_type": "Fully specified",
            "user_question": question,
            "variables": {},
            "is_converted_fully_specified": True,
            "converted_from_task_id": base_task_id,
        }
    )
    return candidate_key, converted


def load_scenarios(
    repo_id: str = DATASET_REPO_ID,
    revision: str = DATASET_REVISION,
) -> dict[str, dict]:
    """Download the dataset from the Hugging Face Hub (cached locally) and build scenarios."""
    rows = pq.read_table(_download_dataset(repo_id, revision)).to_pylist()

    scenarios: dict[str, dict] = {}
    for row in rows:
        task_id = _to_str(row["task_id"]) or f"task_{len(scenarios)+1}"
        question_type_raw = _to_str(row["question_type"])
        question_type = question_type_raw.lower()
        underspecified_question = _to_str(row["underspecified_question"])
        fully_specified_question = _to_str(row["fully_specified_question"])
        if question_type == "underspecified":
            question = underspecified_question or fully_specified_question
        else:
            question = fully_specified_question or underspecified_question
        if not question:
            continue

        category = _to_str(row["category"]) or "Unknown"
        topic = _to_str(row["topic"]) or "Unknown"
        subtopic = _to_str(row["subtopic"]) or "scenario"

        variables = (
            _parse_clarifying_variables(row["clarifying_facts"] or [])
            if question_type == "underspecified"
            else {}
        )

        rubric = []
        for i, criterion in enumerate(row["rubric"] or [], start=1):
            item_text = _to_str(criterion["criterion"])
            if not item_text:
                continue
            rubric.append(
                {
                    "id": i,
                    "item": item_text,
                    "dimension": _to_str(criterion["dimension"]) or "Issue",
                    "importance": _normalize_importance(_to_str(criterion["importance"])),
                }
            )

        scenario_key = task_id
        title = _derive_title(question, subtopic)
        scenarios[scenario_key] = {
            "task_id": task_id,
            "title": title,
            "output_slug": _slugify(task_id),
            "category": category,
            "topic": topic,
            "subtopic": subtopic,
            "question_type": _normalize_question_type(question_type_raw),
            "user_question": question,
            "variables": variables,
            "rubric": rubric,
            "underspecified_question": underspecified_question,
            "fully_specified_question": fully_specified_question,
            "is_converted_fully_specified": False,
        }

        # If an underspecified row also has a fully specified question, emit
        # a second synthetic scenario that is treated as a normal fully-specified run.
        if question_type == "underspecified":
            converted_result = _build_converted_fully_specified_scenario(
                scenarios[scenario_key],
                fully_specified_question=fully_specified_question,
                scenarios=scenarios,
            )
            if converted_result is not None:
                converted_key, converted_scenario = converted_result
                scenarios[converted_key] = converted_scenario

    return scenarios
