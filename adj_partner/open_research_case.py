"""Host-only projection and isolated execution of one open-research case."""

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .application import AgentService
from .application_config import PROJECT_ROOT
from .tools.catalog import canonical_json


MAX_DATASET_BYTES = 32 * 1024 * 1024
MAX_DATASET_ROWS = 10_000
OPEN_RESEARCH_BUDGET = {"max_steps": 64, "max_input_tokens": 800_000,
                        "max_output_tokens": 80_000, "wall_time_seconds": 1800.0}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Dataset contains a duplicate JSON key")
        result[key] = value
    return result


def _read_dataset(path):
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    selected = selected.absolute()
    if selected.is_symlink() or selected.resolve() != selected:
        raise PermissionError("Dataset must be an explicit non-symlink file")
    descriptor = os.open(selected, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_DATASET_BYTES:
            raise ValueError("Dataset must be one regular file of at most 32 MiB")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read(MAX_DATASET_BYTES + 1)
        after = os.fstat(descriptor)
        if (len(content) > MAX_DATASET_BYTES or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns or before.st_ino != after.st_ino):
            raise ValueError("Dataset changed while being read or exceeds 32 MiB")
        return selected, content
    finally:
        os.close(descriptor)


def _rows(path, content):
    try:
        text = content.decode("utf-8-sig")
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line, object_pairs_hook=_object) for line in text.splitlines() if line.strip()]
        elif path.suffix.lower() == ".json":
            rows = json.loads(text, object_pairs_hook=_object)
        else:
            raise ValueError("Dataset must be .json or .jsonl")
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Dataset is not bounded valid UTF-8 JSON/JSONL") from exc
    if not isinstance(rows, list) or not rows or len(rows) > MAX_DATASET_ROWS:
        raise ValueError("Dataset must contain 1..10000 rows")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every dataset row must be a JSON object")
    canonical_json(rows)
    return rows


def _select(path, rows, selector):
    if not isinstance(selector, str) or not selector.strip() or len(selector) > 200:
        raise ValueError("A bounded case selector is required")
    selector = selector.strip()
    if path.suffix.lower() == ".jsonl":
        if not re.fullmatch(r"[1-9][0-9]{0,5}", selector):
            raise ValueError("JSONL selector must be a one-based row number")
        index = int(selector) - 1
        if index >= len(rows):
            raise ValueError("JSONL selector is outside the dataset")
        return rows[index], f"row:{index + 1}"
    identifiers = [row.get("id") for row in rows]
    if any(isinstance(value, str) and value for value in identifiers):
        matches = [row for row in rows if row.get("id") == selector]
        if len(matches) != 1:
            raise ValueError("JSON selector must match exactly one case id")
        return matches[0], f"id:{selector}"
    if not re.fullmatch(r"[1-9][0-9]{0,5}", selector) or int(selector) > len(rows):
        raise ValueError("JSON selector must be a one-based row number")
    return rows[int(selector) - 1], f"row:{int(selector)}"


def load_public_case(path, selector):
    selected, content = _read_dataset(path)
    row, coordinate = _select(selected, _rows(selected, content), selector)
    question = next((row.get(key) for key in ("question", "original_question", "Question:", "Question")
                     if isinstance(row.get(key), str) and row.get(key).strip()), "").strip()
    if not question or len(question) > 20_000:
        raise ValueError("Selected benchmark item has no bounded public question")
    difficulty = row.get("difficulty_level", "")
    if not isinstance(difficulty, str) or difficulty not in {"", "L1", "L2", "L3", "L4"}:
        raise ValueError("Selected benchmark item has an invalid difficulty label")
    # Drop `row` here: no caller receives the paper id, answer, rationale,
    # symbolic formalization, image pointer, or any other hidden field.
    return {"question": question, "difficulty_level": difficulty,
            "dataset_sha256": hashlib.sha256(content).hexdigest(), "selector": coordinate}


def research_goal(case):
    difficulty = f" Difficulty label: {case['difficulty_level']}." if case["difficulty_level"] else ""
    return ("Independently research and answer the following question." + difficulty + "\n\n"
        + case["question"] + "\n\n"
        "Discover relevant literature yourself. Compare sources, distinguish source text from inference, "
        "record limitations, and deliver one complete Markdown research report with host-bound clickable citations. "
        "Do not assume a particular paper, identifier, answer, section count, report length, or minimum citation count.")


def _fresh_directory(path):
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    selected = selected.absolute()
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected.parent.resolve() != selected.parent or selected.is_symlink():
        raise PermissionError("Output parent must not traverse a symlink")
    selected.mkdir(mode=0o700)
    return selected


def _write(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def run_case(dataset, selector, output, *, project_root=PROJECT_ROOT, env_file=None,
             model=None, budget=None, workflow=None, discovery_bundle=None, service_factory=AgentService):
    from .workflows import load_workflow
    selected_workflow = load_workflow(workflow) if workflow is not None else None
    if discovery_bundle is not None and (selected_workflow is None or not selected_workflow["acceptance"].get("require_critic")):
        raise ValueError("Discovery bundle requires the critic workflow")
    case = load_public_case(dataset, selector)
    destination = _fresh_directory(output)
    service = None
    manifest = {**case, "status": "initializing"}
    if selected_workflow is not None:
        manifest.update(workflow=selected_workflow, requested_model=model)
    try:
        service_options = {"request_output_tokens": 12_000} if selected_workflow else {}
        service = service_factory(destination / "state", project_root=project_root,
                                  env_file=env_file, **service_options)
        session_options = {"instructions": selected_workflow["instructions"]} if selected_workflow else {}
        session = service.create_session(profile="adjuvant", allow_network=True, **session_options,
            allowed_roots=[], budget=dict(OPEN_RESEARCH_BUDGET) if budget is None else budget)["id"]
        if model:
            service.set_model(session, model)
        goal = research_goal(case)
        acceptance = {
            "require_report": True, "require_evidence": True,
            "require_limitations": True, "required_sections": []}
        if selected_workflow is not None:
            goal = ("独立研究以下公开问题，按 host 注入的佐剂设计规范形成中文概念设计报告。"
                    "逐项回答原题，区分事实、推断、待验证设计；没有定量依据不编造参数。\n\n"
                    + case["question"] + "\n\n难度标签：" + (case["difficulty_level"] or "未提供")
                    + "。章节名称遵守 Host task policy 的 required_sections。")
            acceptance = selected_workflow["acceptance"]
        if discovery_bundle is not None:
            acceptance = {**acceptance, "require_concise_discovery": False}
        started = service.submit(session, goal, acceptance=acceptance)
        if discovery_bundle is not None:
            from .research.discovery_bundle import seed_bundle
            manifest["discovery"] = seed_bundle(service, session, started["id"], case["question"], discovery_bundle)
            manifest["effective_acceptance"] = acceptance
        final = service.drive(started["id"])
        run_id = final["id"]
        delivery = final.get("state", {}).get("delivery")
        manifest.update(status=final.get("status"), session_id=session, run_id=run_id,
                        report_id=delivery.get("report_id") if isinstance(delivery, dict) else None,
                        report_sha256=delivery.get("sha256") if isinstance(delivery, dict) else None)
        sources = service.research.list_sources(session, run_id)
        _write(destination / "sources.json", (canonical_json({"sources": sources}) + "\n").encode())
        if acceptance.get("require_critic"):
            from .research.critic import export_reviews
            reviews = export_reviews(service.store, session, run_id)
            _write(destination / "critic-reviews.json", (canonical_json(reviews) + "\n").encode())
            manifest["critic_accounting"] = reviews["accounting"]
        trace = service.trace(session, destination / "trace", run_id)
        manifest["trace"] = trace
        report_path = None
        if final.get("status") == "succeeded" and isinstance(delivery, dict) and delivery.get("kind") == "accepted_report":
            content = delivery.get("content")
            if not isinstance(content, str):
                raise ValueError("Accepted report delivery lacks exact UTF-8 content")
            report_bytes = content.encode("utf-8")
            if hashlib.sha256(report_bytes).hexdigest() != delivery.get("sha256"):
                # The runtime hash is authoritative; a mismatch must fail rather
                # than publishing a transformed report.
                raise ValueError("Accepted report delivery hash mismatch")
            report_path = destination / "report.md"
            _write(report_path, report_bytes)
            if isinstance(delivery.get("readable_content"), str):
                readable = delivery["readable_content"].encode("utf-8")
                _write(destination / "report-readable.md", readable)
                if isinstance(delivery.get("research_evidence_map"), dict):
                    _write(destination / "research-evidence-map.json",
                           (canonical_json(delivery["research_evidence_map"]) + "\n").encode())
                manifest.update(readable_sha256=hashlib.sha256(readable).hexdigest(),
                                research_design_status=delivery.get("research_design_status"),
                                readable_view_of=delivery["sha256"])
        _write(destination / "manifest.json", (canonical_json(manifest) + "\n").encode())
        result = {"status": final.get("status"), "output_dir": str(destination),
                  "report_path": str(report_path) if report_path else None,
                  "manifest_path": str(destination / "manifest.json")}
        if "research_design_status" in manifest:
            result["research_design_status"] = manifest["research_design_status"]
        return result
    finally:
        if service is not None:
            service.close()


def check_finalization_parent(run, manifest):
    """One explicit synthesis phase after budget exhaustion, never provider refusal."""
    if run['status'] != 'budget_exhausted' or manifest.get('completion_of') or run.get('state', {}).get('completion_of'):
        raise ValueError('Finalization requires an initial budget-exhausted research run')
    error=(run.get('state', {}).get('error') or {}).get('code')
    if error not in (None,'context_budget','input_budget','output_budget','step_budget','wall_time_budget'):
        raise ValueError('Finalization cannot retry provider errors or refusals')
