"""Bounded, reproducible case runner using AgentService, preserving every trace.

python -m adj_partner.evaluation --mode offline --output /absolute/new/directory
Live calls require explicit --confirm-model-requests and an approved host config.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .contracts import Budget
from .tools.catalog import canonical_json


PROJECT = Path(__file__).resolve().parents[1]


def load_cases():
    return json.loads((PROJECT / "eval/session_cases.json").read_text(encoding="utf-8"))["cases"]


def _numbers(service, sid, rid, case_id):
    computed = [source for source in service.research.list_sources(sid, rid) if source["kind"] == "computed"]
    if case_id == "historical_ferritin": return {"applicable": False, "passed": None, "reason": "No numerical scientific benchmark for this design task"}
    if not computed: return {"applicable": True, "passed": False, "reason": "No actual computed source"}
    data = json.loads(service.store.read_artifact(computed[-1]["artifact_id"], max_bytes=8 * 1024 * 1024))
    checks = {}
    if case_id == "protein_extension":
        expected = {"length": 18, "molecular_weight_da": 2140.5405, "theoretical_isoelectric_point": 8.344739341735842,
                    "gravy": 1.2333333333333334, "aromaticity": .2777777777777778}
        for key, value in expected.items():
            checks[key] = type(data.get(key)) in {int, float} and math.isclose(data[key], value, rel_tol=1e-12, abs_tol=1e-12)
        checks["composition_sum"] = sum(data.get("composition", {}).values()) == 18
    else:
        baseline = json.loads((PROJECT / "docs/refactor_2026-08-31/data-baseline.json").read_text())["evidence_pack_baseline"]
        expected = baseline["top_five_centered_reversal_r"]
        table = data["tables"]["panel_centered_reversal"]
        observed = {row["compound"]: row["centered_reversal_r"] for row in table[:5]}
        checks["top_five_identity"] = list(observed) == list(expected)
        checks["top_five_values"] = all(key in observed and math.isclose(observed[key], value, rel_tol=1e-12, abs_tol=1e-12) for key, value in expected.items())
        checks["table_rows"] = all(len(data["tables"][key]) == count for key, count in baseline["table_rows"].items())
        checks["two_sided_significant_compound_count"] = data["statistics"]["n_significant_compounds_two_sided"] == 17
        checks["biological_replicates"] = data["statistics"]["biological_replicates"] == 1
    return {"applicable": True, "passed": all(checks.values()), "checks": checks,
        "scope": "Deterministic numeric agreement only; not biological or scientific-answer validation"}


def run_cases(output, *, case_ids=None, mode="offline", confirm_model_requests=False,
              env_file=None, dataset_dir=None, budget=None):
    if mode not in {"offline", "live"}: raise ValueError("Mode must be offline or live")
    if mode == "live" and not confirm_model_requests:
        raise ValueError("Explicit confirm_model_requests is required before remote model calls")
    cases = load_cases()
    selected = set(case_ids) if case_ids is not None else {case["id"] for case in cases}
    if not selected or selected - {case["id"] for case in cases}: raise ValueError("Unknown/empty case selection")
    limits = budget if isinstance(budget, Budget) else Budget(**(budget or {}))
    destination = Path(output).expanduser().absolute()
    if destination.exists(): raise FileExistsError("Evaluation output must be a new directory; prior attempts are preserved")
    if destination.parent.resolve() != destination.parent: raise ValueError("Evaluation output parent must not be a symlink")
    from .application import AgentService
    from .evaluation_scenarios import OfflineScenario
    destination.mkdir(parents=True, exist_ok=False)
    fixture = PROJECT / "eval/fixtures/ferritin_engineering.json"
    summary = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(), "mode": mode,
        "cases_sha256": hashlib.sha256((PROJECT / "eval/session_cases.json").read_bytes()).hexdigest(),
        "scientific_assurance": "Not inferred from runtime completion or structural acceptance; manual/domain review required",
        "private_chain_of_thought": "not_requested_or_recorded", "cases": [], "summary_path": str(destination / "summary.json")}
    for case in cases:
        if case["id"] not in selected: continue
        folder = destination / case["id"]
        folder.mkdir()
        port = OfflineScenario(case, fixture) if mode == "offline" else None
        service = AgentService(folder / "state", project_root=PROJECT, model=port, env_file=env_file if mode == "live" else None,
                               dataset_dir=dataset_dir)
        roots = [str(fixture.parent)] if mode == "offline" and case["id"] == "historical_ferritin" else []
        if case["id"] == "historical_deg" and service.native.dataset_dir: roots.append(service.native.dataset_dir)
        sid = service.create_session(profile=case["profile"], allowed_roots=roots, allow_network=mode == "live", budget=limits)["id"]
        item = {"case_id": case["id"], "question": case["question"], "acceptance": case["acceptance"], "session_id": sid,
            "model_mode": "offline-scripted" if mode == "offline" else "live-configured-provider", "budget": limits.to_dict(),
            "runtime_completed": False, "structural_acceptance": False, "scientific_answer_validated": False,
            "scientific_review_status": "not_assessed", "review_focus": case["review_focus"], "state_directory": str(folder / "state")}
        started, rid = time.monotonic(), None
        try:
            rid = service.submit(sid, case["question"], acceptance=case["acceptance"])["id"]
            item["run_id"] = rid
            run = service.drive(rid)
            item.update(status=run["status"], stop_reason=run["state"].get("stop_reason", run["state"].get("waiting")),
                        usage=run["usage"], model=service.model_info(sid), runtime_completed=run["status"] == "succeeded")
            delivery = run["state"].get("delivery", {})
            if delivery.get("kind") == "accepted_report":
                actual = service.store.read_artifact(delivery["artifact_id"], max_bytes=8 * 1024 * 1024)
                item["structural_acceptance"] = actual == delivery["content"].encode("utf-8") and hashlib.sha256(actual).hexdigest() == delivery["sha256"]
                item.update(report_id=delivery["report_id"], report_artifact_id=delivery["artifact_id"], report_sha256=delivery["sha256"])
            item["numerical_checks"] = _numbers(service, sid, rid, case["id"])
        except Exception as exc:
            item.update(status="evaluation_error", error=service.store.redact(f"{type(exc).__name__}: {exc}"))
        finally:
            item["elapsed_seconds"] = round(time.monotonic() - started, 3)
            item["job_count"] = len(service.jobs(sid))
            try: item["trace"] = service.trace(sid, folder / "trace", run_id=rid)
            except Exception as exc: item["trace_error"] = service.store.redact(f"{type(exc).__name__}: {exc}")
            if mode == "offline":
                item["coverage_limits"] = ["Scripted decisions: not autonomous LLM performance or long-horizon reasoning evaluation",
                    "No live literature retrieval; ferritin uses an explicitly synthetic source and does not answer the scientific design task"]
            summary["cases"].append(service.store.redact(item))
            # Fresh directory; update one generated summary after each attempt so
            # a later case failure does not erase earlier evidence. No old run is overwritten.
            Path(summary["summary_path"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            service.close()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--case", action="append", dest="case_ids", choices=[case["id"] for case in load_cases()])
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-file")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--confirm-model-requests", action="store_true")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-input-tokens", type=int, default=120000)
    parser.add_argument("--max-output-tokens", type=int, default=24000)
    parser.add_argument("--wall-time-seconds", type=float, default=600)
    args = parser.parse_args(argv)
    try:
        result = run_cases(args.output, case_ids=args.case_ids, mode=args.mode, confirm_model_requests=args.confirm_model_requests,
            env_file=args.env_file, dataset_dir=args.dataset_dir, budget=Budget(args.max_steps, args.max_input_tokens, args.max_output_tokens, args.wall_time_seconds))
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["cases"] and all(
            item["runtime_completed"] and item["structural_acceptance"] and "trace" in item
            and (item.get("numerical_checks", {}).get("applicable") is False
                 or item.get("numerical_checks", {}).get("passed") is True)
            for item in result["cases"]) else 1
    except Exception as exc:
        # Model configuration errors are deliberately generic in the provider.
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__": raise SystemExit(main())
