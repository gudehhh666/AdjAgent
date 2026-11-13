"""Immutable structured reports and exact-byte, version-rechecked handoff.

Task goal+acceptance fingerprints bind requirements; observed task revision is
recorded for audit. A status-only update does not invalidate the report. Handoff
rechecks after artifact I/O; current-secret changes veto, never rewrite bytes.
"""

from copy import deepcopy
import hashlib

from .acceptance import ReportNotAccepted, structural_issues, task_binding
from .rendering import render_report
from .design import validate_design
from .validation import bounded_json, ids, strings, text


def report_fields(title, sections, limitations):
    text(title, "report title", 2000)
    if not isinstance(sections, list) or len(sections) > 50:
        raise ValueError("sections must be a bounded list")
    normalized = []
    for section in sections:
        if not isinstance(section, dict) or set(section) not in ({"heading", "claim_ids"}, {"heading", "claim_ids", "paragraphs"}, {"heading", "paragraphs"}):
            raise ValueError("Sections require heading and claim_ids or paragraphs")
        text(section["heading"], "section heading", 200)
        paragraphs = section.get("paragraphs", [])
        if not isinstance(paragraphs, list) or len(paragraphs) > 100:
            raise ValueError("Section paragraphs must be a bounded list")
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict) or set(paragraph) != {"text", "claim_ids"}:
                raise ValueError("Paragraphs require exactly text and claim_ids")
            text(paragraph["text"], "paragraph text")
            ids(paragraph["claim_ids"])
        claim_ids = section.get("claim_ids") if "claim_ids" in section else list(dict.fromkeys(
            claim_id for paragraph in paragraphs for claim_id in paragraph["claim_ids"]))
        ids(claim_ids)
        for paragraph in paragraphs:
            if not set(paragraph["claim_ids"]) <= set(claim_ids):
                raise ValueError("Paragraph claim_ids must be declared by their section")
        normalized.append({"heading": section["heading"], "claim_ids": list(claim_ids),
                           "paragraphs": deepcopy(paragraphs)})
    strings(limitations, "report limitations")
    bounded_json({"title": title, "sections": normalized, "limitations": limitations})
    return normalized


class Reports:
    def get_report(self, session_id, run_id, report_id):
        return self._read("report", session_id, run_id, report_id)

    def list_reports(self, session_id, run_id):
        """Current task reports in creation order, including its earlier runs."""
        return self._read_list("report", session_id, run_id)

    def get_validation(self, session_id, run_id, validation_id):
        """Historical result only; accepted_report is the authority for handoff."""
        return self._read("validation", session_id, run_id, validation_id)

    def list_validations(self, session_id, run_id):
        return self._read_list("validation", session_id, run_id)

    def _report_inputs(self, conn, session_id, run_id, sections):
        task = self._scope(conn, session_id, run_id)
        snapshots, bindings = [], {"claims": {}, "task": task_binding(task)}
        claim_ids = dict.fromkeys(claim_id for section in sections for claim_id in section["claim_ids"])
        for claim_id in claim_ids:
            snapshot = self._claim_snapshot(conn, session_id, run_id, claim_id)
            assessment = self._head(conn, "assessment", session_id, task["id"], claim_id)
            current = bool(assessment and self._bindings_current(conn, session_id, run_id, assessment["bindings"]))
            snapshot.update(assessment=assessment, assessment_current=current)
            snapshots.append(snapshot)
            bindings["claims"][claim_id] = {**snapshot["bindings"], "assessment_id": assessment["id"] if assessment else None}
        return task, snapshots, bindings

    def _report_current_issues(self, conn, session_id, run_id, report, *, creating=False):
        task, snapshots, current = self._report_inputs(conn, session_id, run_id, report["sections"])
        issues = []
        # Status changes are irrelevant; exact goal/requirements remain bound.
        if current["task"]["requirements_hash"] != report["bindings"]["task"]["requirements_hash"]:
            issues.append("stale_task_requirements")
        if current["claims"] != report["bindings"]["claims"]:
            issues.append("stale_claim_evidence_or_assessment")
        artifact = self._artifact(conn, session_id, report["artifact_id"], report["run_id"])
        if artifact["sha256"] != report["artifact_hash"]:
            issues.append("report_hash_mismatch")
        if self.store.redact(report) != report:
            issues.append("report_requires_current_secret_redaction")
        from .critic import gate
        issues.extend(gate(self, conn, session_id, run_id, report, creating=creating))
        return task, issues, artifact

    def _report_materials(self, report):
        materials = {}
        for snapshot in report["claim_snapshots"]:
            for source in snapshot["sources"]:
                if source["artifact_id"] not in materials:
                    materials[source["artifact_id"]] = self._verify_source(source)
        return materials

    def create_report(self, session_id, run_id, *, title, sections, limitations, research_design=None):
        sections = report_fields(title, sections, limitations)
        with self.store.transaction() as conn:
            task, snapshots, bindings = self._report_inputs(conn, session_id, run_id, sections)
        if research_design is not None:
            research_design = validate_design(research_design, {x['claim']['id'] for x in snapshots})
        renderer_version = 3 if task["acceptance"].get("report_layout") == "review" else 2
        record = self._record("report", session_id, run_id, task["id"], renderer_version=renderer_version, title=title,
                              sections=deepcopy(sections), limitations=list(limitations),
                              claim_snapshots=snapshots, bindings=bindings)
        if research_design is not None:
            record.update(research_design=research_design,
                          renderer_version=5 if 'checkpoints' in research_design['verification'] else 4)
        record = self.store.redact(record)
        bounded_json(record)
        self._report_materials(record)
        rendered = render_report(record)
        artifact = self.store.put_artifact(rendered, "research-report.md", "text/markdown",
                                           {"kind": "structured_research_report", "report_id": record["id"]},
                                           session_id=session_id, run_id=run_id)
        record.update(artifact_id=artifact["id"], artifact_hash=artifact["sha256"])
        with self.store.transaction(write=True) as conn:
            _, issues, _ = self._report_current_issues(conn, session_id, run_id, record, creating=True)
            if issues:
                # Artifact publication precedes metadata transaction. An orphan
                # private immutable artifact on a race is safe and not a report.
                raise ReportNotAccepted(issues)
            record = self._save(conn, "report", record)
        return self._mirror("report", record)

    def _verified_report_io(self, report):
        content = self.store.read_artifact(report["artifact_id"])
        if hashlib.sha256(content).hexdigest() != report["artifact_hash"]:
            raise ValueError("Report artifact hash mismatch")
        if render_report(report) != content:
            raise ValueError("Report bytes do not match its immutable structured record")
        # Checking both structured values and rendered bytes also catches newly
        # registered secrets containing characters escaped in the HTML.
        if self.store.redact(report) != report or self.store.redact(content.decode("utf-8")) != content.decode("utf-8"):
            raise ValueError("Report requires current-secret redaction; recreate it")
        return content, self._report_materials(report)

    def validate_report(self, session_id, run_id, report_id):
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            report = self._get(conn, "report", session_id, task["id"], report_id)
        io_issues, materials = [], {}
        try:
            _, materials = self._verified_report_io(report)
        except (ValueError, OSError) as exc:
            io_issues.append(f"artifact_integrity_or_redaction: {exc}")
        with self.store.transaction(write=True) as conn:
            task = self._scope(conn, session_id, run_id)
            current_report = self._get(conn, "report", session_id, task["id"], report_id)
            issues = list(io_issues)
            if current_report != report:
                issues.append("stale_report_record")
            try:
                _, binding_issues, _ = self._report_current_issues(conn, session_id, run_id, report)
                issues.extend(binding_issues)
            except (KeyError, ValueError, PermissionError) as exc:
                issues.append(f"binding_integrity: {exc}")
            issues.extend(structural_issues(report, task, materials))
            record = self._record("validation", session_id, run_id, task["id"], report_id=report_id,
                                  report_hash=report["artifact_hash"], bindings=report["bindings"],
                                  accepted=not issues, issues=list(dict.fromkeys(issues)),
                                  scope="structural_acceptance_not_scientific_assurance")
            record = self._save(conn, "validation", record, subject_id=report_id)
        return self._mirror("validation", record)

    def accepted_report(self, session_id, run_id, report_id):
        """Return exact {content: bytes, report, validation, artifact}, or raise.

        Call immediately before delivery. It performs no writes or synthesis.
        Enumerate list_reports in reverse creation order to find candidates for
        this task; a cached validation accepted=True is never a handoff grant.
        """
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            report = self._get(conn, "report", session_id, task["id"], report_id)
            validation = self._head(conn, "validation", session_id, task["id"], report_id)
            _, issues, _ = self._report_current_issues(conn, session_id, run_id, report)
            if not validation or not validation["accepted"] or validation["issues"]:
                issues.append("current_accepted_validation_required")
            elif validation["report_hash"] != report["artifact_hash"] or validation["bindings"] != report["bindings"]:
                issues.append("stale_validation")
        if issues:
            raise ReportNotAccepted(issues)
        try:
            content, materials = self._verified_report_io(report)
        except (OSError, ValueError) as exc:
            raise ReportNotAccepted([f"artifact_integrity_or_redaction: {exc}"]) from exc
        # This second snapshot is essential: artifact reads can take time, and a
        # concurrent claim/evidence/provider/task write must veto cached success.
        with self.store.transaction() as conn:
            task, issues, artifact = self._report_current_issues(conn, session_id, run_id, report)
            current_report = self._get(conn, "report", session_id, task["id"], report_id)
            current_validation = self._head(conn, "validation", session_id, task["id"], report_id)
            if current_report != report or current_validation != validation:
                issues.append("concurrent_report_or_validation_change")
            issues.extend(structural_issues(report, task, materials))
        if issues:
            raise ReportNotAccepted(issues)
        return {"content": content, **self.store.redact({"report": report, "validation": validation, "artifact": artifact})}
