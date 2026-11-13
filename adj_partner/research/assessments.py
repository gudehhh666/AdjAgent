"""Observable host assessment calls, never hidden models or lexical entailment.

HostAssessor.callback receives one scoped JSON snapshot and returns exactly
{status, rationale, evidence_ids}. It must not return private chain-of-thought.
The host owns reviewer identity, method, credentials and budgeted model gateway.
Calls occur once, outside transactions; semantic judgments remain provisional.
"""

from copy import deepcopy
from dataclasses import dataclass, field

from .records import extract
from .validation import bounded_json, claim_fields, ids, text


STATUSES = frozenset({"supported", "partial", "contradicted", "unassessed"})


@dataclass(frozen=True)
class HostAssessor:
    callback: object
    reviewer_id: str = "host-injected-reviewer"
    method: str = "semantic_review"
    secrets: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self):
        if not callable(self.callback):
            raise TypeError("Assessment callback must be callable")
        text(self.reviewer_id, "reviewer identity", 256)
        text(self.method, "assessment method", 256)
        if isinstance(self.secrets, (str, bytes)) or any(not isinstance(s, str) for s in self.secrets):
            raise TypeError("Assessment secrets must be host-configured strings")
        object.__setattr__(self, "secrets", tuple(self.secrets))


def qualifications(source):
    availability = ("userfile" if source["kind"] == "userfile" else
                    source["provenance"].get("text_availability", "computed" if source["kind"] == "computed" else "unknown"))
    limitations = list(source["provenance"].get("limitations", []))
    for item in source.get("artifact_provenance", {}).get("limitations", []):
        if item not in limitations:
            limitations.append(item)
    return {"kind": source["kind"], "text_availability": availability, "limitations": limitations}


def valid_verdict(value, evidence_ids):
    bounded_json(value, 100000)
    if not isinstance(value, dict) or set(value) != {"status", "rationale", "evidence_ids"}:
        raise ValueError("Assessment must have exactly status, rationale, evidence_ids")
    if not isinstance(value["status"], str) or value["status"] not in STATUSES:
        raise ValueError("Unknown assessment status")
    text(value["rationale"], "public assessment rationale")
    ids(value["evidence_ids"])
    if not set(value["evidence_ids"]) <= set(evidence_ids):
        raise ValueError("Assessment cites unbound evidence")
    if value["status"] != "unassessed" and not value["evidence_ids"]:
        raise ValueError("A support verdict requires actual evidence references")
    return deepcopy(value)


class Assessments:
    def _provider(self):
        if self.assessor is None:
            return None
        return self.assessor if isinstance(self.assessor, HostAssessor) else HostAssessor(self.assessor)

    def _claim_snapshot(self, conn, session_id, run_id, claim_id):
        task = self._scope(conn, session_id, run_id)
        claim = self._get(conn, "claim", session_id, task["id"], claim_id)
        evidence = [self._get(conn, "evidence", session_id, task["id"], eid) for eid in claim["evidence_ids"]]
        sources = {e["source_id"]: self._get(conn, "source", session_id, task["id"], e["source_id"]) for e in evidence}
        for source in sources.values():
            self._source_metadata(conn, source)
        return {"session_id": session_id, "run_id": run_id, "task_id": task["id"],
                "claim": claim, "evidence": evidence, "sources": list(sources.values()),
                "bindings": {"claim_id": claim_id, "claim_revision": claim["revision"],
                             "evidence": {e["id"]: e["revision"] for e in evidence},
                             "sources": {s["id"]: s["artifact_hash"] for s in sources.values()}}}

    def _bindings_current(self, conn, session_id, run_id, bindings):
        try:
            current = self._claim_snapshot(conn, session_id, run_id, bindings["claim_id"])
        except (KeyError, ValueError, PermissionError):
            return False
        return current["bindings"] == bindings

    def get_assessment(self, session_id, run_id, assessment_id):
        """Immutable historical verdict; use current_assessment for eligibility."""
        return self._read("assessment", session_id, run_id, assessment_id)

    def list_assessments(self, session_id, run_id):
        return self._read_list("assessment", session_id, run_id)

    def current_assessment(self, session_id, run_id, claim_id):
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            self._get(conn, "claim", session_id, task["id"], claim_id)
            result = self._head(conn, "assessment", session_id, task["id"], claim_id)
            if result is None:
                return None
            current = self._bindings_current(conn, session_id, run_id, result["bindings"])
            snapshot = self._claim_snapshot(conn, session_id, run_id, claim_id) if current else None
        reason = None if current else "stale_inputs"
        if snapshot:
            try:
                for source in snapshot["sources"]:
                    self._verify_source(source)
            except (OSError, ValueError):
                reason = "artifact_integrity"
        with self.store.transaction() as conn:
            self._scope(conn, session_id, run_id)
            head = self._head(conn, "assessment", session_id, task["id"], claim_id)
            if head != result or not self._bindings_current(conn, session_id, run_id, result["bindings"]):
                reason = "stale_inputs"
        if reason:
            result = {**result, "recorded_status": result["status"], "status": "unassessed",
                      "reason_code": reason, "stale": True}
        return self.store.redact(result)

    def assess_claim(self, session_id, run_id, claim_id):
        provider = self._provider()
        if provider:
            self.store.register_secrets(provider.secrets)
        with self.store.transaction() as conn:
            snapshot = self._claim_snapshot(conn, session_id, run_id, claim_id)
        request = self.store.redact(snapshot)
        result = {"status": "unassessed", "rationale": "No host assessment provider is configured.", "evidence_ids": []}
        reason = "assessor_unavailable"
        try:
            for source in snapshot["sources"]:
                self._verify_source(source)
        except (OSError, ValueError) as exc:
            reason = "artifact_integrity"
            result["rationale"] = f"Source integrity check failed: {exc}"
        else:
            if provider:
                try:
                    response = provider.callback(deepcopy(request))
                except Exception as exc:
                    reason = "assessor_error"
                    result["rationale"] = f"Host assessment failed: {type(exc).__name__}: {str(exc)[:10000]}"
                else:
                    try:
                        result = valid_verdict(response, snapshot["claim"]["evidence_ids"])
                        reason = "assessed"
                    except (ValueError, TypeError, RecursionError):
                        reason = "malformed_assessment"
                        # Do not retain arbitrary fields, including accidental
                        # private reasoning, from an invalid response envelope.
                        result = {"status": "unassessed", "rationale": "Host returned a malformed assessment envelope.",
                                  "evidence_ids": []}
        integrity = True
        try:
            for source in snapshot["sources"]:
                self._verify_source(source)
        except (OSError, ValueError):
            integrity = False
        with self.store.transaction(write=True) as conn:
            task = self._scope(conn, session_id, run_id)
            current = self._bindings_current(conn, session_id, run_id, snapshot["bindings"])
            status = result["status"]
            if not current or not integrity:
                status, reason = "unassessed", "stale_inputs" if not current else "artifact_integrity"
            record = self._record("assessment", session_id, run_id, task["id"], claim_id=claim_id,
                                  status=status, rationale=result["rationale"], evidence_ids=result["evidence_ids"],
                                  reason_code=reason, stale=not current, provisional=True,
                                  method=provider.method if provider else "semantic_review_unavailable",
                                  reviewer_id=provider.reviewer_id if provider else "host",
                                  request=request, result=result, bindings=snapshot["bindings"])
            record = self._save(conn, "assessment", record, subject_id=claim_id)
        return self._mirror("assessment", record)

    def observe_evidence(self, session_id, run_id, evidence_id):
        """Supported only as a faithful public quotation/computed-output record.

        This never asserts the source is true or generalizes a numeric output to
        efficacy, mechanism, function or safety. Public excerpts may be redacted.
        """
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            evidence = self._get(conn, "evidence", session_id, task["id"], evidence_id)
            source = self._get(conn, "source", session_id, task["id"], evidence["source_id"])
        content = self._verify_source(source)
        if "passage" in evidence["locator"]:
            # Stored public passage text may be redacted. Immutable character
            # offsets let us verify the actual source slice before redaction.
            locator = evidence["locator"]
            material = content.decode("utf-8")
            if (type(locator["start"]) is not int or type(locator["end"]) is not int
                    or not 0 <= locator["start"] < locator["end"] <= len(material)):
                raise ValueError("Invalid stored literal locator")
            selected = self.store.redact({"text": material[locator["start"]:locator["end"]], "value": None})
        else:
            selected = self.store.redact(extract(content, {"pointer": evidence["locator"]["pointer"]}))
        if selected["text"] != self.store.redact(evidence["text"]) or selected["value"] != self.store.redact(evidence["value"]):
            raise ValueError("Evidence no longer matches exact source material")
        selected_value = selected["text"] if selected["text"] is not None else selected["value"]
        label = {"literature": "Source quotation", "web": "Search-result quotation",
                 "computed": "Computed output", "userfile": "User-supplied content"}[source["kind"]]
        observation = (f"{label} at {bounded_json(evidence['locator'])}: {bounded_json(selected_value)}. "
                       "This records only what the material says/outputs, not independent experimental truth, "
                       "mechanism, efficacy, function or safety. Public excerpts may be redacted. "
                       f"Source/method qualifications: {bounded_json(qualifications(source))}")
        claim_fields(observation, "observation", [evidence_id])
        with self.store.transaction(write=True) as conn:
            self._scope(conn, session_id, run_id)
            current = self._get(conn, "evidence", session_id, task["id"], evidence_id)
            if current != evidence:
                raise ValueError("Stale evidence during observation")
            self._source_metadata(conn, source)
            claim = self._record("claim", session_id, run_id, task["id"], text=observation,
                                 kind="observation", evidence_ids=[evidence_id], origin="host_observation")
            claim = self._save(conn, "claim", claim)
            snapshot = self._claim_snapshot(conn, session_id, run_id, claim["id"])
            result = {"status": "supported", "rationale": "Exact selected material rendered as a narrow attributed observation only.",
                      "evidence_ids": [evidence_id]}
            assessment = self._record("assessment", session_id, run_id, task["id"], claim_id=claim["id"],
                                      **result, reason_code="literal_material", stale=False, provisional=False,
                                      method="literal_material_observation", reviewer_id="host-literal-verifier",
                                      request=snapshot, result=result, bindings=snapshot["bindings"])
            assessment = self._save(conn, "assessment", assessment, subject_id=claim["id"])
        self._mirror("assessment", assessment)
        return self._mirror("claim", claim)
