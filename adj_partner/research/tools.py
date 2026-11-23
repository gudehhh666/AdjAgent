"""Research operations in the one host catalog; no source or verdict forgery API."""

from ..contracts import ToolResult
from ..tools import ToolSpec
from ..tools.executor import ToolFailure
from .views import summary
from .design import tool_schema


ID = {"type": "string", "minLength": 1, "maxLength": 128, "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]*$"}
TEXT = {"type": "string", "minLength": 1, "maxLength": 20000}
IDS = {"type": "array", "items": ID, "maxItems": 100, "uniqueItems": True}
REVISION = {"type": "integer", "minimum": 0, "maximum": 2147483647}


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


SELECTOR = {"oneOf": [object_schema({"passage": TEXT}),
                       object_schema({"pointer": {"type": "string", "maxLength": 2000}})]}
CLAIM_FIELDS = {"text": TEXT, "kind": {"enum": ["observation", "proposal", "rejected"]}, "evidence_ids": IDS}


def register_research_tools(catalog, research):
    """Register once; a reopened same-root Store uses the actual ToolContext.

    Credentials are host-only registration metadata, primed by ToolExecutor
    before journaling and by assessment operations before public provider I/O.
    A catalog bound to a different database is rejected, never silently written.
    """
    definitions = [
        ("research_state", "Page current research record summaries; follow next_cursor. Full history stays in the host trace.", "state_page",
         {"cursor": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}),
        ("evidence_select", "Select a literal passage or JSON pointer from a host-ingested source. After document_read, pass its returned derived source_id and an exact returned matches[].selector.", "select_evidence",
         {"source_id": ID, "selector": SELECTOR}),
        ("evidence_reselect", "Correct a selection with a checked revision; prior evidence remains in history.", "reselect_evidence",
         {"evidence_id": ID, "expected_revision": REVISION, "selector": SELECTOR}),
        ("claim_record", "Record an observation, proposal, or rejected claim. Call after evidence_select has returned evidence_ IDs in a previous completed step; evidence_ids never accepts source_ IDs. This does not establish support.", "record_claim", CLAIM_FIELDS),
        ("claim_revise", "Revise claim text/references with a checked revision; old assessments become stale.", "revise_claim",
         {"claim_id": ID, "expected_revision": REVISION, **CLAIM_FIELDS}),
        ("claim_observe", "Host-render only what a selected source says or computed; not mechanism, efficacy or safety.", "observe_evidence", {"evidence_id": ID}),
        ("claim_assess", "Ask the injected host reviewer once; missing/failing review remains unassessed, semantic review provisional.", "assess_claim", {"claim_id": ID}),
        ("design_record_read", "Read a same-task immutable design or experimental-result record; full JSON is in the returned artifact.", "get_design_record", {"record_id": ID}),
        ("experiment_state", "Read evidence-bound reported experiment outcomes for a design. Quality/context are reporter judgments, not host scientific certification; do not replace mechanism with function.", "experiment_state", {"design_id": ID}),
        ("experiment_design_revise", "Create a new design version using host-ingested experimental results. Each affected checkpoint must change its evidence narrative and cite the result evidence. Cannot create or certify experiment results.", "revise_design_from_experiments",
         {"design_id": ID,"result_ids":IDS,"design":tool_schema(ID)}),
        ("report_create", "Render an immutable Markdown report. Each section needs claim_ids or paragraphs; omitted section claim_ids are derived from paragraph claim_ids. Host-owned citations and source caveats remain authoritative.", "create_report",
         {"title": {"type": "string", "minLength": 1, "maxLength": 2000},
          "sections": {"type": "array", "maxItems": 50, "items": {
              "type": "object", "properties": {
                  "heading": {"type": "string", "minLength": 1, "maxLength": 200}, "claim_ids": IDS,
                  "paragraphs": {"type": "array", "maxItems": 100,
                                 "items": object_schema({"text": TEXT, "claim_ids": IDS})}},
              "required": ["heading"], "additionalProperties": False}},
          "limitations": {"type": "array", "maxItems": 100, "items": TEXT}}),
        ("report_validate", "Record structural acceptance against current versions/requirements; not scientific assurance.", "validate_report", {"report_id": ID}),
    ]

    def handler_for(method):
        def handler(context, **arguments):
            context.check_cancelled()
            if (context.store.root.resolve() != research.store.root.resolve()
                    or context.store.db_path.resolve() != research.store.db_path.resolve()):
                raise ToolFailure("research_store_mismatch", "Research tools are bound to a different database/root", "denied")
            # Do not use the older Store object's secret set when the executing
            # Store was reopened or received fresh host redaction material.
            bound = research if context.store is research.store else type(research)(context.store, assessor=research.assessor)
            try:
                data = getattr(bound, method)(context.session_id, context.run_id, **arguments)
            except (KeyError, ValueError, TypeError) as exc:
                raise ToolFailure("research_invalid", str(exc)) from exc
            artifacts = []
            if method in ('experiment_state','get_design_record','revise_design_from_experiments'):
                from ..tools.catalog import canonical_json
                artifact=context.store.put_artifact(canonical_json(data),'experiment-record.json','application/json',
                    {'kind':method,'scientific_validation':'not_independently_validated'},
                    session_id=context.session_id,run_id=context.run_id)
                artifacts=[artifact]
                if method=='experiment_state':
                    compact={k:data[k] for k in ('design_id','decision','scientific_validation')}
                    compact['checkpoints']=[{**{k:v[k] for k in ('checkpoint_id','branch','decision_level','status')},
                        'result_count':len(v['result_ids']),'stale_count':len(v['stale_results'])} for v in data['checkpoints']]
                    if 'mechanism_graph' in data:
                        g=data['mechanism_graph']
                        compact['mechanism_graph']={'hypotheses':[{k:h[k] for k in ('hypothesis_id','status')} for h in g['hypotheses']],
                            'next_mechanism_checkpoints':[r['checkpoint_id'] for r in g['mechanism_order']],
                            'deprioritized_checkpoints':[r['checkpoint_id'] for r in g['deprioritized_checkpoints']],
                            'hypotheses_without_mechanism_checkpoint':g['hypotheses_without_mechanism_checkpoint'],
                            'claims_to_revisit':g['claims_to_revisit']}
                else:
                    compact={'record_id':data['id'],'kind':data['kind'],'hash':data['hash'],
                             'parent_id':data['payload'].get('parent_id'),
                             'scientific_validation':'Record/source binding only; read full artifact for claims and result details.'}
                data={'summary':compact,'artifact_id':artifact['id']}
            if method == "create_report":
                artifacts = [{"id": data["artifact_id"], "sha256": data["artifact_hash"], "media_type": "text/markdown"}]
                data = summary("report", data)
            elif method == "validate_report":
                data = {**summary("validation", data), "issues": data["issues"]}
            return ToolResult(data=data, artifacts=artifacts)
        return handler

    # Keep optional fields optional for legacy report callers.
    def schema_for(method, fields):
        schema = object_schema(fields)
        if method == "state_page":
            schema["required"] = []
        if method == "create_report":
            schema["properties"] = {**schema["properties"], "research_design": tool_schema(ID)}
        return schema

    provider = research._provider()
    if provider is None:
        definitions = [(name, description + (" No semantic assessor is configured: this call will remain unassessed. Use claim_observe only for exact attributed quotations; authored synthesis remains a proposal." if method == "assess_claim" else ""), method, schema)
                       for name, description, method, schema in definitions]
    catalog.register_many([(ToolSpec(name, description, schema_for(method, schema)), handler_for(method))
                           for name, description, method, schema in definitions],
                          secrets=provider.secrets if provider else ())
