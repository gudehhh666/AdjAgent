"""Shared registration and native completion import, with no domain-specific loop."""

from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path

from ..contracts import ToolResult
from ..execution import JobManager
from ..execution.manager import TERMINAL
from ..research import ResearchStore
from ..tools import ToolSpec, ToolFailure
from ..tools.catalog import canonical_json
from .files import inspect_input
from .literature import EuropePMC
from .scientific_resources import ScientificResources, import_response


PATH = {"type": "string", "minLength": 1, "maxLength": 4096}
ID = {"type": "string", "minLength": 1, "maxLength": 128, "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]*$"}
HANDLERS = {"protein_sequence": "adj_partner.native.workers:sequence_job",
            "protein_fasta": "adj_partner.native.workers:fasta_job",
            "protein_structure": "adj_partner.native.workers:structure_job",
            "deg_analyze": "adj_partner.native.deg_worker:deg_job",
            "deg_explore": "adj_partner.native.deg_worker:deg_explore_job"}

DEG_COMPOUNDS = sorted(item["name"].removesuffix("VSDMSO.parquet")
    for item in json.loads(Path(__file__).with_name("deg-cache-manifest.json").read_text(encoding="utf-8"))
    if item["name"].endswith("VSDMSO.parquet"))


def schema(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


class NativeCapabilities:
    def __init__(self, store, *, research=None, jobs=None, dataset_dir=None, literature=None,
                 fulltext=None, documents=None, web_search=None, scientific_resources=None, resource_pack=None):
        self.store = store
        for service in (research, jobs):
            if service is not None and (service.store.root != store.root or service.store.db_path != store.db_path):
                raise ValueError("Native services must share the same Store root/database")
        self.research = research if research is not None and research.store is store else ResearchStore(store, assessor=research.assessor if research else None)
        self.jobs = jobs if jobs is not None and jobs.store is store else JobManager(store)
        # Host paths resolve at configuration time, never from model arguments.
        self.dataset_dir = str(Path(dataset_dir).expanduser().resolve()) if dataset_dir is not None else None
        self.literature = literature or EuropePMC()
        self.fulltext = fulltext
        self.documents = documents
        self.web_search = web_search
        self.scientific_resources = scientific_resources or ScientificResources()
        self.resource_pack = resource_pack

    def register(self, catalog):
        definitions = [
            ToolSpec("protein_sequence", "Compute canonical sequence properties in a durable job; no structure/function inference.",
                     schema({"sequence": {"type": "string", "minLength": 1, "maxLength": 200000}}, ["sequence"])),
            ToolSpec("protein_fasta", "Compute real properties for up to16 FASTA records in a durable job; permitted files only.", schema({"path": PATH}, ["path"])),
            ToolSpec("protein_structure", "Parse actual PDB/mmCIF counts in an isolated job; not prediction or experimental validation.",
                     schema({"path": PATH, "format": {"enum": ["pdb", "mmcif"]}}, ["path"])),
            ToolSpec("deg_analyze", "Analyze the host-configured DEG cache in a durable job; exploratory ranking with all design limitations.",
                     schema({"top_n": {"type": "integer", "minimum": 10, "maximum": 1000},
                             "n_permutations": {"type": "integer", "minimum": 100, "maximum": 5000}})),
            ToolSpec("deg_explore", "Inspect one exact compound from deg_analyze using deterministic top-DEG and GO/KEGG/Reactome enrichment calculations. Use for candidate-specific exploration; it does not establish mechanism, efficacy or safety.",
                     schema({"compound": {"type": "string", "enum": DEG_COMPOUNDS},
                             "direction": {"enum": ["up", "down", "both"]},
                             "annotation": {"enum": ["GO", "KEGG", "Reactome"]},
                             "top_genes": {"type": "integer", "minimum": 1, "maximum": 20},
                             "top_terms": {"type": "integer", "minimum": 1, "maximum": 15}}, ["compound"])),
        ]
        bindings = []
        for spec in definitions:
            job_schema = spec.input_schema
            if spec.name in {"deg_analyze", "deg_explore"}:
                job_schema = deepcopy(job_schema)
                job_schema["properties"]["dataset_dir"] = PATH
                job_schema["required"].append("dataset_dir")
            self.jobs.register(ToolSpec(spec.name, spec.description, job_schema, version=spec.version), HANDLERS[spec.name])
            bindings.append((spec, self._submitter(spec.name)))
        bindings.extend([
            (ToolSpec("file_inspect", "Inspect permitted UTF-8/JSON/CSV/TSV input as user-supplied material, not verified science.", schema({"path": PATH}, ["path"])), self._inspect),
            (ToolSpec("artifact_read", "Read a bounded, redacted text slice of an intact current-session artifact. Continue only with the returned next_start and stop when it is null; never guess character offsets.", schema({
                "artifact_id": ID, "start": {"type": "integer", "minimum": 0, "maximum": 100000000},
                "max_characters": {"type": "integer", "minimum": 1, "maximum": 16000}}, ["artifact_id"])), self._read),
            (ToolSpec("literature_search", "Find scholarly candidates by exact paper ID or a few distinctive terms. Use the matching record's source_id for reading. Titles and abstracts have separate scopes.", schema({
                "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                "page_size": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]), network=True), self._search),
            (ToolSpec("article_resolve", "Resolve one current-run literature discovery source to DOI/PMID/PMCID and lawful acquisition options. Metadata is not full text.",
                schema({"source_id": ID}, ["source_id"]), network=True), self._resolve_article),
            (ToolSpec("fulltext_fetch", "Fetch one resolved article through its eligible public sources. Returns the selected provider and attempt history.",
                schema({"source_id": ID}, ["source_id"]), network=True), self._fetch_fulltext),
            (ToolSpec("document_read", "Read ingested JATS, BioC, PDF or abstracts. For optional PDF table cells or OCR text, supply mode and a page number. Use the returned source_id and exact matches[].selector with evidence_read or evidence_select. Page filters apply to PDFs. Exact article headings select that subtree. Start at cursor 0 for each new query or filter; next_cursor continues only the same search.",
                schema({"source_id": ID, "query": {"type": "string", "maxLength": 1000},
                        "page": {"type": "integer", "minimum": 1, "maximum": 500},
                        "mode": {"type": "string", "enum": ["text", "tables", "ocr"]},
                        "section": {"type": "string", "minLength": 1, "maxLength": 200},
                        "cursor": {"type": "integer", "minimum": 0, "maximum": 1000000},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, ["source_id"])), self._read_document),
            (ToolSpec("job_status", "Reconcile this run's native job and import computed evidence only after actual successful completion.", schema({"job_id": ID}, ["job_id"])), self._job_status),
        ])
        public_query = {"type": "string", "minLength": 1, "maxLength": 500}
        small_limit = {"type": "integer", "minimum": 1, "maximum": 10}
        for name, description, properties, required in [
            ("chembl_compound", "Look up an exact ChEMBL molecule ID or search public compound names. Database records only; no efficacy or safety validation.",
             {"query": public_query, "max_results": small_limit}, ["query"]),
            ("chembl_activity", "Retrieve a small first page of assay-context activity records for an exact ChEMBL molecule ID. Assay-specific database values, not validated findings.",
             {"molecule_chembl_id": {"type": "string", "pattern": "^CHEMBL[0-9]{1,12}$"}, "max_results": small_limit}, ["molecule_chembl_id"]),
            ("reactome_pathway", "Retrieve one exact Reactome stable pathway ID. Database pathway content does not establish compound efficacy or modulation.",
             {"stable_id": {"type": "string", "pattern": "^R-[A-Z]{3}-[0-9]{1,12}$"}}, ["stable_id"]),
            ("geo_search", "Discover public GEO candidate dataset metadata through NCBI search and summaries. No expression analysis or independent replication is performed.",
             {"query": public_query, "max_results": small_limit}, ["query"]),
        ]:
            bindings.append((ToolSpec(name, description, schema(properties, required), network=True), self._resource(name)))
        catalog.register_many(bindings, secrets=getattr(self.literature, "secrets", ()))
        from .sequence_inventory import register_sequence_inventory
        register_sequence_inventory(catalog)
        from .workbooks import register_workbooks
        register_workbooks(catalog)
        from .pmc_cloud import register_pmc_attachments
        register_pmc_attachments(self, catalog)
        from ..resources.tools import register_resource_tools
        register_resource_tools(self, catalog)
        if self.web_search is not None:
            catalog.register_many([(ToolSpec("web_search", "Search the public web through the host-configured provider. Results are titles and search-provider snippets only; no result page or paper full text was read.", schema({
                "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]), network=True), self._web)],
                secrets=self.web_search.secrets)

    def _bind(self, context):
        context.check_cancelled()
        if context.store.root != self.store.root or context.store.db_path != self.store.db_path:
            raise ToolFailure("native_store_mismatch", "Native tools are bound to another Store", "denied")
        if context.store is self.store:
            return self
        return type(self)(context.store, research=ResearchStore(context.store, assessor=self.research.assessor),
                          dataset_dir=self.dataset_dir, literature=self.literature, fulltext=self.fulltext,
                          documents=self.documents, web_search=self.web_search, scientific_resources=self.scientific_resources,
                          resource_pack=self.resource_pack)

    def _resource(self, name):
        def retrieve(context, **arguments):
            bound = self._bind(context)
            response = getattr(bound.scientific_resources, name)(**arguments, check_cancelled=context.check_cancelled)
            return import_response(context, bound.research, response)
        return retrieve

    def _submitter(self, name):
        def submit(context, **arguments):
            bound = self._bind(context)
            if not context.call_id:
                raise ToolFailure("call_identity_required", "Durable native work requires its original tool call ID")
            idempotency_key = context.call_id
            if name in {"deg_analyze", "deg_explore"}:
                if bound.dataset_dir is None:
                    raise ToolFailure("dataset_unconfigured", "No host DEG dataset is configured")
                context.check_path(bound.dataset_dir)
                idempotency_key = self._deg_idempotency_key(name, arguments)
                arguments = {**arguments, "dataset_dir": bound.dataset_dir}
            job = bound.jobs.submit(context.session_id, context.run_id, name, arguments, idempotency_key,
                                    timeout_seconds=300 if name in {"deg_analyze", "deg_explore"} else 30)
            return ToolResult(status="pending", data={"job_id": job["id"], "status": job["status"]}, job_id=job["id"])
        return submit

    @staticmethod
    def _deg_idempotency_key(name, arguments):
        digest = hashlib.sha256(canonical_json(arguments).encode("utf-8")).hexdigest()
        return f"{name}:{digest}"

    def _scope(self, session_id, run_id):
        if self.store.get_run(run_id)["session_id"] != session_id:
            raise PermissionError("Run does not belong to session")
        self.store.get_session(session_id)

    def resolve_job(self, session_id, run_id, job_id):
        self._scope(session_id, run_id)
        job = self.jobs.get(job_id, session_id)
        if job["run_id"] != run_id or job["handler_ref"] != HANDLERS.get(job["tool_name"]):
            raise PermissionError("Job is not this run's host-native capability")
        job = self.jobs.reconcile(job_id, session_id)
        if job["status"] not in TERMINAL:
            return ToolResult(status="pending", data={"job_id": job_id, "status": job["status"]}, job_id=job_id)
        result = self.jobs.result(job_id, session_id)
        if job["status"] != "succeeded" or result.status != "ok":
            return result
        # Host-only identity uses the exact verified immutable envelope, never
        # a current-secret projection that can change across reopen/imports.
        raw_result = json.loads(self.store.read_artifact(job["result_artifact"]["id"]))
        descriptor = raw_result["data"].get("source") if isinstance(raw_result["data"], dict) else None
        if (not isinstance(descriptor, dict) or descriptor.get("kind") != "computed"
                or descriptor.get("artifact_id") not in {a["id"] for a in result.artifacts}
                or descriptor.get("provenance", {}).get("job_id") != job_id):
            raise ValueError("Native success lacks a scoped actual computed-source descriptor")
        source = self.research.register_source(session_id, run_id, **descriptor, idempotency_key="native-job:" + job_id)
        return ToolResult(data={**result.data, "source_id": source["id"]}, artifacts=result.artifacts, job_id=job_id)

    def recover_action(self, session_id, run_id, call):
        """Never resubmit: locate exact run/call/tool intent after an outer crash."""
        self._scope(session_id, run_id)
        # Public JobManager views intentionally redact current secrets. Compare
        # immutable raw intent host-side through Store's read-only transaction;
        # no raw argument data is returned, journaled anew or sent to the model.
        keys = [call.id]
        if call.name in {"deg_analyze", "deg_explore"}:
            keys.insert(0, self._deg_idempotency_key(call.name, call.arguments))
        with self.store.transaction() as conn:
            job = None
            for key in keys:
                job = conn.execute("SELECT id,tool_name,arguments FROM execution_jobs WHERE session_id=? AND run_id=? AND idempotency_key=?",
                                   (session_id, run_id, key)).fetchone()
                if job is not None: break
        if job is None:
            return None
        public_args = json.loads(job["arguments"])
        if job["tool_name"] in {"deg_analyze", "deg_explore"}:
            public_args.pop("dataset_dir", None)
        if job["tool_name"] != call.name or canonical_json(public_args) != canonical_json(call.arguments):
            raise ValueError("Existing job identity conflicts with proposed action")
        return self.resolve_job(session_id, run_id, job["id"])

    def _job_status(self, context, job_id):
        return self._bind(context).resolve_job(context.session_id, context.run_id, job_id)

    def _inspect(self, context, path):
        bound = self._bind(context)
        item, inspection, media_type = inspect_input(context, path)
        provenance = {"inputs": [item.manifest()], "text_availability": "userfile",
                      "limitations": ["User-supplied file contents, not independent experimental truth or verified scientific conclusions."]}
        artifact = context.store.put_artifact(item.content, Path(item.path).name, media_type, provenance,
                                             session_id=context.session_id, run_id=context.run_id)
        source = bound.research.register_source(context.session_id, context.run_id, artifact_id=artifact["id"],
                    kind="userfile", title=Path(item.path).name, provenance=provenance, idempotency_key="native-file:" + context.call_id)
        return ToolResult(data={"source_id": source["id"], "inspection": inspection, "input": item.manifest()}, artifacts=[artifact])

    def _read(self, context, artifact_id, start=0, max_characters=8000):
        self._bind(context)
        metadata = next((item for item in context.store.artifacts(context.session_id) if item["id"] == artifact_id), None)
        if metadata is None:
            raise PermissionError("Unknown artifact in this session")
        if metadata["size"] > 8 * 1024 * 1024:
            raise ValueError("Text artifact exceeds the 8 MiB read limit")
        content = context.store.redact(context.store.read_artifact(artifact_id).decode("utf-8-sig"))
        if start > len(content):
            raise ValueError("Artifact character offset is beyond content")
        requested_end = min(len(content), start + max_characters)
        # The provider protocol limits tool messages by encoded bytes, while
        # this public API pages by character offsets. Keep the entire result
        # below ContextView's 2.4 KiB preview boundary so artifact_read exposes
        # the requested text itself instead of recursively producing another
        # context artifact (especially for multibyte scientific text).
        low, high = start, requested_end
        while low < high:
            middle = (low + high + 1) // 2
            if len(canonical_json(content[start:middle]).encode("utf-8")) <= 1400:
                low = middle
            else:
                high = middle - 1
        end = low
        return ToolResult(data={"artifact_id": artifact_id, "artifact_hash": metadata["sha256"],
                               "content": content[start:end], "start": start, "next_start": end if end < len(content) else None,
                               "total_characters": len(content), "projection": "redacted_text_character_slice"})

    def _search(self, context, query, page_size=5):
        bound = self._bind(context)
        task = context.store.get_task(context.store.get_run(context.run_id)["task_id"])
        if task["acceptance"].get("require_concise_discovery"):
            with context.store.transaction() as conn:
                completed = conn.execute("SELECT COUNT(*) FROM actions WHERE run_id=? AND tool_name='literature_search' AND status='completed'", (context.run_id,)).fetchone()[0]
            terms = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
            if completed < 3 and (len(terms) > 4 or len(query) > 96):
                raise ToolFailure("discovery_query_too_broad",
                    "Initial discovery requires three successful short searches: at most four words and 96 characters each. "
                    "Use one distinctive physical/storage constraint plus a broad research term; omit guessed platform and secondary endpoints. "
                    "Vary the constraint or synonym across queries. After discovery, detailed searches are allowed.")
        response = bound.literature.search(query, page_size=page_size, check_cancelled=context.check_cancelled)
        normalized = hasattr(response, "diagnostics") and hasattr(response, "request_urls")
        provider_name = "scholarly_multi" if normalized else "Europe PMC"
        provenance = {"provider": provider_name, "query": query, "request_url": response.request_url,
                      "retrieved_at": response.received_at,
                      "limitations": ["Core metadata and available abstracts only; full papers were not retrieved or reviewed."]}
        if normalized:
            provenance.update(request_urls=list(response.request_urls), diagnostics=list(response.diagnostics))
        raw = context.store.put_artifact(response.raw, "scholarly-response.json" if normalized else "europe-pmc-response.json", "application/json", provenance,
                                       session_id=context.session_id, run_id=context.run_id)
        results, artifacts = [], [raw]
        for original_record in response.records:
            # Raw provider bytes remain in the private response artifact. Every
            # public derived field is redacted before slicing or title limits.
            record = context.store.redact(original_record)
            identifiers = {key: record[key] for key in ("doi", "pmid", "pmcid") if isinstance(record.get(key), str)}
            providers = record.get("providers") if isinstance(record.get("providers"), list) else [record.get("provider", provider_name)]
            item_provenance = {**provenance, **identifiers, "providers": providers,
                               "response_artifact_id": raw["id"], "text_availability": record["text_availability"],
                               "public_text_redacted": record != original_record,
                               "provider_id": record["id"], "provider_source": record["source"], "url": record["url"]}
            artifact = context.store.put_artifact(canonical_json(record), "literature-record.json", "application/json",
                item_provenance, session_id=context.session_id, run_id=context.run_id)
            source = bound.research.register_source(context.session_id, context.run_id, artifact_id=artifact["id"], kind="literature",
                title=record["title"][:2000], provenance=item_provenance,
                idempotency_key=f"native-lit:{context.call_id}:{record['source']}:{record['id']}")
            results.append({"source_id": source["id"], "artifact_id": artifact["id"], "provider_id": record["id"],
                            "title": record["title"][:2000], "text_availability": record["text_availability"],
                            "abstract_preview": record["abstract"][:4000], "abstract_truncated": len(record["abstract"]) > 4000,
                            **identifiers, "providers": providers})
            artifacts.append(artifact)
        summary_results = []
        summary = {"provider": provider_name, "query": query[:96], "hit_count": response.hit_count,
                   "returned": len(results), "results_shown": 0, "results_omitted_from_summary": len(results),
                   "results": summary_results,
                   "limitations": ["Metadata and available abstracts only; no full papers were read."]}
        if len(query) > 96:
            summary["query_truncated"] = True
        for index, item in enumerate(results[:5]):
            compact = {"source_id": item["source_id"], "title": item["title"][:180],
                       "text_availability": item["text_availability"]}
            for key in ("doi", "pmcid", "pmid"):
                if item.get(key):
                    compact[key] = item[key]
                    break
            if index < 2:
                length = 600 if index == 0 else 160
                excerpt = item["abstract_preview"].encode("utf-8")[:length].decode("utf-8", errors="ignore")
                compact.update(abstract_preview=excerpt,
                    abstract_truncated=item["abstract_truncated"] or excerpt != item["abstract_preview"])
            summary_results.append(compact)
            # Preserve readable evidence from the leading candidate before spending
            # the model's preview budget on lower-ranked metadata. Full rows stay below.
            if index and len(canonical_json(summary).encode()) > 1760:
                summary_results.pop()
                break
            summary["results_shown"] = len(summary_results)
            summary["results_omitted_from_summary"] = len(results) - len(summary_results)
        return ToolResult(data={"provider": provider_name, "hit_count": response.hit_count, "returned": len(results), "results": results,
                               "summary": summary, "request_url": response.request_url,
                               "request_urls": list(response.request_urls) if normalized else [response.request_url],
                               "diagnostics": list(response.diagnostics) if normalized else [],
                               "retrieved_at": response.received_at}, artifacts=artifacts)

    def _web(self, context, query, max_results=5):
        bound = self._bind(context)
        if bound.web_search is None:
            raise ToolFailure("web_search_unconfigured", "No host web search is configured")
        response = bound.web_search.search(query, max_results=max_results, check_cancelled=context.check_cancelled)
        provider = getattr(response, "provider", "") or getattr(bound.web_search, "provider", "serper")
        fallback_from = list(getattr(response, "fallback_from", ()))
        display_provider = provider.replace("_", " ").title()
        provenance = {"provider": display_provider, "query": query, "request_url": response.request_url,
                      "retrieved_at": response.received_at, "text_availability": "search_snippet",
                      "limitations": ["Search-engine titles/snippets only; result pages and full papers were not fetched or read.",
                                      "Ranking and snippets may be incomplete, stale or generated by the search provider."]}
        if fallback_from:
            provenance["fallback_from"] = fallback_from
        raw = context.store.put_artifact(response.raw, f"{provider}-response.json", "application/json", provenance,
                                         session_id=context.session_id, run_id=context.run_id)
        artifacts, results = [raw], []
        for index, original_record in enumerate(response.records):
            record = context.store.redact(original_record)
            record_provenance = {**provenance, "response_artifact_id": raw["id"], "result_index": index,
                                 "public_text_redacted": record != original_record, "url": record["url"]}
            artifact = context.store.put_artifact(canonical_json(record), f"{provider}-result-{index + 1}.json",
                "application/json", record_provenance, session_id=context.session_id, run_id=context.run_id)
            source = bound.research.register_source(context.session_id, context.run_id, artifact_id=artifact["id"],
                kind="web", title=record["title"][:2000], provenance=record_provenance,
                idempotency_key=f"{provider}:{context.call_id}:{index}")
            artifacts.append(artifact)
            results.append({**record, "source_id": source["id"], "artifact_id": artifact["id"]})
        data = {"query": query, "returned": len(results), "results": results,
                "provider": provider, "text_scope": "search-provider title/snippet only"}
        if fallback_from:
            data["fallback_from"] = fallback_from
        return ToolResult(data=data, artifacts=artifacts)

    def _resolve_article(self, context, source_id):
        bound = self._bind(context)
        source = bound.research.get_source(context.session_id, context.run_id, source_id)
        if source["kind"] != "literature" or source["run_id"] != context.run_id:
            raise PermissionError("article_resolve accepts only a current-run literature source")
        try:
            record = json.loads(bound.research._verify_source(source))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ToolFailure("literature_record", "Discovery source is not a normalized literature record") from exc
        if not isinstance(record, dict) or not isinstance(record.get("title"), str):
            raise ToolFailure("literature_record", "Discovery source is not a normalized literature record")
        candidates = bound.literature.resolve(record, check_cancelled=context.check_cancelled)
        resolution = {"parent_source_id": source_id, "candidates": candidates}
        provenance = {"provider": "scholarly_resolution", "parent_source_id": source_id,
            "text_availability": "title_only",
            "limitations": ["Identity and open-access location metadata only; no full text was fetched or read."]}
        artifact = context.store.put_artifact(canonical_json(resolution), "article-resolution.json",
            "application/json", provenance, session_id=context.session_id, run_id=context.run_id)
        registered = bound.research.register_source(context.session_id, context.run_id,
            artifact_id=artifact["id"], kind="literature", title=record["title"][:2000],
            provenance=provenance, idempotency_key="article-resolve:" + context.call_id)
        locations = candidates[0].get("acquisition", []) if candidates else []
        summary = {"source_id": registered["id"], "acquisition_count": len(locations),
                   "next_tool": "fulltext_fetch" if locations else None}
        return ToolResult(data={"source_id": registered["id"], "artifact_id": artifact["id"],
            "candidates": candidates, "resolution": resolution, "summary": summary}, artifacts=[artifact])

    def _fetch_fulltext(self, context, source_id):
        bound = self._bind(context)
        if bound.fulltext is None:
            raise ToolFailure("fulltext_unconfigured", "No host full-text fetcher is configured")
        source = bound.research.get_source(context.session_id, context.run_id, source_id)
        if (source["kind"] != "literature" or source["run_id"] != context.run_id
                or source["provenance"].get("provider") != "scholarly_resolution"):
            raise PermissionError("fulltext_fetch accepts only a current-run article resolution source")
        try:
            resolution = json.loads(bound.research._verify_source(source))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ToolFailure("fulltext_resolution", "Resolution source is malformed") from exc
        response = bound.fulltext.fetch(resolution, check_cancelled=context.check_cancelled)
        candidate = resolution["candidates"][0]
        identity = {key: candidate[key] for key in ("doi", "pmid", "pmcid")
                    if isinstance(candidate.get(key), str)}
        attempts = list(getattr(response, "attempts", ()))
        provenance = {"provider": response.acquisition.get("provider"), "url": response.url,
            "parent_source_id": source_id, "retrieved_at": response.received_at,
            "text_availability": "full_text", "document_scope": "article",
            "acquisition": response.acquisition, "attempts": attempts, **identity,
            "limitations": ["Acquisition covers article bytes. Scientific claims require passage-level assessment."]}
        if type(response.acquisition.get("pmc_version")) is int:
            provenance["pmc_version"] = response.acquisition["pmc_version"]
        if response.acquisition.get("license"):
            provenance["license"] = response.acquisition["license"]
        name = {"application/xml":"article.xml", "application/pdf":"article.pdf",
                "application/bioc+json":"article.bioc.json"}[response.media_type]
        artifact = context.store.put_artifact(response.body, name, response.media_type, provenance,
            session_id=context.session_id, run_id=context.run_id)
        registered = bound.research.register_source(context.session_id, context.run_id,
            artifact_id=artifact["id"], kind="literature", title=source["title"], provenance=provenance,
            idempotency_key="fulltext-fetch:" + context.call_id)
        return ToolResult(data={"source_id": registered["id"], "artifact_id": artifact["id"],
            "provider": provenance["provider"], "attempts": attempts,
            "summary": {"source_id": registered["id"], "media_type": response.media_type,
                        "next_tool": "document_read", "scope": "Article acquired; no passage selected yet."},
            "media_type": response.media_type, "url": response.url,
            "retrieved_at": response.received_at}, artifacts=[artifact])

    def _read_document(self, context, source_id, query="", page=None, section=None, cursor=0, limit=8, mode="text"):
        bound = self._bind(context)
        if bound.documents is None:
            raise ToolFailure("documents_unconfigured", "No host document parser is configured")
        source = bound.research.get_source(context.session_id, context.run_id, source_id)
        if source["run_id"] != context.run_id:
            raise PermissionError("document_read requires a current-run source; prior same-task material needs host handoff")
        if source["kind"] != "literature" or source["provenance"].get("text_availability") not in {"full_text", "derived_full_text", "abstract"}:
            raise PermissionError("document_read requires readable literature: raw full text, host-parsed full text, or an abstract; titles/snippets are insufficient")
        if mode in {"tables", "ocr"}:
            if type(page) is not int or page < 1:
                raise ToolFailure("document_page_required", "Select one PDF page for table extraction or OCR")
            if source["provenance"].get("provider") == "document_parser":
                parent = source["provenance"].get("parent_source_id")
                source = bound.research.get_source(context.session_id, context.run_id, parent)
            if (source["run_id"] != context.run_id or source["kind"] != "literature"
                    or source["media_type"] != "application/pdf"
                    or source["provenance"].get("text_availability") != "full_text"):
                raise ToolFailure("document_media", "This mode requires a current-run full-text PDF")
            source_id = source["id"]
        key = f"document-parse-v2:{source_id}:{mode}"
        if mode != "text":
            key += f":{page}"
        cached = bound.research.current_source_by_key(context.session_id, context.run_id, key)
        if cached is not None:
            bound.research._verify_source(source)
            source = cached
        content = bound.research._verify_source(source)
        if mode != "text" and cached is None:
            from .pdf_enhanced import parse_pdf_page
            document = parse_pdf_page(content, page, mode, check_cancelled=context.check_cancelled)
            reused, scope = False, "derived_full_text"
        else:
            from .documents import prepare_readable_document
            document, reused, scope = prepare_readable_document(
                content, source["media_type"], source["provenance"], bound.documents)
        if reused:
            registered = source
            provenance = source["provenance"]
            artifact = {"id": source["artifact_id"], "sha256": source["artifact_hash"], "media_type": source["media_type"]}
        else:
            inherited = {key: source["provenance"][key] for key in
                         ("doi", "pmid", "pmcid", "url", "license", "host_type", "document_scope",
                          "attachment", "archive_member", "pmc_version", "parent_article_source_id")
                         if isinstance(source["provenance"].get(key), str)}
            provenance = {"provider": "document_parser", "parent_source_id": source_id,
                "text_availability": scope, "document_type": document.get("document_type"),
                "limitations": list(document.get("limitations", [])), **inherited}
            if type(source["provenance"].get("pmc_version")) is int:
                provenance["pmc_version"] = source["provenance"]["pmc_version"]
            if mode != "text":
                provenance.update(parse_mode=mode, page=page, backend=document.get("backend"))
            artifact = context.store.put_artifact(canonical_json(document), "parsed-document.json",
                "application/json", provenance, session_id=context.session_id, run_id=context.run_id)
            registered = bound.research.register_source(context.session_id, context.run_id,
                artifact_id=artifact["id"], kind="literature", title=source["title"], provenance=provenance,
                idempotency_key=key)
        projection = bound.documents.query(document, query=query, page=page, section=section,
                                           cursor=cursor, limit=limit)
        summary_matches = []
        for match in projection.get("matches", [])[:5]:
            text_value = match.get("text", "") if isinstance(match, dict) else ""
            summary_matches.append({key: match[key] for key in ("locator", "selector", "section", "text_availability")
                                    if key in match} | {"text": text_value[:500],
                                    "text_truncated": len(text_value) > 500})
        summary = {"source_id": registered["id"], "document_scope": provenance.get("document_scope", scope),
                   "mode": document.get("mode", "text"), "returned": projection.get("returned"), "total_matches": projection.get("total_matches"),
                   "next_cursor": projection.get("next_cursor"), "matches": summary_matches,
                   "matches_omitted": max(0, len(projection.get("matches", [])) - len(summary_matches))}
        if projection.get("ignored_filters"):
            summary["ignored_filters"] = projection["ignored_filters"]
        return ToolResult(data={"source_id": registered["id"], "artifact_id": registered["artifact_id"],
            "document_type": document.get("document_type"),
            "document_scope": provenance.get("document_scope", scope),
            "mode": document.get("mode", "text"), **projection,
            "summary": summary, "text_availability": scope, "limitations": provenance.get("limitations", [])}, artifacts=[] if reused else [artifact])
