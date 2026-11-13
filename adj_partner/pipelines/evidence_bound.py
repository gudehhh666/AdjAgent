"""F2 evidence-bound agent, v3: one AgentService session over typed resource packs, the local literature library
and sandboxed computation, steered only by session instructions chosen from the task kind.

Kinds: ``qa`` (literature and resource questions; adds ``library_search``), ``field`` and ``entry`` (record
reading), ``data`` (computation over supplied files). Build the service with ``service_options()`` plus the
task's ``resource_pack`` and ``dataset_dir``.
"""
import json

from . import library_search
from .sessions import drive, open_session

NAME = 'evidence-bound-v3'
PROFILE = 'research-evidence-bound'
BUDGET = dict(max_steps=10, max_input_tokens=100000, max_output_tokens=8192, wall_time_seconds=240)
REQUEST_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS = 4096, 120
TOOLS = ('catalog_search', 'tools_enable', 'ask_user', 'update_plan', 'file_inspect', 'artifact_read',
         'code_interpreter', 'job_status', 'research_resources', 'research_skill', 'adjuvant_read_fields',
         'adjuvant_cite', 'adjuvant_search', 'adjuvant_record', 'adjuvant_resource_guide', 'adjuvant_lookup',
         'adjuvant_excerpt', 'adjuvant_skill')
KINDS = ('qa', 'field', 'entry', 'data')


ROUTING_V1 = (
    'Evidence routing for this host (all offline): '
    '(1) Paper-specific findings, mechanisms, formulations, measured results and design precedents live in the local '
    'literature library: call library_search with 3-8 distinctive terms from the question (adjuvant, antigen, '
    'pathogen, model species, readout). If a hit is the matching study, call library_search again with that paper_id '
    'and a focused query for the exact result asked. Use what the passages say (direction, numbers, comparator, model) '
    'in the answer and list the paper_id values in record_ids. Two searches without a matching study: answer from '
    'established knowledge and name the gap briefly; never refuse a substantive answer. '
    '(2) Structured snapshot records (ontology terms, assay rows, workbook cells, trial registrations, UniProt/Rhea '
    'annotations, NCL protocols, ImmGen catalog) live in the resource pack: known record ID -> adjuvant_read_fields '
    'or adjuvant_cite; unknown ID -> adjuvant_resource_guide for the dataset then adjuvant_lookup with exact filters. '
    'Resource records are not paper results. '
    '(3) Supplied files -> code_interpreter with the exact paths. '
    'Answer the full question in the requested JSON; keep the answer specific, not a list of caveats.')

READERS = ['adjuvant_read_fields', 'adjuvant_record', 'adjuvant_search', 'adjuvant_lookup', 'adjuvant_resource_guide',
           'adjuvant_cite', 'adjuvant_excerpt']

RES_ROUTING = (
    'Resource-pack routing (offline snapshot). The readers adjuvant_read_fields, adjuvant_record, adjuvant_search, '
    'adjuvant_lookup, adjuvant_resource_guide and adjuvant_cite are ALREADY ENABLED: do not call catalog_search or '
    'tools_enable for them. '
    'Record ID taxonomy: tang2023:row:N and viget:row:N (N = CSV line number, header = line 1); '
    'vaxjo_master:row:N, vaxjo_audit:row:N, vaxjo_matrix:row:N (worksheet row); agent_optimization:Sheet1:row:N, '
    'agent_nhp:Sheet1:row:N, agent_storage:Sheet1:row:N (worksheet row as given); VO_NNNNNNN (ontology term: its '
    'label is the record title, its definition is in annotations); uniprot:<ACCESSION> (function text in '
    'annotations); rhea:<ID> (equation); ctgov:<NCT ID> (summary, status, design); ncl:<PROTOCOL>:page:N (field '
    'text); immgen:catalog:N (field scope states what data the catalog holds). '
    'Steps: (a) a row/sheet/accession/NCT/VO/RHEA identifier in the question -> build the ID from the taxonomy and '
    'call adjuvant_read_fields once with the requested original field names; on an unknown-field error call '
    'adjuvant_record(record_id) to see the real field names, never guess again. (b) An external key (Vaxjo ID, '
    'biosample_id, gsm, Catalog, PME Code, gene, adjuvant name) -> adjuvant_search with dataset and exact typed '
    'filters (numbers as numbers); never derive a row number from an external key; then read the returned ID. '
    '(c) A passage (protocol, annotation, registry summary, catalog scope) -> adjuvant_lookup(dataset, query=entity, '
    'focus=specific concept) or adjuvant_cite(record_id, field, focus); copy the returned quote, record_id and field '
    'exactly; field is the original top-level field name. (d) Check that the read record matches every locator '
    'value before answering. '
    'evidence_status: retrieved when a record of the requested type was read; insufficient only when the pack '
    'cannot hold the requested evidence (expression values, nonexistent target, a specific paper result) after '
    'one or two targeted reads; then cite the metadata that shows the gap (e.g. the ImmGen catalog scope field). '
    'Claims: supports = the read value equals the claimed value; contradicts = the record holds a different value '
    'for that exact field; not_determined = the claim goes beyond what one record can show (efficacy, mechanism, '
    'transfer to the user experiment). '
    'values: exactly the requested original field names with original JSON types (numbers stay numbers, null stays '
    'null, formula text and _cached_formula_values as stored). Hashes: copy the raw source hash returned by '
    'adjuvant_record (or manifest.json raw_sha256 for that dataset) character by character.')

CALC = (
    'code_interpreter: pass exact host file paths in inputs and open inputs/<basename> in the script; print the '
    'numbers you need. End every script with: import json, os; os.makedirs("outputs", exist_ok=True); '
    'json.dump({"summary": RESULT_DICT, "limitations": []}, open("outputs/analysis_result.json", "w")) where '
    'RESULT_DICT is a JSON object (dict) of your computed values; never a string or list, no other top-level keys. '
    'The tool message only shows about the first 1000 characters of stdout, so keep printed output small: print '
    'column names, shapes and value counts truncated to a few hundred characters, never raw file heads; print the '
    'final numbers as ONE compact JSON line. Do the whole pipeline (parse, join, filter, count) in one script '
    'rather than many small inspection calls. If you must see more stdout, artifact_read the stdout.txt artifact id '
    '(not full_result_artifact_id). pandas, numpy and openpyxl are installed. Read ZIP members in memory '
    '(zipfile.ZipFile(path).open(name) -> io.BytesIO -> pandas.read_excel/read_csv); never extract to /tmp; '
    'scratch files only under relative tmp/ or outputs/. GEO series-matrix files: lines starting with '
    '!Sample_ are tab-separated rows of per-sample values (one column per GSM); strip quotes. '
    'If a call fails, fix the reported error once instead of switching approach. Reserve the last 2 steps to '
    'deliver the final JSON.')

RES_V2 = (
    ' Output rules checked literally: (1) values = {} unless the question explicitly asks to return/read named '
    'fields into values ("in values", "into values", "return X and Y exactly", "using key", "读取字段"); explanatory '
    'questions ("quote", "explain", "inspect", "can it report") keep values {} and put facts in answer. '
    '(2) Every quote is one contiguous substring of the plain field text exactly as returned (no JSON keys, braces, '
    'escaped quotes or ellipses); keep it short (one sentence) and choose the sentence that literally contains the '
    'key concept words of the question (e.g. interference, inhibition, enhancement, recovery; contains aluminum '
    'hydroxide). Prefer adjuvant_cite(record_id, field, focus=<key words>) and copy its quote. '
    '(3) If the question asks for a data type the pack does not hold (expression values, measured results, '
    'efficacy), evidence_status = insufficient even though a catalog/metadata record was read; cite the field that '
    'states the gap (e.g. immgen:catalog:N field scope, text "no expression values"). '
    '(4) decision: contradicts ONLY when the claim names a value for one of the read fields and the record holds a '
    'different value for that same field. A claim about efficacy, mechanism, causality, BHMT dependence or the '
    "user's own experiment is not_determined, even if the record is about a different vaccine or sample.")

CALC_V2 = (
    ' Wrap the whole script body in try/except Exception as e: and in the except branch print one line '
    '"ERR", type(e).__name__, str(e)[:400] plus traceback.format_exc().splitlines()[-3:] joined, so the error is '
    'visible in the short stdout preview. The host file preview below already shows structure; do not spend calls '
    're-inspecting it - write the full computation in your first code_interpreter call.')

SHA_V3 = (' Source hash: tool results show both pack_sha256 (hash of the whole resource pack) and raw_sha256 (next to '
          'raw_filename, hash of the original data file). source_sha256 / raw source hash = raw_sha256 for that '
          "record's dataset, copied character by character; never pack_sha256 or manifest_hash.")


def _short(s, n=300):
    s = s.replace('\t', ' | ').replace('\r', '')
    return s if len(s) <= n else s[:n] + '…'


def file_preview(paths, limit=6000):
    """Generic structural preview of staged input files (names, sizes, headers, first rows). No reference content."""
    import io
    import zipfile
    from pathlib import Path
    out = []
    zips_detailed = 0
    for p in sorted(map(Path, paths), key=lambda x: (x.suffix == '.zip', x.stat().st_size)):
        size = p.stat().st_size
        head = f'FILE {p.name} ({size} bytes)'
        if p.suffix == '.zip' and zips_detailed >= 1:
            names = [n for n in zipfile.ZipFile(p).namelist() if not n.endswith('/')]
            out.append(f'{head}: zip with {len(names)} members, same layout as the first zip above')
            continue
        if p.suffix == '.zip':
            zips_detailed += 1
            z = zipfile.ZipFile(p)
            names = [n for n in z.namelist() if not n.endswith('/')]
            exts = {}
            for n in names:
                exts[Path(n).suffix or '(none)'] = exts.get(Path(n).suffix or '(none)', 0) + 1
            out.append(f'{head}: zip with {len(names)} members by extension {exts}; first members: '
                       + ', '.join(names[:8]) + '; directories: '
                       + ', '.join(sorted({str(Path(n).parent) for n in names})[:12]))
            shown = 0
            for n in names:
                if shown >= 3:
                    break
                if n.endswith(('.xlsx', '.xls')):
                    import openpyxl
                    wb = openpyxl.load_workbook(io.BytesIO(z.read(n)), data_only=True)
                    for ws in wb.worksheets[:2]:
                        rows = [[c for c in r] for _, r in zip(range(4), ws.iter_rows(values_only=True))]
                        out.append(f'  member {n} sheet {ws.title!r} first 4 rows: '
                                   + _short(json.dumps(rows, default=str), 1100))
                    shown += 1
                elif n.endswith(('.tsv', '.csv', '.txt', '.tab', '.faa', '.fasta', '.json', '.gff', '.tbl')) and shown < 3:
                    text = z.read(n)[:1500].decode('utf-8', 'replace').splitlines()[:3]
                    out.append(f'  member {n} first lines: ' + _short(' / '.join(text), 400))
                    shown += 1
        elif p.suffix in ('.csv', '.tsv', '.txt', '.json'):
            text = p.read_text(errors='replace')
            lines = text.splitlines()
            if p.suffix == '.json':
                out.append(f'{head}: ' + _short(text, 700))
            elif any(line.startswith('!') for line in lines[:200]):
                labels = [line.split('\t', 1) for line in lines if line.startswith('!Sample_')]
                out.append(f'{head}: {len(lines)} lines; GEO-style !label rows (label: first values): '
                           + '; '.join(_short(f'{a}: {b[:150]}', 190) for a, b in
                                       [(x[0], x[1] if len(x) > 1 else '') for x in labels[:25]]))
            else:
                out.append(f'{head}: {len(lines)} lines; first lines: ' + _short(' / '.join(lines[:3]), 600))
        else:
            out.append(head)
    text = '\n'.join(out)
    return text if len(text) <= limit else text[:limit] + '…'


def recipe(kind, prompt, input_files=()):
    """Session settings for one task, as in the frozen P1 runs."""
    if kind not in KINDS:
        raise ValueError('unknown task kind ' + str(kind))
    if kind == 'qa':
        instructions, host = ROUTING_V1 + ' ' + RES_ROUTING + RES_V2, 'library'
    elif kind in ('field', 'entry'):
        instructions, host = RES_ROUTING + RES_V2 + ' ' + CALC, 'readers'
    else:
        instructions, host = CALC + CALC_V2 + '\nHost file preview (structure only):\n' + file_preview(input_files or []), None
    if 'sha256' in prompt.lower():
        instructions += SHA_V3
    return dict(profile=PROFILE, instructions=instructions, tools=list(TOOLS), budget=dict(BUDGET), host_tools=host)


def service_options():
    return dict(request_output_tokens=REQUEST_OUTPUT_TOKENS, request_timeout_seconds=REQUEST_TIMEOUT_SECONDS)


def run(service, kind, prompt, *, allowed_roots, input_files=(), library_db=None):
    """One task in a fresh session; returns the delivery, run status and actions."""
    r = recipe(kind, prompt, input_files)
    service._prime()
    host = []
    if r['host_tools'] == 'library':
        host = library_search.register(service, library_db) + READERS
    elif r['host_tools'] == 'readers':
        host = list(READERS)
    session = open_session(service, profile=r['profile'], instructions=r['instructions'], allowed_roots=allowed_roots,
                           allow_network=False, budget=r['budget'], allowed_tools=r['tools'], host_tools=host)
    return dict(drive(service, session, prompt), session_config=session['config'])


SPEC = dict(name=NAME, profile=PROFILE, budget=BUDGET, request_output_tokens=REQUEST_OUTPUT_TOKENS,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS, tools=TOOLS, readers=READERS,
            texts=dict(ROUTING_V1=ROUTING_V1, RES_ROUTING=RES_ROUTING, CALC=CALC, RES_V2=RES_V2, CALC_V2=CALC_V2,
                       SHA_V3=SHA_V3, library_search=library_search.DESCRIPTION),
            library_search_parameters=library_search.PARAMETERS)
