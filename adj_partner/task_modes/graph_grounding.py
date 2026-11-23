"""Literature status of mechanism-graph edges: model queries, host retrieval, model judgment, host gate.

Only edges on authored hypothesis paths are checked. A supports/opposes/mixed judgment must cite at
least one passage the host retrieved for that same edge; otherwise the host records not_addressed.
Literature status is context for ordering and reporting, never an experimental result.
"""
import json
import re

WORKFLOW_VERSION = 'unified-mechanism-graph-v2'
MAX_EDGES = 12
STATUSES = ('supports', 'opposes', 'mixed', 'not_addressed')

QUERY_SYSTEM = '''You write PubMed keyword queries that would find published evidence for or against specific mechanism
edges of a vaccine adjuvant design. For each listed edge give 1-2 queries of 3-6 keywords, using words likely to appear
in titles or abstracts (agent, receptor, pathway, cell type, readout). No boolean operators, quotes or field tags.
Return ONLY JSON {"edges": [{"edge": "E1", "queries": ["..."]}]}.'''

JUDGE_SYSTEM = '''You judge whether retrieved literature passages bear on mechanism edges of a vaccine adjuvant design.
Passages were found automatically by keyword search and may be irrelevant. For each edge choose exactly one status:
supports (a passage reports this relation in a comparable context), opposes (a passage reports its absence or the
reverse), mixed (passages disagree), not_addressed (no passage bears on the edge). Cite the passage IDs you rely on;
a status other than not_addressed without a valid citation of that edge's passages is discarded. Different species,
cell types or agents count only if you say so in the note. Return ONLY JSON
{"edges": [{"edge": "E1", "status": "...", "passages": ["E1.P1"], "note": "one sentence"}]}.'''


def targets(graph):
    nodes = {n['id']: n for n in graph['nodes']}
    relation = {(e['source'], e['target']): e['relation'] for e in graph['edges']}
    pairs = []
    for h in graph['hypotheses']:
        for pair in zip(h['node_ids'], h['node_ids'][1:]):
            if pair not in pairs:
                pairs.append(pair)
    return [{'edge': f'E{i}', 'source': a, 'target': b,
             'claim': f"{nodes[a]['label']} -> {nodes[b]['label']}: {relation[(a, b)]}",
             'context': nodes[b]['context']}
            for i, (a, b) in enumerate(pairs[:MAX_EDGES], 1)]


def query_messages(question, edges, max_queries=2):
    """max_queries caps the queries requested per edge; the default keeps the frozen workflow's prompt."""
    listed = [{'edge': e['edge'], 'claim': e['claim'], 'context': e['context']} for e in edges]
    content = json.dumps({'design_question': question, 'edges': listed}, ensure_ascii=False)
    if max_queries != 2:
        content += f'\n\nGive up to {max_queries} queries per edge, most specific first.'
    return [{'role': 'system', 'content': QUERY_SYSTEM},
            {'role': 'user', 'content': content}]


def _json(text):
    match = re.search(r'\{.*\}', text or '', re.S)
    try:
        value = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and isinstance(value.get('edges'), list) else None


def parse_queries(text, edges, max_queries=2):
    known = {e['edge'] for e in edges}
    value = _json(text) or {'edges': []}
    found = {}
    for row in value['edges']:
        if not isinstance(row, dict) or row.get('edge') not in known or not isinstance(row.get('queries'), list):
            continue
        queries = [re.sub(r'["()\[\]]', ' ', q).strip()[:200] for q in row['queries'] if isinstance(q, str)]
        found[row['edge']] = [q for q in queries if q][:max_queries]
    return {e['edge']: found.get(e['edge'], []) for e in edges}


def passage_ids(edge, passages):
    return [f'{edge}.P{i}' for i in range(1, len(passages) + 1)]


def judge_messages(question, edges, evidence):
    blocks = []
    for e in edges:
        rows = evidence.get(e['edge'], [])
        text = '\n'.join(f"[{pid}] {p['title']} ({p['year']}), {p['section'] or 'Body'}: {p['text']}"
                         for pid, p in zip(passage_ids(e['edge'], rows), rows)) or '(no passages retrieved)'
        blocks.append(f"## {e['edge']}: {e['claim']}\nContext: {e['context']}\n{text}")
    return [{'role': 'system', 'content': JUDGE_SYSTEM},
            {'role': 'user', 'content': f'Design question: {question}\n\n' + '\n\n'.join(blocks)}]


def gate(text, edges, evidence):
    """Host gate: unknown edges ignored, uncited or foreign citations downgrade to not_addressed."""
    value = _json(text) or {'edges': []}
    judged = {row['edge']: row for row in value['edges'] if isinstance(row, dict) and isinstance(row.get('edge'), str)}
    items = []
    for e in edges:
        rows = evidence.get(e['edge'], [])
        own = dict(zip(passage_ids(e['edge'], rows), rows))
        row = judged.get(e['edge'], {})
        status = row.get('status') if row.get('status') in STATUSES else 'not_addressed'
        cited = [pid for pid in row.get('passages', []) if isinstance(pid, str) and pid in own] if isinstance(row.get('passages'), list) else []
        note = row.get('note') if isinstance(row.get('note'), str) else ''
        gated = ''
        if status != 'not_addressed' and not cited:
            status, gated = 'not_addressed', 'no_valid_citation'
        if status == 'not_addressed':
            cited = []
        items.append({'source': e['source'], 'target': e['target'], 'status': status,
                      'pmids': list(dict.fromkeys(own[pid]['pmid'] for pid in cited)),
                      'passages_retrieved': len(rows), 'note': note.strip()[:600], 'gate': gated})
    return items
