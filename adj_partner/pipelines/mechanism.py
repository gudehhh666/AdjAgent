"""F7 mechanism graph, host-driven steps around the solver.

v2 (retrospective grounding): the solver writes per-edge queries; the host retrieves literature up to a cutoff
year; the solver judges each edge; the host citation gate keeps only judgments citing that edge's passages; the
final mechanism answer is written only from gated edges. v3 (decision planning): the solver states off-path
predictions and its own order; the host ranks checkpoints by expected eliminations per cost, propagates each
readout and replans until one hypothesis is live.

``complete(messages, max_output_tokens) -> str`` is one solver call; transport retries belong to the caller.
"""
import json
import re

from ..research import mechanism_graph as mg
from ..research.mechanism_graph import validate_literature_edges
from ..task_modes import graph_grounding as gg

CONFIGS = {
    'base':     dict(maxq=2, k_papers=2, k_abstracts=1, k_passages=3, host_expand=False),
    'improved': dict(maxq=3, k_papers=2, k_abstracts=3, k_passages=6, host_expand=True),
}

ANSWER_SYSTEM = '''You write the mechanism conclusion for a vaccine-adjuvant design question, limited to literature
available up to the stated cutoff year. You are given mechanism-graph edges, each with a host-verified literature
status and (for cited edges) the retrieved passages.
RULES:
- Assert a pathway/receptor as the established mechanism ONLY if its edge status is "supports" with cited passages.
- Treat "mixed" as contested and "opposes" / "not_addressed" as not established by the literature up to the cutoff.
- Do NOT use any knowledge beyond the provided passages; do not add receptors or pathways that are not listed.
- If no edge is "supports", state plainly that the principal mechanism is not established by the literature as of the cutoff.
- Name the single best-supported principal mechanism if one exists. Cite passages as [E#.P#]. Answer in 3-6 sentences.'''

STOP = {'the', 'of', 'and', 'a', 'as', 'an', 'with', 'for', 'from', 'by', 'to', 'in', 'on', 'acting', 'signal',
        'danger', 'recognition', 'activation', 'sensing', 'sensor', 'principal', 'sole', 'direct', 'obligatory',
        'upstream', 'conserved', 'long', 'cytosolic', 'endogenous', 'release', 'release'}


def host_query(graph, edge):
    """Host-side query expansion aimed at the key receptor/pathway (improved config only)."""
    target = edge['claim'].split(' -> ', 1)[1].split(':', 1)[0]
    toks = re.findall(r'[A-Za-z0-9]+', target)
    kws = [t for t in toks if t.lower() not in STOP][:5]
    intervention = re.findall(r'[A-Za-z0-9]+', graph['nodes'][0]['label'])[0]
    return ' '.join(dict.fromkeys([intervention] + kws))


def _gather(search, claim, queries, cfg):
    """Host retrieval for one edge, resilient to transient Europe PMC / NCBI network failures."""
    import time
    import httpx
    if not queries:
        return []
    for attempt in range(4):
        try:
            found = search.gather(claim, queries, k_papers=cfg['k_papers'], k_abstracts=cfg['k_abstracts'],
                                  k_passages=cfg['k_passages'], max_chars=900)
            return found['passages']
        except (httpx.HTTPError, OSError):
            if attempt == 3:
                return []
            time.sleep(3 * (attempt + 1))
    return []


def _raw_statuses(text, edges):
    value = gg._json(text) or {'edges': []}
    out = {}
    for row in value['edges']:
        if isinstance(row, dict) and row.get('edge') in {e['edge'] for e in edges}:
            out[row['edge']] = row.get('status')
    return out


def answer_messages(question, cutoff, edges, items, evidence):
    by_edge = {e['edge']: e for e in edges}
    blocks = []
    for it in items:
        e = by_edge[it['edge']]
        line = f"## {it['edge']}: {e['claim']}\nhost_status: {it['status']}"
        if it['gate']:
            line += f" (gate: {it['gate']})"
        rows = evidence.get(it['edge'], []) if it['status'] in ('supports', 'mixed', 'opposes') else []
        if rows:
            line += '\n' + '\n'.join(f"[{pid}] {p['title']} ({p['year']}): {p['text']}"
                                     for pid, p in zip(gg.passage_ids(it['edge'], rows), rows))
        else:
            line += '\n(no qualifying literature up to the cutoff)'
        blocks.append(line)
    user = (f"Design question: {question}\nCutoff year: {cutoff}\n\n"
            f"Mechanism-graph edges with host-verified literature status:\n\n" + '\n\n'.join(blocks))
    return [{'role': 'system', 'content': ANSWER_SYSTEM.replace('the cutoff', f'{cutoff}')}, {'role': 'user', 'content': user}]


QUERY_TOKENS, JUDGE_TOKENS, ANSWER_TOKENS = 1500, 3000, 1500


def ground(complete, question, graph, cutoff, *, config='base', retriever=None, ncbi=(None, None)):
    """Gate every hypothesis-path edge against literature published up to ``cutoff``."""
    from ..native.grounding import GroundingRetriever
    cfg = CONFIGS[config]
    edges = gg.targets(graph)
    queries = gg.parse_queries(complete(gg.query_messages(question, edges, max_queries=cfg['maxq']), QUERY_TOKENS),
                               edges, max_queries=cfg['maxq'])
    if cfg['host_expand']:
        for e in edges:
            hq = host_query(graph, e)
            queries[e['edge']] = ([hq] + [q for q in queries[e['edge']] if q != hq])[:cfg['maxq']]
    search = retriever or GroundingRetriever(max_year=cutoff, api_key=ncbi[0], email=ncbi[1], timeout_seconds=25)
    evidence = {}
    for e in edges:
        # PubMed maxdate can admit an online-ahead-of-print paper dated cutoff+1.
        evidence[e['edge']] = [p for p in _gather(search, e['claim'], queries[e['edge']], cfg)
                               if not p.get('year') or int(p['year']) <= cutoff]
    jtext = complete(gg.judge_messages(question, edges, evidence), JUDGE_TOKENS)
    items = validate_literature_edges(gg.gate(jtext, edges, evidence), graph)
    raw = _raw_statuses(jtext, edges)
    years = {p['pmid']: p['year'] for rows in evidence.values() for p in rows}
    for item, e in zip(items, edges):
        item['edge'] = e['edge']
        item['claim'] = e['claim']
        item['model_status'] = raw.get(e['edge'])
        item['cited_years'] = [years.get(p) for p in item['pmids']]
    return dict(cutoff=cutoff, config=config, queries=queries, targets=edges, edges=items, evidence=evidence)


def answer(complete, question, grounded):
    """Mechanism conclusion from the gated edges and their passages only."""
    return complete(answer_messages(question, grounded['cutoff'], grounded['targets'], grounded['edges'],
                                    grounded['evidence']), ANSWER_TOKENS)


COST_BY_LEVEL = {'molecular_presence': 'low', 'molecular_activity': 'medium', 'metabolic_link': 'medium',
                 'transcript': 'low', 'causal_dependence': 'high', 'antigen_function': 'medium'}

PREDICT_SYSTEM = ('You plan discriminating experiments for competing mechanism hypotheses. Respond ONLY with JSON. '
                  'For each mechanism checkpoint and each listed off-path hypothesis (one whose path bypasses that '
                  "checkpoint's node), state the outcome that hypothesis predicts there: \"observed\", \"not_observed\", "
                  'or \"no_prediction\" if that hypothesis is genuinely silent. Predictions are hypotheses, not results.')
ORDER_SYSTEM = ('You plan discriminating experiments. Respond ONLY with JSON {"order": ["C1", ...]} listing the '
                'mechanism checkpoints from most to least decisive for telling the competing hypotheses apart. '
                'Include every mechanism checkpoint exactly once.')


def _json(text):
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise ValueError('model did not return JSON')
    return json.loads(text[start:end + 1])


def costs_of(graph):
    return {m['checkpoint_id']: COST_BY_LEVEL[next(n['level'] for n in graph['nodes'] if n['id'] == m['node_id'])]
            for m in graph['checkpoint_nodes']}


def payload(public):
    g = public['graph']
    labels = {n['id']: n['label'] for n in g['nodes']}
    node_of = {m['checkpoint_id']: m['node_id'] for m in g['checkpoint_nodes']}
    level = {n['id']: n['level'] for n in g['nodes']}
    cost = costs_of(g)
    checks = [dict(id=c['id'], branch=c['branch'], level=c['decision_level'], measures=labels[node_of[c['id']]],
                   cost=cost.get(c['id']),
                   off_path_hypotheses=[h['id'] for h in g['hypotheses'] if node_of[c['id']] not in h['node_ids']])
              for c in public['cards'] if c['branch'] == 'mechanism']
    return dict(question=public['question'],
                hypotheses=[dict(id=h['id'], label=h['label'],
                                 path=[labels[n] for n in h['node_ids'] if level[n] not in (mg.ROOT,)])
                            for h in g['hypotheses']],
                checkpoints=checks)


def required_pairs(public):
    g = public['graph']
    node_of = {m['checkpoint_id']: m['node_id'] for m in g['checkpoint_nodes']}
    return [(c['id'], h['id']) for c in public['cards'] if c['branch'] == 'mechanism'
            for h in g['hypotheses'] if node_of[c['id']] not in h['node_ids']]


def predictions_map(public, stated):
    given = {(p['checkpoint_id'], p['hypothesis_id']): p.get('outcome') for p in stated['predictions']
             if isinstance(p, dict) and p.get('outcome') in mg.OUTCOMES}
    reasons = {(p['checkpoint_id'], p['hypothesis_id']): (p.get('reason') or '')[:580] for p in stated['predictions']
               if isinstance(p, dict)}
    out = []
    for cid, hid in required_pairs(public):
        out.append(dict(checkpoint_id=cid, hypothesis_id=hid, outcome=given.get((cid, hid), 'no_prediction'),
                        reason=reasons.get((cid, hid), 'no reason supplied')))
    return out


def mechanism_graph_nodes(public, hid):
    h = next(h for h in public['graph']['hypotheses'] if h['id'] == hid)
    level = {n['id']: n['level'] for n in public['graph']['nodes']}
    return [n for n in h['node_ids'] if level[n] not in (mg.ROOT, mg.FUNCTION)]


def checkpoint_at(public, node_id):
    return next((m['checkpoint_id'] for m in public['graph']['checkpoint_nodes'] if m['node_id'] == node_id), None)


def library_live_and_order(graph, cards, statuses):
    res = mg.analyze(graph, cards, statuses)
    live = [h['hypothesis_id'] for h in res['hypotheses'] if h['status'] != 'weakened_under_tested_conditions']
    order = [r['checkpoint_id'] for r in res['mechanism_order']]
    return live, order


PREDICT_TOKENS, ORDER_TOKENS = 3200, 1500


def elicit(complete, public):
    """Solver off-path predictions and its own checkpoint order for a public case (question, graph, cards)."""
    body = json.dumps(payload(public), ensure_ascii=False)
    pairs = required_pairs(public)
    shape = ('{"predictions": [{"checkpoint_id": "C1", "hypothesis_id": "H2", "outcome": "not_observed", '
             '"reason": "one sentence"}]}\nCover exactly these (checkpoint_id, hypothesis_id) pairs: '
             + json.dumps([list(p) for p in pairs]))
    stated = _json(complete([{'role': 'system', 'content': PREDICT_SYSTEM},
                             {'role': 'user', 'content': body + '\nReturn ' + shape}], PREDICT_TOKENS))
    order = _json(complete([{'role': 'system', 'content': ORDER_SYSTEM}, {'role': 'user', 'content': body}],
                           ORDER_TOKENS))['order']
    return dict(predictions=stated.get('predictions', []), model_order=order)


def build_graph(public, stated):
    """The v3 graph: public graph plus host cost tiers and the solver's predictions, validated."""
    g3 = json.loads(json.dumps(public['graph']))
    cost = costs_of(g3)
    for m in g3['checkpoint_nodes']:
        m['cost'] = cost[m['checkpoint_id']]
    g3['predictions'] = predictions_map(public, stated)
    mg.validate_mechanism_graph(g3, public['cards'], ['c'])
    return g3


def plan(graph, cards, statuses):
    """Live hypotheses and the next checkpoint to read (None once one hypothesis is left or nothing is open)."""
    live, order = library_live_and_order(graph, cards, statuses)
    return dict(live=live, order=order, next=order[0] if len(live) > 1 and order else None)


def settle(public, graph, statuses):
    """The concluded hypothesis: the single live one, else the strictly best-supported live one, else None."""
    live, _ = library_live_and_order(graph, public['cards'], statuses)
    if len(live) == 1:
        return live[0]
    supports = {h: sum(1 for n in mechanism_graph_nodes(public, h) if statuses.get(checkpoint_at(public, n)) == 'supports')
                for h in live}
    best = sorted(supports.items(), key=lambda kv: -kv[1])
    return best[0][0] if len(best) == 1 or (len(best) > 1 and best[0][1] > best[1][1]) else None


SETTINGS = ('retro', 'decision')
SPEC = dict(retro=dict(configs=CONFIGS, default='base', tokens=[QUERY_TOKENS, JUDGE_TOKENS, ANSWER_TOKENS],
                       texts=dict(ANSWER_SYSTEM=ANSWER_SYSTEM), stop=sorted(STOP)),
            decision=dict(cost_by_level=COST_BY_LEVEL, tokens=[PREDICT_TOKENS, ORDER_TOKENS],
                          texts=dict(PREDICT_SYSTEM=PREDICT_SYSTEM, ORDER_SYSTEM=ORDER_SYSTEM)))
