"""Competing mechanism hypotheses as one acyclic graph per design revision.

The host checks structure, orders planned checkpoints from graph position and
propagates reported checkpoint states. Graph edges are authored hypotheses, not
inferred causal structure; ordering is not a measured discriminating power.
"""
from copy import deepcopy
from .checkpoints import LEVELS
from .validation import identifier, ids, text

ROOT = 'intervention'
NODE_LEVELS = (ROOT, *LEVELS)
RANK = {level: i for i, level in enumerate(NODE_LEVELS)}
FUNCTION = 'antigen_function'
KEYS = ('nodes', 'edges', 'hypotheses', 'checkpoint_nodes')
# Predictive graphs (v3) add per-hypothesis outcome predictions and assay cost tiers.
PREDICTIVE_KEYS = (*KEYS, 'predictions')
OUTCOMES = ('observed', 'not_observed', 'no_prediction')
PREDICTION_KEYS = ('checkpoint_id', 'hypothesis_id', 'outcome', 'reason')
COST_WEIGHT = {'low': 1, 'medium': 2, 'high': 3}


def _object(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(name + ' requires: ' + ', '.join(keys))


def _refs(item, claim_ids, name, *, required=True):
    ids(item['claim_ids'])
    if required and not item['claim_ids']:
        raise ValueError(name + ' requires bound claims')
    if not set(item['claim_ids']) <= set(claim_ids):
        raise ValueError(name + ' claims must be bound in verification')


def _order(nodes, edges):
    indegree = {n: 0 for n in nodes}
    children = {n: [] for n in nodes}
    for e in edges:
        children[e['source']].append(e['target']); indegree[e['target']] += 1
    ready = [n for n in nodes if not indegree[n]]
    order = []
    while ready:
        n = ready.pop(0); order.append(n)
        for c in children[n]:
            indegree[c] -= 1
            if not indegree[c]: ready.append(c)
    if len(order) != len(nodes):
        raise ValueError('mechanism_graph must be acyclic')
    return order, children


def ancestors(graph):
    order, children = _order([n['id'] for n in graph['nodes']], graph['edges'])
    up = {n: set() for n in order}
    for n in order:
        for c in children[n]:
            up[c] |= up[n] | {n}
    return up


def validate_mechanism_graph(graph, cards, claim_ids):
    predictive = isinstance(graph, dict) and 'predictions' in graph
    _object(graph, PREDICTIVE_KEYS if predictive else KEYS, 'mechanism_graph')
    nodes = graph['nodes']
    if not isinstance(nodes, list) or not 3 <= len(nodes) <= 30:
        raise ValueError('mechanism_graph requires 3..30 nodes')
    level = {}
    for n in nodes:
        _object(n, ('id', 'label', 'level', 'context', 'claim_ids'), 'mechanism node')
        identifier(n['id'])
        if n['id'] in level: raise ValueError('mechanism node IDs must be unique')
        if n['level'] not in NODE_LEVELS: raise ValueError('Unknown mechanism node level')
        text(n['label'], 'mechanism node label', 1000); text(n['context'], 'mechanism node context', 2000)
        _refs(n, claim_ids, 'mechanism node')
        level[n['id']] = n['level']
    roots = [k for k, v in level.items() if v == ROOT]
    if len(roots) != 1: raise ValueError('mechanism_graph requires exactly one intervention node')
    if FUNCTION not in level.values(): raise ValueError('mechanism_graph requires an antigen_function node')
    edges = graph['edges']
    if not isinstance(edges, list) or not 1 <= len(edges) <= 60:
        raise ValueError('mechanism_graph requires 1..60 edges')
    pairs = set()
    for e in edges:
        _object(e, ('source', 'target', 'relation', 'claim_ids'), 'mechanism edge')
        if e['source'] not in level or e['target'] not in level: raise ValueError('mechanism edge references unknown node')
        if e['source'] == e['target'] or (e['source'], e['target']) in pairs: raise ValueError('mechanism edges must be distinct non-loops')
        # Lower measurement levels must not be inferred from higher ones.
        if RANK[level[e['target']]] < RANK[level[e['source']]] or level[e['target']] == ROOT:
            raise ValueError('mechanism edges must not point to a lower evidence level')
        text(e['relation'], 'mechanism edge relation', 1000)
        _refs(e, claim_ids, 'mechanism edge', required=False)
        pairs.add((e['source'], e['target']))
    up = ancestors(graph)
    hypotheses = graph['hypotheses']
    if not isinstance(hypotheses, list) or not 2 <= len(hypotheses) <= 8:
        raise ValueError('mechanism_graph requires 2..8 competing hypotheses')
    seen, paths = set(), set()
    for h in hypotheses:
        _object(h, ('id', 'label', 'node_ids', 'claim_ids'), 'mechanism hypothesis')
        identifier(h['id'])
        if h['id'] in seen: raise ValueError('hypothesis IDs must be unique')
        text(h['label'], 'hypothesis label', 1000); ids(h['node_ids']); _refs(h, claim_ids, 'mechanism hypothesis')
        path = h['node_ids']
        if len(path) < 3 or path[0] != roots[0] or level.get(path[-1]) != FUNCTION:
            raise ValueError('hypothesis must run from the intervention through a mechanism to antigen_function')
        if any((a, b) not in pairs for a, b in zip(path, path[1:])):
            raise ValueError('hypothesis path must follow declared edges')
        if tuple(path) in paths: raise ValueError('hypotheses must be distinct paths')
        seen.add(h['id']); paths.add(tuple(path))
    by_card = {c['id']: c for c in cards}
    mapping = graph['checkpoint_nodes']
    if not isinstance(mapping, list) or len(mapping) != len(cards):
        raise ValueError('every checkpoint requires exactly one mechanism node')
    node_of = {}
    for m in mapping:
        _object(m, ('checkpoint_id', 'node_id', 'cost') if predictive else ('checkpoint_id', 'node_id'), 'checkpoint node')
        if predictive and m['cost'] not in COST_WEIGHT: raise ValueError('checkpoint cost must be low, medium or high')
        card = by_card.get(m['checkpoint_id'])
        if card is None or m['checkpoint_id'] in node_of: raise ValueError('checkpoint node mapping must cover each card once')
        if level.get(m['node_id']) in (None, ROOT): raise ValueError('checkpoint must target a measurable mechanism node')
        if (level[m['node_id']] == FUNCTION) != (card['branch'] == 'function'):
            raise ValueError('function checkpoints must target antigen_function nodes')
        if card.get('decision_level', level[m['node_id']]) != level[m['node_id']]:
            raise ValueError('checkpoint decision level does not match its node')
        node_of[m['checkpoint_id']] = m['node_id']
    for card in cards:
        prerequisite = {node_of[d] for d in card['depends_on']}
        if not prerequisite <= up[node_of[card['id']]]:
            raise ValueError('checkpoint dependencies must lie upstream in the mechanism graph')
    if predictive:
        _validate_predictions(graph, cards, node_of)
    return deepcopy(graph)


def _off_path_pairs(graph, cards, node_of):
    """Mechanism checkpoint × hypothesis pairs the graph cannot decide; on-path pairs are host-derived."""
    return {(c['id'], h['id']) for c in cards if c['branch'] == 'mechanism'
            for h in graph['hypotheses'] if node_of[c['id']] not in h['node_ids']}


def _validate_predictions(graph, cards, node_of):
    items = graph['predictions']
    required = _off_path_pairs(graph, cards, node_of)
    if not isinstance(items, list) or len(items) > 40:
        raise ValueError('predictions requires at most 40 items')
    seen = set()
    for p in items:
        _object(p, PREDICTION_KEYS, 'hypothesis prediction')
        pair = (p['checkpoint_id'], p['hypothesis_id'])
        if pair not in required:
            raise ValueError('predictions cover only mechanism checkpoints off a hypothesis path; on-path outcomes are host-derived')
        if pair in seen: raise ValueError('duplicate hypothesis prediction')
        if p['outcome'] not in OUTCOMES: raise ValueError('Unknown predicted outcome')
        text(p['reason'], 'prediction reason', 600)
        seen.add(pair)
    if seen != required:
        raise ValueError('predictions must cover every mechanism checkpoint for each hypothesis that bypasses its node')


def predicted_outcomes(graph, cards):
    """Outcome each hypothesis predicts at each mechanism checkpoint.

    A hypothesis whose path runs through the checkpoint's node requires that node,
    so the host fixes its prediction to 'observed'; the model only predicts for
    hypotheses that bypass the node.
    """
    node_of = {m['checkpoint_id']: m['node_id'] for m in graph['checkpoint_nodes']}
    given = {(p['checkpoint_id'], p['hypothesis_id']): p['outcome'] for p in graph.get('predictions', ())}
    return {(c['id'], h['id']): 'observed' if node_of[c['id']] in h['node_ids'] else given[(c['id'], h['id'])]
            for c in cards if c['branch'] == 'mechanism' for h in graph['hypotheses']}


OBSERVED = {'supports': 'observed', 'opposes': 'not_observed'}


def discrimination(outcomes):
    """Expected hypotheses eliminated (uniform prior) and definite pairs separated.

    A hypothesis without a prediction is compatible with either outcome and is
    taken as 50/50 when it is the true one.
    """
    n = len(outcomes)
    if not n: return 0.0, 0
    against = {o: sum(1 for x in outcomes if x not in (o, 'no_prediction')) for o in ('observed', 'not_observed')}
    expected = 0.0
    for x in outcomes:
        weights = {x: 1.0} if x != 'no_prediction' else {'observed': .5, 'not_observed': .5}
        expected += sum(w * against[o] for o, w in weights.items())
    definite = [x for x in outcomes if x != 'no_prediction']
    pairs = sum(1 for i, a in enumerate(definite) for b in definite[i + 1:] if a != b)
    return round(expected / n, 3), pairs


def derived_dependencies(graph, cards):
    """Mechanism checkpoints that measure strict ancestors of each card's node."""
    node_of = {m['checkpoint_id']: m['node_id'] for m in graph['checkpoint_nodes']}
    up = ancestors(graph)
    order = [c['id'] for c in cards]
    mech = {c['id'] for c in cards if c['branch'] == 'mechanism'}
    return {c['id']: sorted((d for d in mech if d != c['id'] and node_of[d] in up[node_of[c['id']]]), key=order.index)
            for c in cards}


LITERATURE_STATUSES = ('supports', 'opposes', 'mixed', 'not_addressed')
LITERATURE_KEYS = ('source', 'target', 'status', 'pmids', 'passages_retrieved', 'note', 'gate')


def validate_literature_edges(items, graph):
    """Host-written edge annotations; the model cannot author them in a decision."""
    if not isinstance(items, list) or not 1 <= len(items) <= 60:
        raise ValueError('literature_edges requires 1..60 host annotations')
    edges, seen = {(e['source'], e['target']) for e in graph['edges']}, set()
    for item in items:
        _object(item, LITERATURE_KEYS, 'literature edge')
        pair = (item['source'], item['target'])
        if pair not in edges or pair in seen:
            raise ValueError('literature_edges must annotate distinct declared edges')
        if item['status'] not in LITERATURE_STATUSES or item['gate'] not in ('', 'no_valid_citation'):
            raise ValueError('Unknown literature edge status')
        if (not isinstance(item['pmids'], list) or len(item['pmids']) > 10
                or any(not isinstance(p, str) or not p.isdigit() or len(p) > 12 for p in item['pmids'])):
            raise ValueError('literature edge PMIDs must be numeric strings')
        if (item['status'] == 'not_addressed') != (not item['pmids']):
            raise ValueError('only cited literature statuses carry PMIDs')
        if type(item['passages_retrieved']) is not int or not 0 <= item['passages_retrieved'] <= 20:
            raise ValueError('Invalid retrieved passage count')
        if not isinstance(item['note'], str) or len(item['note']) > 600:
            raise ValueError('literature edge note exceeds 600 characters')
        seen.add(pair)
    return deepcopy(items)


def _incoming_literature(graph, literature):
    status = {(x['source'], x['target']): x['status'] for x in literature or ()}
    incoming = {}
    for h in graph['hypotheses']:
        for a, b in zip(h['node_ids'], h['node_ids'][1:]):
            if (a, b) in status:
                incoming.setdefault(b, {})[a] = status[(a, b)]
    return incoming


def _combine(states):
    states = set(states)
    if 'conflicting' in states or {'supports', 'opposes'} <= states: return 'conflicting'
    for s in ('opposes', 'supports', 'inconclusive'):
        if s in states: return s
    return 'unmeasured'


def analyze(graph, cards, statuses, literature=None):
    """Host ordering and propagation over reported checkpoint states.

    ``statuses`` maps checkpoint IDs to experiment_state statuses. Function
    checkpoints stay a separate branch and do not weaken mechanism hypotheses.
    ``literature`` (host edge annotations) only breaks ties after the hypothesis
    split: a node whose incoming path edges are not all literature-supported is
    measured first. Literature never weakens or supports a hypothesis here.
    """
    incoming = _incoming_literature(graph, literature)
    predictive = 'predictions' in graph
    predicted = predicted_outcomes(graph, cards) if predictive else {}
    cost = {m['checkpoint_id']: m['cost'] for m in graph['checkpoint_nodes']} if predictive else {}
    level = {n['id']: n['level'] for n in graph['nodes']}
    node_of = {m['checkpoint_id']: m['node_id'] for m in graph['checkpoint_nodes']}
    up = ancestors(graph)
    on_node = {}
    for cid, nid in node_of.items():
        on_node.setdefault(nid, []).append(statuses.get(cid, 'unmeasured'))
    node_status = {nid: _combine(on_node[nid]) if nid in on_node else 'untested' for nid in level}
    mechanism = lambda path: [n for n in path if level[n] not in (ROOT, FUNCTION)]
    hypotheses, blocked = [], {}
    for h in graph['hypotheses']:
        steps = mechanism(h['node_ids'])
        states = [node_status[n] for n in steps]
        opposed = [n for n in steps if node_status[n] == 'opposes']
        # Off-path: a reliable result at a node the hypothesis bypasses contradicts its stated prediction.
        # Only a perturbation (causal_dependence) result eliminates the hypothesis; a correlational signal
        # (presence/activity) at a bypassed node merely co-occurs and cannot refute a competing route, so it
        # is recorded as a conflict but keeps the hypothesis live. This keeps a true mechanism from being
        # pruned when the model's off-path prediction about a co-occurring signature is wrong.
        contradicted = [c['id'] for c in cards if predictive and c['branch'] == 'mechanism'
                        and node_of[c['id']] not in h['node_ids'] and statuses.get(c['id']) in OBSERVED
                        and predicted[(c['id'], h['id'])] not in ('no_prediction', OBSERVED[statuses[c['id']]])]
        causal_contradicted = [c for c in contradicted if level[node_of[c]] == 'causal_dependence']
        if opposed:
            status = 'weakened_under_tested_conditions'
            first = steps.index(opposed[0])
            blocked[h['id']] = steps[first + 1:]
        elif causal_contradicted:
            status = 'weakened_under_tested_conditions'
            blocked[h['id']] = []
        elif contradicted: status = 'conflicting'
        elif 'conflicting' in states: status = 'conflicting'
        elif steps and all(s == 'supports' for s in states): status = 'supported_along_tested_nodes'
        else: status = 'open'
        row = {'hypothesis_id': h['id'], 'status': status, 'opposed_nodes': opposed,
               'withdrawn_downstream_nodes': blocked.get(h['id'], [])}
        if predictive: row['contradicted_by_off_path_checkpoints'] = contradicted
        hypotheses.append(row)
    live = [h for h, s in zip(graph['hypotheses'], hypotheses) if s['status'] != 'weakened_under_tested_conditions']
    live_nodes = {n for h in live for n in h['node_ids']}
    order = [c['id'] for c in cards]
    unresolved = lambda cid: statuses.get(cid, 'unmeasured') in ('unmeasured', 'inconclusive')
    ranked, deprioritized, function = [], [], []
    for card in cards:
        cid, nid = card['id'], node_of[card['id']]
        state = statuses.get(cid, 'unmeasured')
        if card['branch'] == 'function':
            function.append({'checkpoint_id': cid, 'node_id': nid, 'status': state}); continue
        if not unresolved(cid): continue
        if nid not in live_nodes:
            deprioritized.append({'checkpoint_id': cid, 'node_id': nid, 'reason': 'only_on_weakened_hypotheses'}); continue
        through = [h['id'] for h in live if nid in h['node_ids']]
        downstream = [c for c, n in node_of.items() if c != cid and nid in up[n] and unresolved(c) and n in live_nodes]
        waiting = [c for c, n in node_of.items() if n in up[nid] and unresolved(c) and level[n] != FUNCTION and n in live_nodes]
        edges_in = incoming.get(nid, {})
        row = {'checkpoint_id': cid, 'node_id': nid, 'status': state,
               'open_upstream_checkpoints': sorted(waiting, key=order.index),
               'live_hypotheses_through_node': through,
               'live_hypotheses_split': min(len(through), len(live) - len(through)),
               'downstream_checkpoints': sorted(downstream, key=order.index),
               **({'incoming_literature': edges_in} if literature is not None else {})}
        if predictive:
            outcomes = {h['id']: predicted[(cid, h['id'])] for h in live}
            expected, pairs = discrimination(list(outcomes.values()))
            row.update(predicted_outcomes=outcomes, expected_hypotheses_eliminated=expected,
                       hypothesis_pairs_separated=pairs, cost=cost[cid],
                       value_per_cost=round(expected / COST_WEIGHT[cost[cid]], 3))
        ranked.append(row)
    settled = lambda r: bool(r.get('incoming_literature')) and all(
        s == 'supports' for s in r['incoming_literature'].values())
    if predictive:
        ranked.sort(key=lambda r: (bool(r['open_upstream_checkpoints']), -r['value_per_cost'],
                                   -r['expected_hypotheses_eliminated'], settled(r),
                                   -len(r['live_hypotheses_through_node']), -len(r['downstream_checkpoints']),
                                   RANK[level[r['node_id']]], order.index(r['checkpoint_id'])))
    else:
        ranked.sort(key=lambda r: (bool(r['open_upstream_checkpoints']), -r['live_hypotheses_split'], settled(r),
                                   -len(r['live_hypotheses_through_node']), -len(r['downstream_checkpoints']),
                                   RANK[level[r['node_id']]], order.index(r['checkpoint_id'])))
    for i, r in enumerate(ranked, 1): r['rank'] = i
    keep = {c for h in live for c in h['claim_ids']} | {c for n in graph['nodes'] if n['id'] in live_nodes for c in n['claim_ids']}
    weakened = [h for h in graph['hypotheses'] if h['id'] in blocked]
    revisit = {c for h in weakened for c in h['claim_ids']}
    revisit |= {c for n in graph['nodes'] if any(n['id'] in h['node_ids'] for h in weakened) and n['id'] not in live_nodes for c in n['claim_ids']}
    measured = {node_of[c['id']] for c in cards if c['branch'] == 'mechanism'}
    unchecked = [h['id'] for h in graph['hypotheses'] if not measured & set(mechanism(h['node_ids']))]
    extra = {} if literature is None else {'literature_edges': deepcopy(literature)}
    if predictive:
        extra['derived_dependencies'] = derived_dependencies(graph, cards)
        extra['ranking_basis'] = ('Expected hypotheses eliminated under a uniform prior over live hypotheses, '
                                  'from host-derived on-path and model-stated off-path predictions, per cost tier.')
    return {**extra, 'node_status': node_status, 'hypotheses': hypotheses,
            'hypotheses_without_mechanism_checkpoint': unchecked,
            'mechanism_order': ranked, 'deprioritized_checkpoints': deprioritized,
            'parallel_function_checkpoints': function,
            'claims_to_revisit': sorted(revisit - keep),
            'scope': 'Host ordering from authored graph position and reported checkpoint states; '
                     'not inferred causal structure, measured discriminating power or vaccine efficacy.'}


def graph_schema(id_schema):
    def obj(fields): return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}
    def txt(n): return {'type': 'string', 'minLength': 1, 'maxLength': n}
    refs = {'type': 'array', 'items': id_schema, 'maxItems': 100, 'uniqueItems': True}
    node = obj({'id': id_schema, 'label': txt(1000), 'level': {'enum': list(NODE_LEVELS)}, 'context': txt(2000), 'claim_ids': refs})
    edge = obj({'source': id_schema, 'target': id_schema, 'relation': txt(1000), 'claim_ids': refs})
    hypothesis = obj({'id': id_schema, 'label': txt(1000), 'node_ids': {'type': 'array', 'items': id_schema, 'minItems': 3, 'maxItems': 30}, 'claim_ids': refs})
    mapping = obj({'checkpoint_id': id_schema, 'node_id': id_schema})
    return obj({'nodes': {'type': 'array', 'minItems': 3, 'maxItems': 30, 'items': node},
                'edges': {'type': 'array', 'minItems': 1, 'maxItems': 60, 'items': edge},
                'hypotheses': {'type': 'array', 'minItems': 2, 'maxItems': 8, 'items': hypothesis},
                'checkpoint_nodes': {'type': 'array', 'minItems': 3, 'maxItems': 5, 'items': mapping}})


OUTCOME_LABELS = {'observed': '应观测到', 'not_observed': '应观测不到', 'no_prediction': '无预测'}
COST_LABELS = {'low': '低', 'medium': '中', 'high': '高'}
LITERATURE_LABELS = {'supports': '文献支持', 'opposes': '文献反对', 'mixed': '文献分歧', 'not_addressed': '未检到相关文献'}


def render_literature_edges(items, labels):
    from .rendering import authored
    lines = ['', '### 机制边的文献状态（宿主检索、模型判定、宿主核对引用）', '',
             '检索为关键词检索，判定未经人工核实；文献状态不是本设计的实验结果，也不改变假设状态。', '']
    for x in items:
        cited = '、'.join(f'[PMID {p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)' for p in x['pmids'])
        line = (f"- {authored(labels[x['source']])} → {authored(labels[x['target']])}："
                f"{LITERATURE_LABELS[x['status']]}（检得 {x['passages_retrieved']} 段）")
        if cited: line += '；' + cited
        if x['note']: line += '；' + authored(x['note'])
        if x['gate']: line += '；模型判定未引用本边检得的段落，宿主记为未检到'
        lines.append(line)
    return lines


def render_mechanism_graph(graph, cards, refs, literature=None):
    from .rendering import authored
    labels = {n['id']: n['label'] for n in graph['nodes']}
    plan = analyze(graph, cards, {}, literature)
    lines = ['', '### 机制假设图（宿主排序，计划）', '']
    for h in graph['hypotheses']:
        lines.extend(['- ' + authored(h['id']) + '：' + authored(h['label']),
                      '  路径：' + ' → '.join(authored(labels[n]) for n in h['node_ids']),
                      '  绑定论断：' + refs(h)])
    predictive = 'predictions' in graph
    if predictive:
        lines.extend(['', '机制检测顺序（宿主按各假设的预期结果计算期望淘汰数、再除以成本档；预期结果是假设，不是实验结果）：', ''])
    else:
        lines.extend(['', '机制检测顺序（由图中位置计算，不代表实验区分力已被证实）：', ''])
    for r in plan['mechanism_order']:
        if predictive:
            seen = '、'.join(f"{authored(h)}{OUTCOME_LABELS[o]}" for h, o in r['predicted_outcomes'].items())
            reason = (f"期望淘汰 {r['expected_hypotheses_eliminated']} 个假设，可区分 {r['hypothesis_pairs_separated']} 对；"
                      f"成本{COST_LABELS[r['cost']]}；各假设预期：{seen}；下游检测 {len(r['downstream_checkpoints'])} 项")
        else:
            reason = f"区分 {r['live_hypotheses_split']} 组竞争假设；经过 {len(r['live_hypotheses_through_node'])} 条假设路径；下游检测 {len(r['downstream_checkpoints'])} 项"
        if r['open_upstream_checkpoints']: reason += '；需先完成 ' + ', '.join(authored(c) for c in r['open_upstream_checkpoints'])
        if r.get('incoming_literature') and all(s == 'supports' for s in r['incoming_literature'].values()):
            reason += '；入边已有文献支持，同等条件下后测'
        lines.append(f"{r['rank']}. " + authored(r['checkpoint_id']) + '（' + authored(labels[r['node_id']]) + '）：' + reason)
    if predictive:
        deps = {k: v for k, v in plan['derived_dependencies'].items() if v}
        if deps:
            lines.extend(['', '由图推导的检测依赖（上游节点的机制检测）：' + '；'.join(
                f"{authored(k)} ← {', '.join(authored(d) for d in v)}" for k, v in deps.items())])
        off = [p for p in graph['predictions'] if p['outcome'] != 'no_prediction']
        if off:
            lines.extend(['', '路径外假设的预期（模型陈述，宿主据此计算区分力）：', ''])
            lines.extend(f"- {authored(p['checkpoint_id'])} × {authored(p['hypothesis_id'])}：{OUTCOME_LABELS[p['outcome']]}；{authored(p['reason'])}"
                         for p in off)
    if plan['hypotheses_without_mechanism_checkpoint']:
        lines.extend(['', '未设机制检测、当前计划无法区分的假设：' + ', '.join(authored(h) for h in plan['hypotheses_without_mechanism_checkpoint'])])
    if plan['parallel_function_checkpoints']:
        lines.extend(['', '功能检测独立并行：' + ', '.join(authored(f['checkpoint_id']) for f in plan['parallel_function_checkpoints'])])
    if literature:
        lines.extend(render_literature_edges(literature, labels))
    return lines
