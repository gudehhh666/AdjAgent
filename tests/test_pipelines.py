"""Frozen pipeline catalog: equality with the runs behind the frozen results, and runnable flows offline."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from adj_partner import pipelines
from adj_partner.contracts import ModelReply
from adj_partner.models import ScriptedPort
from adj_partner.pipelines import (citation_binding, citations, direct, evidence_bound, guided, library_search,
                                   mechanism, research_report, synth)
from adj_partner.research import mechanism_graph as mg
from adj_partner.task_modes import graph_grounding as gg
from test_adjuvant_resource_pack import make_pack
from test_application import PROJECT, service_type
from test_mechanism_graph import betaine, cards

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / 'experiments/p1_resource_data'
RETRO = ROOT / 'experiments/p3_mechanism_retro/run.py'
DECISION = ROOT / 'experiments/p3_mechanism_decision/run.py'
PLGA = ROOT / 'experiments/p3_plga/run.py'
UNREGISTERED = {'AdjNER': 'pre-freeze 2026-09-30 pilot', 'AdjDrylab-ADB': 'no incumbent yet'}


def reply(text='done', finish='stop'):
    return ModelReply(content=text, tool_calls=[], finish_reason=finish, usage={'input_tokens': 1, 'output_tokens': 1})


def top_level(path):
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in ([target] if isinstance(target, ast.Name) else getattr(target, 'elts', [])):
                    out[name.id] = node
    return out


def literal(path, name):
    return ast.literal_eval(top_level(path)[name].value)


def code(node, *replacements):
    text = ast.unparse(node)
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def same_functions(ours, theirs, names, *replacements):
    mine, frozen = top_level(ours), top_level(theirs)
    for name in names:
        assert code(mine[name]) == code(frozen[name], *replacements), name


def frozen_functions(path, names, **namespace):
    """Execute selected top-level definitions of a frozen runner without importing its module."""
    defs = top_level(path)
    for name in names:
        exec(compile(ast.Module(body=[defs[name]], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_every_frozen_incumbent_slot_has_a_registered_pipeline_of_its_form():
    freeze = ROOT / 'harness_freeze.json'
    if not freeze.exists():
        pytest.skip('freeze record not present')
    for slot in json.loads(freeze.read_text())['slots']:
        pipeline = pipelines.for_slot(slot['task'])
        if slot['task'] in UNREGISTERED:
            assert pipeline is None
            continue
        assert pipeline is not None, slot['task']
        assert re.match(r'F\d+b?', slot['incumbent']['form']).group(0) == pipeline.form
    assert len({p.name for p in pipelines.PIPELINES.values()}) == len(pipelines.PIPELINES)
    assert all(len(pipelines.digest(name)) == 64 for name in pipelines.PIPELINES)


def test_direct_pipeline_is_one_tool_free_call():
    port = ScriptedPort([reply('Answer.')])
    out = direct.run(port, 'System.', 'Question?', max_output_tokens=900, timeout_seconds=30)
    assert out.content == 'Answer.' and port.requests == [
        {'max_output_tokens': 900, 'messages': [{'content': 'System.', 'role': 'system'},
                                                {'content': 'Question?', 'role': 'user'}],
         'timeout_seconds': 30, 'tools': []}]


def test_controller_pipeline_is_the_registered_active_workflow():
    from adj_partner.orchestration.registry import WORKFLOWS
    spec = pipelines.spec('qa-grounded-revision-v2')
    assert spec['status'] == 'active' and WORKFLOWS[spec['workflow']].runtime == 'qa_grounded_v2_runtime'
    assert pipelines.get('qa-grounded-revision-v2').module() is None


def library_db(path):
    conn = sqlite3.connect(path)
    conn.executescript('''
        CREATE TABLE papers(paper_id TEXT PRIMARY KEY, title TEXT, year INTEGER, best_text TEXT);
        CREATE TABLE passages(id INTEGER PRIMARY KEY, paper_id TEXT, section TEXT, locator TEXT, text TEXT);
        CREATE VIRTUAL TABLE passages_fts USING fts5(text, section, title, content='');''')
    rows = [(1, 'p1', 'Results', 'para:1', 'Alum adjuvant raised IgG1 titers in BALB/c mice.', 'Alum study'),
            (2, 'p1', 'References', 'reference:1', 'Alum adjuvant reference entry.', 'Alum study'),
            (3, 'p2', 'Abstract', 'para:1', 'CpG adjuvant shifted responses toward Th1 in mice.', 'CpG study')]
    conn.executemany('INSERT INTO papers VALUES(?,?,?,?)', [('p1', 'Alum study', 2020, 'full_text'),
                                                            ('p2', 'CpG study', 2021, 'abstract')])
    for rowid, paper, section, locator, text, title in rows:
        conn.execute('INSERT INTO passages VALUES(?,?,?,?,?)', (rowid, paper, section, locator, text))
        conn.execute('INSERT INTO passages_fts(rowid, text, section, title) VALUES(?,?,?,?)', (rowid, text, section, title))
    conn.commit()
    conn.close()
    return path


def frozen_p1(monkeypatch):
    if not (P1 / 'variants.py').exists():
        pytest.skip('frozen P1 runner copies not present')
    monkeypatch.syspath_prepend(str(P1))
    loaded = {}
    for name in ('p1', 'variants', 'library_tool'):
        spec = importlib.util.spec_from_file_location(name, P1 / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def test_evidence_bound_recipe_equals_the_frozen_p1_harness_v3(tmp_path, monkeypatch):
    frozen = frozen_p1(monkeypatch)
    variants = frozen['variants']
    data = tmp_path / 'table.csv'
    data.write_text('gene,value\nIL6,2.5\n')
    hosts = {variants.library_setup: 'library', variants.readers_setup: 'readers', None: None}
    for kind in evidence_bound.KINDS:
        for prompt in ('Read the record fields.', 'Return the source SHA256 and the values.'):
            task = dict(kind=kind, prompt=prompt, input_files=[str(data)] if kind == 'data' else [])
            old, new = variants.config(variants.FINAL, task), evidence_bound.recipe(kind, prompt, task['input_files'])
            assert {k: old[k] for k in ('profile', 'instructions', 'tools', 'budget')} == \
                {k: new[k] for k in ('profile', 'instructions', 'tools', 'budget')}
            assert hosts[old['setup']] == new['host_tools'] and old['goal'] is old['fallback'] is None
            assert (old['request_output_tokens'], old['request_timeout_seconds']) == \
                tuple(evidence_bound.service_options().values())
    assert variants.FINAL == 'harness_v3' and variants.READERS == evidence_bound.READERS
    with pytest.raises(ValueError):
        evidence_bound.recipe('report', 'x')


def test_library_search_equals_the_frozen_tool(tmp_path, monkeypatch):
    frozen = frozen_p1(monkeypatch)['library_tool']
    db = library_db(tmp_path / 'library.sqlite3')
    monkeypatch.setattr(frozen, 'DB', db)
    monkeypatch.setattr(frozen, '_LOCAL', type(frozen._LOCAL)())
    for query, paper in [('alum adjuvant mice', None), ('adjuvant mice', None), ('Th1 response', 'p2'), ('of the', None)]:
        assert frozen.search(query, paper) == library_search.search(str(db), query, paper)
    assert library_search.search(str(db), 'alum adjuvant')[0]['locator'] == 'para:1'
    captured = {}
    catalog = SimpleNamespace(register=lambda spec, handler: captured.setdefault(len(captured), spec))
    frozen.register(SimpleNamespace(catalog=catalog))
    library_search.register(SimpleNamespace(catalog=catalog), db)
    assert captured[0].model_schema() == captured[1].model_schema()


def test_evidence_bound_qa_run_registers_and_enables_the_library_and_readers(tmp_path):
    db = library_db(tmp_path / 'library.sqlite3')
    service = service_type()(tmp_path / 'state', model=ScriptedPort([reply('Alum raised IgG1 titers.')]),
                             project_root=PROJECT, **evidence_bound.service_options())
    try:
        out = evidence_bound.run(service, 'qa', 'Which isotype did alum raise?', allowed_roots=[], library_db=db)
    finally:
        service.close()
    config = out['session_config']
    assert out['status'] == 'succeeded' and out['content'] == 'Alum raised IgG1 titers.'
    assert {'library_search', *evidence_bound.READERS} <= set(config['enabled_tools']) <= set(config['allowed_tools'])
    assert config['allow_network'] is False and config['profile'] == evidence_bound.PROFILE
    assert config['instructions'].startswith(evidence_bound.ROUTING_V1)


def frozen_eval_runners():
    names = ('drb_fact_runner', 'drb_round5', 'drb_round5_fact', 'quality_round5_eval', 'round5_runner',
             'quality_advantage_round4_eval')
    if not all((ROOT / 'eval' / f'{name}.py').exists() for name in names):
        pytest.skip('frozen DeepResearch runners are not part of this checkout')


def test_research_report_settings_equal_the_round5_runners():
    frozen_eval_runners()
    from eval import drb_fact_runner, drb_round5, quality_round5_eval, round5_runner
    from eval import quality_advantage_round4_eval as r4
    for text in (round5_runner.SESSION_INSTRUCTIONS, drb_fact_runner.BASE_INSTRUCTIONS,
                 quality_round5_eval.SESSION_INSTRUCTIONS):
        assert text == research_report.SESSION_INSTRUCTIONS
    assert research_report.PROFILE == round5_runner.PROFILE == drb_fact_runner.PROFILE == quality_round5_eval.PROFILE
    assert research_report.DIRECT_SYSTEM == r4.DIRECT_SYSTEM and research_report.EXCLUDED_TOOLS == r4.EXCLUDED_TOOLS
    assert research_report.REPORT['budget'] == drb_round5.HARNESS_BUDGET == drb_fact_runner.HARNESS_BUDGET
    assert research_report.QUALITY['budget'] == quality_round5_eval.BUDGET
    assert (research_report.QUALITY['request_output_tokens'], research_report.QUALITY['request_timeout_seconds']) == \
        (quality_round5_eval.REQUEST_OUTPUT_TOKENS, quality_round5_eval.REQUEST_TIMEOUT)
    defaults = {a.arg: d.value for a, d in zip(top_level(ROOT / 'eval/round5_runner.py')['run_arm'].args.kwonlyargs,
                                                top_level(ROOT / 'eval/round5_runner.py')['run_arm'].args.kw_defaults)
                if isinstance(d, ast.Constant)}
    assert research_report.QUALITY['fallback_output_tokens'] == defaults['fallback_output']
    call = next(n for n in ast.walk(top_level(ROOT / 'eval/drb_round5.py')['generate'])
                if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'run_arm')
    kw = {k.arg: k.value.value for k in call.keywords if isinstance(k.value, ast.Constant)}
    assert (kw['request_output_tokens'], kw['request_timeout'], kw['fallback_output']) == \
        tuple(research_report.REPORT[k] for k in ('request_output_tokens', 'request_timeout_seconds', 'fallback_output_tokens'))
    case = {'level': 'L2', 'question': 'Which particle size favors lymph node drainage?'}
    assert research_report.report_goal(case['question'], 'L2') == drb_round5.goal(case, 'harness')
    assert drb_fact_runner.VARIANTS['fixed_bind'] == {'base': 'harness_fixed', 'post': 'bind'}


def test_research_report_run_drafts_delivers_and_binds_to_read_sources(tmp_path):
    report = 'Final report. See [a trial](https://example.org/unread) for details.\n\nMore reasoning here.'
    port = ScriptedPort([reply('Draft answer with general reasoning.'), reply(report),
                         reply('Final report. See a trial for details.\n\nMore reasoning here.')])
    pack, env = make_pack(tmp_path / 'pack'), tmp_path / 'offline.env'
    env.write_text('')
    out = research_report.run(port, 'Explain lymph node targeting.', tmp_path / 'run', project_root=PROJECT,
                              env_file=env, resource_pack=pack, allowed_roots=[tmp_path])
    assert out['delivery_source'] == 'agent' and out['status'] == 'succeeded'
    assert 'https://example.org/unread' not in out['content'] and out['binding']['bind_status'] == 'succeeded'
    assert (tmp_path / 'run/report.unbound.md').read_text() == report
    draft = json.loads((tmp_path / 'run/draft-stage.json').read_text())
    assert draft['draft'] == 'Draft answer with general reasoning.'
    quality = research_report.run(ScriptedPort([reply('Draft.'), reply('Short answer [x](https://example.org/y).')]),
                                  'Short question.', tmp_path / 'quality', settings=research_report.QUALITY,
                                  project_root=PROJECT, env_file=env, resource_pack=pack, allowed_roots=[tmp_path])
    assert 'binding' not in quality and quality['content'].endswith('(https://example.org/y).')


def test_citation_binding_is_the_frozen_fact_runner_and_scorer_code():
    frozen_eval_runners()
    runner, scorer = ROOT / 'eval/drb_fact_runner.py', ROOT / 'eval/drb_round5_fact.py'
    same_functions(ROOT / 'adj_partner/pipelines/citation_binding.py', runner,
                   ['canonical', 'words', 'chunks', 'evidence', 'unlink', 'LINK', 'STOP', 'EVIDENCE_CHARS',
                    'BIND_SYSTEM', 'BIND_RULES'], ('from eval.drb_round5_fact import', 'from .citations import'))
    same_functions(ROOT / 'adj_partner/pipelines/citations.py', scorer,
                   ['URL', 'LINK', 'BARE', 'SOURCE_CHARS', 'clean_url', 'urls_in', 'units', 'identifiers', 'trim',
                    '_texts', 'norm_url', 'read_sources', 'match'])
    from eval import drb_fact_runner
    assert (citation_binding.BIND_MAX_OUTPUT, citation_binding.BIND_TIMEOUT) == \
        (drb_fact_runner.BIND_MAX_OUTPUT, drb_fact_runner.BIND_TIMEOUT)
    post = ast.unparse(top_level(runner)['post_pass'])
    assert "len(text) >= 0.5 * len(report)" in post and "reply.finish_reason == 'stop'" in post


def read_run(tmp_path):
    body = {'abstract': 'Alum adjuvant raised antigen-specific IgG1 titers in BALB/c mice compared with antigen '
                        'alone; the effect required the NLRP3 inflammasome in this model. ' * 3}
    data = json.dumps(body).encode()
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / 'state/artifacts').mkdir(parents=True)
    (tmp_path / 'state/artifacts' / digest).write_bytes(data)
    (tmp_path / 'sources.json').write_text(json.dumps({'sources': [{
        'title': 'Alum study', 'artifact_hash': digest, 'kind': 'literature',
        'provenance': {'text_availability': 'abstract', 'url': 'https://pubmed.ncbi.nlm.nih.gov/123/', 'pmid': '123'}}]}))
    return tmp_path


def test_binding_keeps_read_links_and_removes_unread_ones(tmp_path):
    folder = read_run(tmp_path)
    report = ('Alum raised IgG1 titers [Alum study](https://pubmed.ncbi.nlm.nih.gov/123/).\n\n'
              'A second claim cites [unread](https://example.org/paper) and https://example.org/bare here.')
    text, record = citation_binding.bind(None, report, folder, mode='drop')
    assert '(https://pubmed.ncbi.nlm.nih.gov/123/)' in text and 'example.org' not in text
    assert record['evidence_sources'] == 1 and record['unread_links_before'] == 2
    text, record = citation_binding.bind(ScriptedPort([reply('too short')]), report, folder)
    assert record['bind_status'] == 'rejected' and 'example.org' not in text and 'pubmed.ncbi.nlm.nih.gov/123' in text
    assert record['bind_input']['EVIDENCE'][0]['url'] == 'https://pubmed.ncbi.nlm.nih.gov/123/'
    assert record['bind_input']['UNREAD_LINKS'] == ['https://example.org/bare', 'https://example.org/paper']
    with pytest.raises(ValueError):
        citation_binding.bind(None, report, folder, mode='strict')


def frozen_config(path, function, **names):
    node = next(n for n in ast.walk(top_level(path)[function]) if isinstance(n, ast.Dict)
                and any(isinstance(k, ast.Constant) and k.value == 'harness_config' for k in n.keys))
    value = node.values[[k.value for k in node.keys].index('harness_config')]
    return eval(compile(ast.Expression(value), str(path), 'eval'), {'str': str, **names})


def frozen_service_kwargs(path, function, **names):
    call = next(n for n in ast.walk(top_level(path)[function])
                if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'AgentService')
    return {k.arg: eval(compile(ast.Expression(k.value), str(path), 'eval'), names) for k in call.keywords
            if k.arg in ('request_output_tokens', 'request_timeout_seconds')}


def test_synth_tracks_equal_the_frozen_adjsynth_configuration(tmp_path):
    decision, computation = ROOT / 'eval/harness_capability_eval.py', ROOT / 'eval/harness_execution_eval.py'
    if not decision.exists() or not computation.exists():
        pytest.skip('AdjSynth runners are not part of this checkout')
    tokens = literal(decision, 'OUTPUT_TOKENS')
    assert synth.session_config('decision') == frozen_config(decision, 'prepare', OUTPUT_TOKENS=tokens)
    inputs = tmp_path / 'inputs'
    assert synth.session_config('computation', inputs) == frozen_config(computation, 'prepare', output=tmp_path)
    assert frozen_service_kwargs(decision, 'run_one', OUTPUT_TOKENS=tokens) == synth.service_options()
    assert frozen_service_kwargs(computation, 'one') == synth.service_options()
    with pytest.raises(ValueError):
        synth.session_config('computation')


def test_synth_decision_run_has_no_roots_and_no_network(tmp_path):
    service = service_type()(tmp_path / 'state', model=ScriptedPort([reply('{"decision":"A"}')]), project_root=PROJECT,
                             **synth.service_options())
    try:
        out = synth.run(service, 'decision', 'Choose one option.')
    finally:
        service.close()
    assert out['content'] == '{"decision":"A"}'
    assert out['session_config']['allowed_roots'] == [] and out['session_config']['allow_network'] is False


def test_guided_session_pins_the_frozen_plga_guidance(tmp_path):
    if PLGA.exists():
        assert literal(PLGA, 'ACTIVE_VERSION') == 'v2' and literal(PLGA, 'PROFILE') == guided.PROFILE
        text = literal(PLGA, 'GUIDANCE_VARIANTS')['v2']
        assert hashlib.sha256(text.encode()).hexdigest() == guided.FROZEN_GUIDANCE_SHA256['plga-structure-v2']
        assert literal(PLGA, 'BUDGET') == guided.BUDGET
        assert (literal(PLGA, 'REQUEST_OUTPUT_TOKENS'), literal(PLGA, 'REQUEST_TIMEOUT')) == \
            tuple(guided.service_options().values())
    service = service_type()(tmp_path / 'state', model=ScriptedPort([reply('Report.')]), project_root=PROJECT,
                             **guided.service_options())
    try:
        out = guided.run(service, 'Which structure gives two-phase release?', 'Compare internal architectures.')
    finally:
        service.close()
    config = service_type()(tmp_path / 'state', model=ScriptedPort([]), project_root=PROJECT).session(out['session_id'])['config']
    assert out['content'] == 'Report.' and config['instructions'].startswith('Compare internal architectures.')
    assert config['allow_network'] is True and config['allowed_roots'] == []


def public_case():
    graph = betaine(('c',) * 4)
    return {'name': 'betaine', 'question': 'Which route explains the betaine effect?', 'graph': graph, 'cards': cards('c')}


def test_mechanism_steps_are_the_frozen_p3_runner_code(tmp_path):
    if not RETRO.exists() or not DECISION.exists():
        pytest.skip('frozen P3 runner copies not present')
    ours = ROOT / 'adj_partner/pipelines/mechanism.py'
    same_functions(ours, RETRO, ['CONFIGS', 'ANSWER_SYSTEM', 'STOP', '_gather', '_raw_statuses'])
    same_functions(ours, DECISION, ['PREDICT_SYSTEM', 'ORDER_SYSTEM', '_json', 'payload', 'required_pairs',
                                    'predictions_map', 'mechanism_graph_nodes', 'checkpoint_at', 'library_live_and_order'])
    graph = public_case()['graph']
    old = frozen_functions(RETRO, ['host_query', 'answer_messages', 'STOP', 'ANSWER_SYSTEM'], re=re, gg=gg)
    edges = gg.targets(graph)
    assert [old['host_query']({'graph': graph}, e) for e in edges] == [mechanism.host_query(graph, e) for e in edges]
    items = [dict(edge='E1', status='supports', gate='', pmids=['1']), dict(edge='E2', status='not_addressed', gate='no_valid_citation', pmids=[])]
    evidence = {'E1': [dict(pmid='1', year='2019', title='T', text='Passage.')]}
    assert old['answer_messages']({'question': 'Q'}, 2020, edges, items, evidence) == \
        mechanism.answer_messages('Q', 2020, edges, items, evidence)
    stated = {'predictions': [dict(checkpoint_id='C1', hypothesis_id='H2', outcome='not_observed', reason='r')],
              'model_order': ['C4', 'C1']}
    private = ROOT / 'outputs/mechanism_graph_decision_20261112/dataset.py'
    if private.exists():
        assert literal(private, 'COST_BY_LEVEL') == mechanism.COST_BY_LEVEL
    dataset = SimpleNamespace(COST_BY_LEVEL=mechanism.COST_BY_LEVEL)
    old = frozen_functions(DECISION, ['costs_of', 'required_pairs', 'predictions_map', 'build_graphs'],
                           json=json, mg=mg, dataset=dataset)
    assert old['build_graphs'](public_case(), stated)[1] == mechanism.build_graph(public_case(), stated)
    calls, mine = [], []

    def call_json(messages, key, max_tokens):
        calls.append((messages, max_tokens))
        return ({'order': []} if len(calls) == 2 else {}), {}

    def complete(messages, limit):
        mine.append((messages, limit))
        return '{"order": []}' if len(mine) == 2 else '{}'

    old = frozen_functions(DECISION, ['ask_model', 'payload', 'required_pairs', 'costs_of', 'PREDICT_SYSTEM',
                                     'ORDER_SYSTEM'], json=json, mg=mg,
                           dataset=dataset, OUT=tmp_path, KEY='k', call_json=call_json,
                           g1=SimpleNamespace(dump=lambda *a: None))
    old['ask_model'](public_case())
    mechanism.elicit(complete, public_case())
    assert len(calls) == 2 and calls == mine


def test_mechanism_grounding_filters_post_cutoff_passages_and_answers_from_gated_edges():
    graph = betaine()
    passages = [dict(pmid='1', year='2019', title='BHMT activity', section='Results', text='BHMT activity rose.'),
                dict(pmid='2', year='2021', title='Later study', section='Results', text='Later finding.')]
    retriever = SimpleNamespace(gather=lambda claim, queries, **kw: {'passages': passages if queries else []})
    edges = gg.targets(graph)
    replies = [json.dumps({'edges': [{'edge': 'E1', 'queries': ['betaine BHMT activity']}]}),
               json.dumps({'edges': [{'edge': 'E1', 'status': 'supports', 'passages': ['E1.P1']},
                                     {'edge': 'E2', 'status': 'supports', 'passages': ['E2.P1']}]}),
               'BHMT activity is the supported route [E1.P1].']
    seen = []
    complete = lambda messages, limit: seen.append(limit) or replies[len(seen) - 1]
    grounded = mechanism.ground(complete, 'Why does betaine help?', graph, 2020, retriever=retriever)
    assert [p['pmid'] for p in grounded['evidence']['E1']] == ['1'] and grounded['evidence']['E2'] == []
    assert [(i['edge'], i['status']) for i in grounded['edges'][:2]] == [('E1', 'supports'), ('E2', 'not_addressed')]
    assert grounded['queries']['E1'] == ['betaine BHMT activity'] and len(grounded['edges']) == len(edges)
    assert mechanism.answer(complete, 'Why does betaine help?', grounded).startswith('BHMT activity')
    assert seen == [mechanism.QUERY_TOKENS, mechanism.JUDGE_TOKENS, mechanism.ANSWER_TOKENS]


def test_mechanism_planning_replans_until_one_hypothesis_is_live():
    case = public_case()
    stated = {'predictions': [dict(checkpoint_id='C1', hypothesis_id='H2', outcome='not_observed', reason='r'),
                              dict(checkpoint_id='C2', hypothesis_id='H2', outcome='not_observed', reason='r'),
                              dict(checkpoint_id='C4', hypothesis_id='H1', outcome='no_prediction', reason='r')],
              'model_order': []}
    graph = mechanism.build_graph(case, stated)
    assert {m['checkpoint_id']: m['cost'] for m in graph['checkpoint_nodes']} == \
        {'C1': 'medium', 'C2': 'high', 'C3': 'medium', 'C4': 'low'}
    readouts, statuses = {'C1': 'opposes', 'C2': 'opposes', 'C4': 'supports'}, {}
    step = mechanism.plan(graph, case['cards'], statuses)
    while step['next']:
        statuses[step['next']] = readouts[step['next']]
        step = mechanism.plan(graph, case['cards'], statuses)
    assert step['live'] == ['H2'] and mechanism.settle(case, graph, statuses) == 'H2'
