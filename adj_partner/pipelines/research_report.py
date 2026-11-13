"""F3 research-report agent and F4 quality agent: tool-free draft, evidence-first session, answer-preserving delivery.

The solver first answers without tools; the draft and the remaining shared budget seed an evidence-first session
with web search, page reading and evidence tools. A run that ends without a delivery returns the draft, else one
tool-free call. F3 (``REPORT``) then applies ``citation_binding.bind``; F4 (``QUALITY``) is the short-answer setting.
"""
import json
from pathlib import Path
import time

from .sessions import delivered

PROFILE = 'research-evidence-first'
SESSION_INSTRUCTIONS = ('Host capability status: web_search (Google results via the host gateway) and web_page_read '
                        '(rendered public pages) are available. Search first for unknown sources, then read the '
                        'returned source_id; do not guess URLs. A failed read is an evidence gap; do not retry it.')
DIRECT_SYSTEM = 'You are a careful research assistant. Answer the question accurately; distinguish observations, proposals, and uncertainty. Do not invent results or citations.'
EXCLUDED_TOOLS = frozenset({
    'bash', 'ask_user', 'deg_analyze', 'deg_explore', 'analysis_history',
    'analysis_history_read', 'analysis_method_register', 'analysis_method_run',
    'research_state', 'research_critic', 'evidence_select', 'evidence_reselect',
    'claim_record', 'claim_revise', 'claim_observe', 'claim_assess',
    'report_create', 'report_validate', 'design_record_read',
    'experiment_state', 'experiment_design_revise',
})
HARNESS_SUFFIX = ('Search for and read relevant primary literature yourself. Compare sources, distinguish source text from '
                  'inference, record limitations, and deliver one complete Markdown research report. Cite only sources you '
                  'actually read, with clickable URLs. ')
TAIL = 'Do not assume a particular paper, identifier, answer, section count, report length, or minimum citation count.'

REPORT = dict(name='research-report', budget={'max_steps': 16, 'max_input_tokens': 150000, 'max_output_tokens': 16000,
                                              'wall_time_seconds': 900},
              request_output_tokens=6000, request_timeout_seconds=240, fallback_output_tokens=12000, binding='bind')
QUALITY = dict(name='research-report-quality', budget={'max_steps': 12, 'max_input_tokens': 80000,
                                                       'max_output_tokens': 12000, 'wall_time_seconds': 480},
               request_output_tokens=3200, request_timeout_seconds=150, fallback_output_tokens=6000, binding=None)


def report_goal(question, level):
    """AdjDeepResearch harness goal; ``level`` is the benchmark difficulty label."""
    return f'Independently research and answer the following question. Difficulty label: {level}.\n\n{question}\n\n{HARNESS_SUFFIX}{TAIL}'


def _dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run(port, goal, folder, *, settings=REPORT, project_root, env_file='.env', resource_pack=None, allowed_roots=None,
        direct_system=DIRECT_SYSTEM):
    """One run in ``folder`` (state/, draft-stage.json, sources.json, report.md); ``port`` is the solver model."""
    from ..answer_preserving import (ANSWER_PRESERVING_GUIDANCE, DraftPreparationError, fallback_delivery,
                                     prepare_draft)
    from ..application import AgentService
    project_root, folder = Path(project_root), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    resource_pack = resource_pack or project_root / 'data/adjuvant_resources_v4'
    allowed_roots = allowed_roots or [project_root / 'data/research_public_inputs']
    budget = settings['budget']
    start = time.monotonic()
    deadline = start + budget['wall_time_seconds']
    result = {'pipeline': settings['name'], 'status': 'failed', 'content': '', 'delivery_source': None}
    service = sid = rid = None
    draft_text = None
    try:
        run_budget, session_goal, instructions = dict(budget), goal, SESSION_INSTRUCTIONS
        try:
            draft = prepare_draft(port, goal, run_budget, system=direct_system)
            _dump(folder / 'draft-stage.json', draft)
            draft_text, session_goal, run_budget = draft['draft'], draft['goal'], draft['remaining_budget']
            instructions += '\n' + ANSWER_PRESERVING_GUIDANCE
        except DraftPreparationError as exc:
            _dump(folder / 'draft-stage.json', {**exc.record, 'code': exc.code, 'fallback': 'plain_goal'})
            if exc.record.get('attempted') and exc.record.get('remaining_budget'):
                run_budget = exc.record['remaining_budget']
            result['draft_failure'] = exc.code
        service = AgentService(folder / 'state', project_root=project_root, env_file=env_file, model=port,
                               resource_pack=resource_pack, request_output_tokens=settings['request_output_tokens'],
                               request_timeout_seconds=settings['request_timeout_seconds'])
        allowed = sorted(set(service._benign_tools) - EXCLUDED_TOOLS)
        run_budget['wall_time_seconds'] = max(30, deadline - time.monotonic() - 60)
        for key in ('max_steps', 'max_input_tokens', 'max_output_tokens'):
            run_budget[key] = max(1, int(run_budget[key]))
        session = service.create_session(profile=PROFILE, instructions=instructions, allow_network=True,
                                         allowed_roots=[str(r) for r in allowed_roots], allowed_tools=allowed,
                                         budget=run_budget)
        sid = session['id']
        rid = service.submit(sid, session_goal)['id']
        final = service.drive(rid)
        _dump(folder / 'runtime_result.json', final)
        content = delivered(final)
        result.update(status=final['status'], content=content, delivery_source='agent' if content else None,
                      stop_reason=final.get('state', {}).get('stop_reason'))
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error_code=str(getattr(exc, 'code', 'unknown')))
    finally:
        if service and rid:
            try:
                _dump(folder / 'actions.json', service.actions(rid))
                _dump(folder / 'sources.json', {'sources': service.research.list_sources(sid, rid)})
            except Exception as exc:
                result['export_error'] = type(exc).__name__
        if service:
            service.close()
    if not result['content'].strip():
        remaining = deadline - time.monotonic() + 60
        content, source = fallback_delivery(port, goal, system=direct_system, draft=draft_text,
                                            max_output_tokens=settings['fallback_output_tokens'],
                                            timeout_seconds=min(240, max(0, remaining)))
        result.update(content=content, delivery_source=source, status='succeeded' if content else result['status'])
    if result['content'] and settings['binding']:
        from .citation_binding import bind
        (folder / 'report.unbound.md').write_text(result['content'])
        result['content'], result['binding'] = bind(port, result['content'], folder, mode=settings['binding'])
    if result['content']:
        (folder / 'report.md').write_text(result['content'])
    result.update(elapsed_seconds=time.monotonic() - start, delivered=bool(result['content'].strip()))
    return result


SETTINGS = ('report', 'quality')
SPEC = dict(profile=PROFILE, report=REPORT, quality=QUALITY, excluded_tools=sorted(EXCLUDED_TOOLS),
            texts=dict(SESSION_INSTRUCTIONS=SESSION_INSTRUCTIONS, DIRECT_SYSTEM=DIRECT_SYSTEM,
                       HARNESS_SUFFIX=HARNESS_SUFFIX, TAIL=TAIL))
