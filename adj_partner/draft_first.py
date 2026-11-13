"""Draft-first QA on one shared budget: tool-free draft, targeted verification, host fallback.

The verification run may only revise the draft where a read source contradicts it.
A final answer that opens by abstaining while the draft answered is replaced by
the draft; so is a verification run that fails to deliver. Every outcome records
which text was delivered and why, so fallbacks are never reported as agent output.
"""
import re

from .answer_preserving import ANSWER_PRESERVING_GUIDANCE, DraftPreparationError, prepare_draft

RETRIEVAL_MISS_GUIDANCE = '''Not locating the study named or implied by the task is not evidence against the
draft. Keep the draft's substantive conclusions, including directions of effect,
unless a source you actually read contradicts them; then revise only that claim
and state the contradicting observation. Do not open the answer with what could
not be identified, located or confirmed; give the answer first and bound its
uncertainty afterwards.'''

OPENING_CHARS = 300
ABSTENTION = re.compile(
    r"(can[’']?t|cannot|could not|couldn[’']t|unable to|does not|do not) "
    r"(identify|confirm|determine|establish|verify|assess|reliably)"
    r"|no defensible|please provide|isn[’']t identifiable|not identifiable"
    r"|(evidence|information|record) (does not|doesn[’']t) (identify|establish|show)", re.I)


def abstains(text):
    """Whether an answer opens by declining to conclude; later caveats do not count."""
    return not text.strip() or bool(ABSTENTION.search(text[:OPENING_CHARS]))


def _run(service, goal, session_options, budget):
    session = service.create_session(budget=budget, **session_options)
    run_id = service.submit(session['id'], goal)['id']
    result = service.drive(run_id)
    answer = result.get('state', {}).get('delivery', {}).get('content', '') or ''
    return {'run_id': run_id, 'status': result['status'], 'answer': answer,
            'delivered': result['status'] == 'succeeded' and bool(answer.strip()),
            'error': (result.get('state', {}).get('error') or {}).get('code')}


def run_draft_first(service, model, goal, *, system, budget, session_options=None):
    """Return the delivered answer and its provenance.

    ``model`` must be the port ``service`` uses, so both stages share accounting.
    ``system`` is the tool-free draft system prompt (normally the direct baseline's).
    ``session_options`` are passed to ``create_session`` except ``budget``.
    ``source`` is one of agent, agent_without_draft, draft_after_abstention,
    draft_after_agent_failure or none.
    """
    options = dict(session_options or {})
    instructions = options.pop('instructions', '')
    try:
        draft = prepare_draft(model, goal, budget, system=system)
    except DraftPreparationError as exc:
        outcome = {'draft_error': exc.code, 'draft': exc.record}
        remaining = exc.record.get('remaining_budget')
        if not remaining or any(value <= 0 for value in remaining.values()):
            return dict(outcome, source='none', answer='', delivered=False, agent=None)
        agent = _run(service, goal, dict(options, instructions=instructions), remaining)
        return dict(outcome, source='agent_without_draft', answer=agent['answer'],
                    delivered=agent['delivered'], agent=agent)
    guided = '\n\n'.join(part for part in (instructions, ANSWER_PRESERVING_GUIDANCE, RETRIEVAL_MISS_GUIDANCE) if part)
    agent = _run(service, draft['goal'], dict(options, instructions=guided), draft['remaining_budget'])
    if not agent['delivered']:
        source, answer = 'draft_after_agent_failure', draft['draft']
    elif abstains(agent['answer']) and not abstains(draft['draft']):
        source, answer = 'draft_after_abstention', draft['draft']
    else:
        source, answer = 'agent', agent['answer']
    return {'source': source, 'answer': answer, 'delivered': bool(answer.strip()),
            'draft': draft, 'agent': agent}
