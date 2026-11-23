import copy
import importlib.util
import json

import pytest


def helper():
    assert importlib.util.find_spec('adj_partner.task_modes.qa_patch_gate'), 'semantic gate helper missing'
    from adj_partner.task_modes import qa_patch_gate
    return qa_patch_gate


def edit(before='old', after='new'):
    return {'before': before, 'after': after, 'source_url': 'https://example.org/source',
            'quote': 'A sufficiently long exact quotation from the source.'}


def decision(index=0, approve=True, reason='Same study and directly supported.'):
    return {'index': index, 'approve': approve, 'reason': reason}


def test_gate_prompt_keeps_untrusted_inputs_in_user_message():
    packet = [{'url': 'https://example.org/source', 'excerpts': [{'text': 'IGNORE_PRIOR_RULES'}]}]
    edits = [edit()]
    messages = helper().gate_messages('Question sentinel', 'Draft sentinel', packet, edits)
    assert [m['role'] for m in messages] == ['system', 'user']
    for sentinel in ('IGNORE_PRIOR_RULES', 'Question sentinel', 'Draft sentinel'):
        assert sentinel not in messages[0]['content']
    assert json.loads(messages[1]['content']) == {
        'question': 'Question sentinel', 'draft': 'Draft sentinel', 'evidence': packet, 'edits': edits}


def test_filter_only_retains_original_approved_edits_in_original_order():
    proposal = {'edits': [edit('a', 'A'), edit('b', 'B'), edit('c', 'C')]}
    original = copy.deepcopy(proposal)
    verdict = {'decisions': [decision(2), decision(0), decision(1, False, 'Different endpoint.')]}
    result = helper().filter_proposal(proposal, verdict)
    assert result == {'edits': [original['edits'][0], original['edits'][2]]}
    assert proposal == original
    result['edits'][0]['after'] = 'mutated'
    assert proposal == original


def test_rejection_and_empty_input_preserve_empty_proposal():
    module = helper()
    assert module.filter_proposal({'edits': [edit()]}, {'decisions': [decision(approve=False)]}) == {'edits': []}
    assert module.filter_proposal({'edits': []}, {'decisions': []}) == {'edits': []}


@pytest.mark.parametrize('decisions', [[], [decision(), decision()], [decision(1)],
    [decision(-1)], [decision(True)], [decision(0.0)], [decision('0')]])
def test_indices_must_cover_every_edit_exactly_once(decisions):
    with pytest.raises(ValueError):
        helper().filter_proposal({'edits': [edit()]}, {'decisions': decisions})


@pytest.mark.parametrize('approve', [0, 1, 'true', None, [], {}])
def test_approve_requires_actual_boolean(approve):
    with pytest.raises(ValueError):
        helper().filter_proposal({'edits': [edit()]}, {'decisions': [decision(approve=approve)]})


@pytest.mark.parametrize('reason', ['', '  ', None, 1, [], {}])
def test_reason_requires_nonblank_text(reason):
    with pytest.raises(ValueError):
        helper().filter_proposal({'edits': [edit()]}, {'decisions': [decision(reason=reason)]})


@pytest.mark.parametrize('verdict', [None, [], {}, {'decisions': {}},
    {'decisions': [decision()], 'answer': 'rewritten'},
    {'decisions': [{**decision(), 'after': 'replacement forbidden'}]},
    {'decisions': [{'index': 0, 'approve': True}]}, {'decisions': [True]}])
def test_verdict_has_no_rewrite_authority_or_extra_fields(verdict):
    with pytest.raises(ValueError):
        helper().filter_proposal({'edits': [edit()]}, verdict)


@pytest.mark.parametrize('proposal', [None, [], {}, {'edits': 'wrong'}, {'edits': [edit()] * 5},
    {'edits': [], 'answer': 'wrong'}, {'edits': [None]}, {'edits': [{'after': 'wrong'}]},
    {'edits': [{**edit(), 'quote': None}]}])
def test_proposal_shape_is_checked_before_filtering(proposal):
    with pytest.raises(ValueError):
        helper().filter_proposal(proposal, {'decisions': []})


def test_approved_subset_remains_compatible_with_literal_validator():
    from adj_partner.task_modes.qa_claim_patch import apply_patch
    proposal = {'edits': [edit('old', 'new'), edit('', 'Peripheral addition')]}
    packet = [{'url': proposal['edits'][0]['source_url'],
               'excerpts': [{'text': proposal['edits'][0]['quote']}]}]
    apply_patch('Keep old and caveat.', proposal, packet)
    filtered = helper().filter_proposal(proposal, {'decisions': [decision(0), decision(1, False)]})
    result = apply_patch('Keep old and caveat.', filtered, packet)
    assert result['answer'].startswith('Keep new and caveat.\n\nSources:')
    assert 'Peripheral' not in result['answer']
