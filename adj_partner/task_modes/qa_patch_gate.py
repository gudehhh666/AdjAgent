"""Semantic veto prompts and strict filtering; never rewrite proposed edits.

Callers must first validate the proposal using qa_claim_patch.apply_patch and
retain its original draft/packet. This gate validates the decision contract,
not the scientific correctness of a model's semantic judgment.
"""
import copy
import json
from pathlib import Path

from ..profiles import load_trusted_guidance


GATE_SKILL = Path(__file__).resolve().parents[2] / 'config/skills/adjuvant-qa-patch-gate/SKILL.md'
EDIT_FIELDS = {'before', 'after', 'source_url', 'quote'}
DECISION_FIELDS = {'index', 'approve', 'reason'}


def _edits(proposal):
    if not isinstance(proposal, dict) or set(proposal) != {'edits'}:
        raise ValueError('Gate proposal must contain only edits')
    edits = proposal['edits']
    if not isinstance(edits, list) or len(edits) > 4:
        raise ValueError('Gate edits must be a list of at most four items')
    for item in edits:
        if not isinstance(item, dict) or set(item) != EDIT_FIELDS:
            raise ValueError('Gate edit must contain before, after, source_url and quote only')
        if any(not isinstance(item[key], str) for key in EDIT_FIELDS):
            raise ValueError('Gate edit fields must be text')
    return edits


def gate_messages(question, draft, packet, edits):
    """Request one veto decision per already literally validated edit."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Gate question must be nonempty text')
    if not isinstance(draft, str) or not draft.strip():
        raise ValueError('Gate draft must be nonempty text')
    if not isinstance(packet, list):
        raise ValueError('Gate evidence must be a list')
    _edits({'edits': edits})
    guidance = load_trusted_guidance(GATE_SKILL).split('---', 2)[2].strip()
    return [{'role': 'system', 'content': guidance},
            {'role': 'user', 'content': json.dumps(
                {'question': question, 'draft': draft, 'evidence': packet, 'edits': edits},
                ensure_ascii=False)}]


def filter_proposal(proposal, verdict):
    """Reject malformed verdicts atomically; copy only approved original edits."""
    edits = _edits(proposal)
    if not isinstance(verdict, dict) or set(verdict) != {'decisions'}:
        raise ValueError('Gate verdict must contain only decisions')
    decisions = verdict['decisions']
    if not isinstance(decisions, list):
        raise ValueError('Gate decisions must be a list')
    approved = {}
    for item in decisions:
        if not isinstance(item, dict) or set(item) != DECISION_FIELDS:
            raise ValueError('Gate decision must contain index, approve and reason only')
        index = item['index']
        if type(index) is not int or not 0 <= index < len(edits):
            raise ValueError('Gate index must be an integer within the original edits')
        if index in approved:
            raise ValueError('Gate index must occur exactly once')
        if type(item['approve']) is not bool:
            raise ValueError('Gate approve must be a boolean')
        if not isinstance(item['reason'], str) or not item['reason'].strip():
            raise ValueError('Gate reason must be nonempty text')
        approved[index] = item['approve']
    if set(approved) != set(range(len(edits))):
        raise ValueError('Gate decisions must cover every original edit exactly once')
    return {'edits': [copy.deepcopy(item) for index, item in enumerate(edits) if approved[index]]}
