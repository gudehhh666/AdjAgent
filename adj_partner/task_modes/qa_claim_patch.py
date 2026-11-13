"""Atomic evidence-quoted edits to an independent draft; no model calls.

Literal quote validation establishes source membership, not scientific truth or
entailment. Invalid proposals raise ValueError; callers must log any fallback.
"""
import json
from pathlib import Path

from ..profiles import load_trusted_guidance


PATCH_SKILL = Path(__file__).resolve().parents[2] / 'config/skills/adjuvant-qa-claim-patch/SKILL.md'
EDIT_FIELDS = {'before', 'after', 'source_url', 'quote'}


def patch_messages(question, draft, packet):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Patch question must be nonempty text')
    if not isinstance(draft, str) or not draft.strip():
        raise ValueError('Patch draft must be nonempty text')
    if not isinstance(packet, list):
        raise ValueError('Patch evidence must be a list')
    guidance = load_trusted_guidance(PATCH_SKILL).split('---', 2)[2].strip()
    return [{'role':'system', 'content':guidance},
            {'role':'user', 'content':json.dumps({'question':question, 'draft':draft, 'evidence':packet}, ensure_ascii=False)}]


def apply_patch(draft, proposal, packet):
    """Validate against original inputs, then apply every edit or reject all."""
    if not isinstance(draft, str) or not draft.strip():
        raise ValueError('Patch draft must be nonempty text')
    if not isinstance(proposal, dict) or set(proposal) != {'edits'}:
        raise ValueError('Patch proposal must contain only edits')
    edits = proposal['edits']
    if not isinstance(edits, list) or len(edits) > 4:
        raise ValueError('Patch edits must be a list of at most four items')
    if not edits:
        return {'answer':draft, 'edits':[], 'used_sources':[]}
    if not isinstance(packet, list):
        raise ValueError('Patch evidence must be a list')
    evidence = {}
    for source in packet:
        if not isinstance(source, dict) or not isinstance(source.get('url'), str):
            continue
        excerpts = source.get('excerpts', [])
        if not isinstance(excerpts, list):
            continue
        evidence.setdefault(source['url'], []).extend(
            excerpt['text'] for excerpt in excerpts
            if isinstance(excerpt, dict) and isinstance(excerpt.get('text'), str))
    replacements, additions, verified, used = [], [], [], []
    for index, item in enumerate(edits):
        prefix = f'Invalid patch edit {index}: '
        if not isinstance(item, dict) or set(item) != EDIT_FIELDS:
            raise ValueError(prefix+'expected before, after, source_url and quote only')
        if any(not isinstance(item[key], str) for key in EDIT_FIELDS):
            raise ValueError(prefix+'all fields must be text')
        before, after, url, quote = (item[key] for key in ('before', 'after', 'source_url', 'quote'))
        if len(before) > 600:
            raise ValueError(prefix+'before exceeds 600 characters')
        if not after.strip() or len(after) > 800:
            raise ValueError(prefix+'after must contain 1–800 characters of nonblank text')
        if not 20 <= len(quote) <= 1200 or not quote.strip():
            raise ValueError(prefix+'quote must contain 20–1200 characters')
        if not url or url not in evidence:
            raise ValueError(prefix+'source_url is not in the evidence packet')
        if not any(quote in excerpt for excerpt in evidence[url]):
            raise ValueError(prefix+'quote is not verbatim within one excerpt at source_url')
        if before:
            start = draft.find(before)
            if start < 0 or draft.find(before, start+1) >= 0:
                raise ValueError(prefix+'before must occur exactly once in the original draft')
            replacements.append((start, start+len(before), after))
        else:
            additions.append(after)
        verified.append(dict(item))
        if url not in used:
            used.append(url)
    ordered = sorted(replacements)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError('Invalid patch proposal: edits overlap in the original draft')
    answer = draft
    for start, end, after in reversed(ordered):
        answer = answer[:start]+after+answer[end:]
    for addition in additions:
        answer += '\n\n'+addition
    answer += '\n\nSources: '+', '.join(used)
    return {'answer':answer, 'edits':verified, 'used_sources':used}
