"""Opt-in QA synthesis over bounded question-selected evidence.

Live retrieval under this policy is an unvalidated integration, not the cached
evidence replay used by the policy experiment. It performs no critic review.
"""
import hashlib
import json
import re

from ..research.qa_review_context import select_evidence
from . import qa


WORKFLOW_VERSION = 'qa-question-first-v1'
ANSWER_SKILL_SHA256 = '5630ccb1b414657f3b23ab1ffb1bd598e1e1528faf7fff30d77c9c72077b2a4c'
# Keep irrelevant computation/design schemas out of every retrieval request.
# This is an authorization allowlist as well as the initial enabled inventory:
# tools_enable cannot silently restore the full domain profile mid-run.
RETRIEVAL_TOOLS = frozenset({'literature_search', 'web_search', 'article_resolve',
                           'fulltext_fetch', 'document_read', 'artifact_read', 'job_status'})
INSTRUCTIONS = '''Gather a bounded evidence brief for the public scientific question. Identify the requested entities, study, endpoint and comparison before using sources. Use targeted primary-source queries only where the question needs source-specific facts; there is no minimum query count. Do not silently substitute a different study. Distinguish general knowledge from specific reported observations. Read the relevant results or figure captions when available, then stop once the essential evidence or its specific gap is clear. Do not run a critic or write a research report. Return a brief of available evidence; the final answer is synthesized separately from verified source records.'''


def evidence_packet(question, documents):
    packet = select_evidence(question, '', documents, budget=16000, per_source=6000)
    lookup = {source.get('id', str(i)): source for i, source in enumerate(documents)}
    for item in packet:
        source = lookup.get(item['source_id'], {})
        document = source.get('document', {})
        item['title'] = source.get('title') or (document.get('title', '') if isinstance(document, dict) else '')
    # Titles are added after excerpt selection, as in the frozen policy replay.
    # Reject an oversized packet instead of silently changing selected evidence.
    if len(json.dumps(packet, ensure_ascii=False)) > 24000:
        raise ValueError('QA evidence packet exceeds budget after source titles')
    return packet


def answer_messages(question, packet):
    # TaskSpec hashes Python implementation; pin the external skill as well.
    if hashlib.sha256(qa.ANSWER_SKILL.read_bytes()).hexdigest() != ANSWER_SKILL_SHA256:
        raise ValueError('Question-first skill changed; create a newly versioned workflow')
    return qa.answer_messages(question, packet=packet)


def _url_token(token):
    token = token.rstrip('.,;!?')
    while token.endswith(')') and token.count(')') > token.count('('):
        token = token[:-1]
    return token


def parse_answer(content, packet):
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Question-first answer must be nonempty plain text')
    allowed = {item['url'] for item in packet if isinstance(item.get('url'), str) and item['url']}
    urls = {_url_token(token) for token in re.findall(r'(?:[A-Za-z][A-Za-z0-9+.-]*://|(?<![\w:])//|www\.|\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}/)[^\s<>\"\']+', content)}
    # A link target without a scheme must not evade the source allowlist.
    for target in re.findall(r'\]\(\s*<?([^\s<>]+)', content):
        urls.add(_url_token(target))
    for target in re.findall(r'^\s*\[[^\]]+\]:\s*<?([^\s<>]+)', content, re.MULTILINE):
        urls.add(_url_token(target))
    for _, target in re.findall(r'\b(?:href|src)\s*=\s*([\"\'])(.*?)\1', content, re.IGNORECASE):
        urls.add(target)
    if not urls <= allowed:
        raise ValueError('Answer cited a URL outside the question-first evidence packet')
    return {'answer': content, 'sources': sorted(urls), 'review': '', 'uncertainty': ''}
