"""Grounded revision QA: independent draft, model-written queries, host retrieval, revise-or-keep.

Query and revision instructions match the adjuvant QA campaign (outputs/adjuvant_qa_grounded_20261103);
its English-only answer preamble is replaced by ANSWER_PREAMBLE so answers follow the question's language.
"""
import json
import re

from ..draft_first import abstains

WORKFLOW_VERSION = 'qa-grounded-revision-v1'
# v2: hybrid PubMed + Scholar/web retrieval, then an anchor self-check of whichever answer would be delivered.
WORKFLOW_V2 = 'qa-grounded-revision-v2'
KEEP = 'KEEP_DRAFT'
QUERY_SYSTEM = 'You help locate the published study behind a scientific question. Reply with JSON only.'
QUERY_USER = '''Question: {question}

Return {{"specific_study": true or false, "queries": [...]}}.
specific_study is true if the question refers to particular study results, a named formulation, experiment,
figure or trial (for example "this study", "the authors", specific materials or measurements), and false if it
asks for general textbook knowledge.
If true, give up to 3 PubMed queries of 3-6 keywords each, most specific first, using words likely to appear in
the title or abstract: full names next to abbreviations, materials, adjuvants, antigens, disease or animal model.
No quotation marks, boolean operators or field tags. If false, return an empty list.'''
REVISE_SYSTEM = ('You are a scientific assistant specializing in immunology and vaccine adjuvants. '
                 'Distinguish evidence, inference and uncertainty. Answer the original question directly.')
ANSWER_PREAMBLE = ('Answer this scientific question concisely in the language of the question. If a requested '
                   'study or result cannot be identified, state the uncertainty. Do not invent measured outcomes '
                   'or sources. Discuss biological findings without step-by-step laboratory instructions.\n\n')
REVISE_USER = '''{preamble}{question}

Your earlier draft answer:
<draft>
{draft}
</draft>

Passages retrieved automatically from PubMed and PubMed Central by keyword search. They may or may not come
from the study the question refers to:
{passages}

Decide whether any passage comes from the study the question refers to (same materials, formulation, model
and design) or directly answers the question.
- If none does, or they add nothing to the draft, reply with exactly {keep} and nothing else.
- Otherwise write the final answer: keep the draft's correct content and structure, correct or add
  study-specific findings supported by the matching passages and cite them as [P#]. Do not present findings
  from other studies as this study's results. Give the answer first; do not open with what could not be
  found. Keep a similar length to the draft.'''


REFLECT_SYSTEM = ('You are an expert in immunology and vaccine adjuvant science reviewing an answer before it is '
                  'sent. Reply with the final answer only.')
REFLECT_USER = '''Question:
{question}

Current answer:
<answer>
{answer}
</answer>

Check the answer against what this specific question asks, then output the improved final answer.
1. Name the question's core entity (adjuvant, formulation, molecule, cell, model), the measured endpoint or
   outcome, the comparator or control, the mechanism linking them, and the conclusion it asks for.
2. Make sure the answer states each of these explicitly and directly: lead with the conclusion, then the
   specific endpoint and comparison (direction of effect, and any values already in the answer), then the
   mechanism that explains it. Add a missing comparator or mechanistic link if it follows from well-established
   immunology; mark inference as such.
3. Remove generic background, assay checklists, safety or clinical boilerplate, and extrapolations across
   species, routes or disease models that the question does not ask about. Do not add numbers, study details or
   citations that are not already in the answer.
4. Keep any [P#] citations attached to the claims they support. Same language as the question, 150-350 words,
   plain prose or a short list. Do not describe this checking process.'''


def reflect_messages(question, answer):
    return [{'role': 'system', 'content': REFLECT_SYSTEM},
            {'role': 'user', 'content': REFLECT_USER.format(question=question, answer=answer)}]


def decide_reflection(content, answer):
    """Return (reflection_status, answer); an empty or newly abstaining rewrite keeps the answer."""
    text = (content or '').strip()
    if not text or text == answer.strip():
        return 'unchanged', answer
    if abstains(text) and not abstains(answer):
        return 'rejected_abstention', answer
    return 'applied', text


def query_messages(question):
    return [{'role': 'system', 'content': QUERY_SYSTEM},
            {'role': 'user', 'content': QUERY_USER.format(question=question)}]


def parse_queries(text):
    """Malformed output means no retrieval, never a failed answer."""
    match = re.search(r'\{.*\}', text or '', re.S)
    try:
        value = json.loads(match.group(0)) if match else {}
    except ValueError:
        value = {}
    if not isinstance(value, dict):
        value = {}
    queries = [re.sub(r'["()\[\]]', ' ', q).strip()[:200] for q in value.get('queries') or [] if isinstance(q, str)]
    queries = [q for q in queries if q][:3]
    return value.get('specific_study') is True and bool(queries), queries


def passage_block(passages):
    return '\n\n'.join(f"[P{i}] {p['title']} ({p['year']}), {p['section'] or 'Body'}: {p['text']}"
                       for i, p in enumerate(passages, 1))


def revise_messages(question, draft, passages):
    return [{'role': 'system', 'content': REVISE_SYSTEM},
            {'role': 'user', 'content': REVISE_USER.format(preamble=ANSWER_PREAMBLE, question=question, draft=draft,
                                                           passages=passage_block(passages), keep=KEEP)}]


def decide(content, draft):
    """Return (enhancement_status, answer). An opening abstention never replaces an answering draft."""
    text = (content or '').strip()
    if not text or text.startswith(KEEP):
        return 'unchanged', draft
    if abstains(text) and not abstains(draft):
        return 'rejected_abstention', draft
    return 'applied', text


def cited(answer, passages):
    """Passages the delivered answer cites as [P#], in order."""
    numbers = {int(n) for group in re.findall(r'\[P(\d+(?:\s*,\s*P?\d+)*)\]', answer)
               for n in re.findall(r'\d+', group)}
    return [passages[n - 1] for n in sorted(numbers) if 0 < n <= len(passages)]
