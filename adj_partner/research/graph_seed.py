"""Host-built candidate skeleton for a mechanism graph from the installed resource pack.

The pack's mechanism families are author-reported literature-matrix labels, not
experimental or causal links. The seed only lists candidate competing explanations
the model should consider; it never becomes a graph node or edge by itself.
"""
import re
from urllib.parse import unquote

DATASET = 'vaxjo_matrix'
KIND = 'extracted_mechanism'
PREDICATE = 'has_reported_mechanism_family'
FAMILY_PREFIX = 'vaxjo_matrix:family:'
MAX_ROWS = 3


def _words(value):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'+-]*", value.casefold()) if len(w) >= 3}


def build(pack, question):
    """Return {'rows': [...], 'families': [...], 'scope': str} or None when nothing matches.

    A row is kept only when every name word (3+ characters) of its adjuvant appears
    in the question, so partial matches on generic words do not seed unrelated rows.
    """
    if pack is None or DATASET not in pack.datasets:
        return None
    try:
        found = pack.search(question[:500], dataset=DATASET, kind=KIND, limit=20, match_mode='ranked')
    except ValueError:
        return None
    asked = _words(question)
    named = [r for r in found['results'] if _words(r['title']) and _words(r['title']) <= asked][:MAX_ROWS]
    rows, families = [], {}
    for record in named:
        names = []
        for edge in pack.neighbors(record['id'], predicate=PREDICATE, limit=20)['relations']:
            name = unquote(edge['object'].removeprefix(FAMILY_PREFIX))
            names.append(name)
            families.setdefault(name, []).append(record['title'])
        rows.append({'record_id': record['id'], 'adjuvant': record['title'],
                     'matched_terms': found.get('retrieval_matches', {}).get(record['id'], {}).get('matched_terms', []),
                     'reported_mechanism_families': names})
    if not rows:
        return None
    return {'pack_sha256': pack.manifest_hash, 'rows': rows,
            'families': [{'family': k, 'adjuvants': v} for k, v in families.items()],
            'scope': 'Lexically matched literature-matrix rows and their author-reported mechanism families. '
                     'Candidate explanations to consider as competing hypotheses, not evidence or causal links.'}
