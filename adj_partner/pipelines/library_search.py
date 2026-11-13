"""Read-only ``library_search`` over the local literature library, registered per AgentService by the host."""
import json
import re
import sqlite3
import threading

from ..library import default_root

STOP = set('the of and in on for to a an by with from is are was were be as at or that this which what how does do did '
           'its it their between than into using use based compared versus vs can could would should may effect effects role '
           'mechanism mechanisms study studies'.split())
_LOCAL = threading.local()


def default_db():
    return default_root() / 'library.sqlite3'


def _conn(db):
    cache = getattr(_LOCAL, 'conns', None)
    if cache is None:
        cache = _LOCAL.conns = {}
    c = cache.get(db)
    if c is None:
        c = sqlite3.connect(f'file:{db}?mode=ro', uri=True, timeout=30, check_same_thread=False)
        c.row_factory = sqlite3.Row
        cache[db] = c
    return c


def search(db, query, paper_id=None, limit=6):
    terms = [t for t in re.findall(r'[A-Za-z0-9][A-Za-z0-9\-]*', query or '') if len(t) > 1 and t.lower() not in STOP][:16]
    terms = [t.replace('-', ' ') for t in terms]
    if not terms:
        return []
    for joiner in (' AND ', ' OR '):
        match = joiner.join(f'"{t}"' for t in terms)
        sql = ('SELECT p.paper_id, p.section, p.locator, p.text, bm25(passages_fts) AS score, a.title, a.year, a.best_text '
               'FROM passages_fts JOIN passages p ON p.id = passages_fts.rowid JOIN papers a ON a.paper_id = p.paper_id '
               "WHERE passages_fts MATCH ? AND p.locator NOT LIKE 'reference:%'")
        params = [match]
        if paper_id:
            sql += ' AND p.paper_id = ?'
            params.append(paper_id)
        sql += ' ORDER BY score LIMIT ?'
        params.append(int(limit) * 3)
        rows = [dict(r) for r in _conn(db).execute(sql, params)]
        if rows:
            out, per = [], {}
            for r in rows:
                if not paper_id and per.get(r['paper_id'], 0) >= 2:
                    continue
                per[r['paper_id']] = per.get(r['paper_id'], 0) + 1
                out.append(r)
                if len(out) >= limit:
                    break
            return out
    return []


DESCRIPTION = (
    'Search the offline project literature library (about 22k vaccine/adjuvant papers; full text or abstracts) and '
    'return ranked passage excerpts with paper_id, title, year, section. Use 3-8 distinctive terms (entities, '
    'adjuvant, antigen, model, readout). Pass paper_id to search inside one paper for a specific result. '
    'Passages are paper text, the strongest local evidence for paper-specific questions; not verified beyond the paper.')
PARAMETERS = {'type': 'object', 'additionalProperties': False, 'required': ['query'],
              'properties': {'query': {'type': 'string', 'minLength': 2, 'maxLength': 300},
                             'paper_id': {'type': 'string', 'maxLength': 200},
                             'limit': {'type': 'integer', 'minimum': 1, 'maximum': 8}}}


def register(service, db=None):
    from ..contracts import ToolResult
    from ..tools import ToolSpec, ToolFailure

    db = str(db or default_db())
    spec = ToolSpec('library_search', DESCRIPTION, PARAMETERS)

    def handler(context, query, paper_id=None, limit=6):
        context.check_cancelled()
        try:
            rows = search(db, query, paper_id, limit)
        except sqlite3.Error as e:
            raise ToolFailure('library_unavailable', 'Local literature library query failed: ' + type(e).__name__)
        hits = []
        width = 900 if paper_id else 650
        for r in rows:
            hits.append({'paper_id': r['paper_id'], 'title': (r['title'] or '')[:160], 'year': r['year'],
                         'text_level': r['best_text'], 'section': (r['section'] or '')[:80], 'locator': r['locator'],
                         'passage': (r['text'] or '')[:width]})
        summary = {'query': query[:200], 'paper_id': paper_id, 'returned': len(hits), 'hits': hits,
                   'scope': 'Offline local library passages (paper text). Absence of a hit is not evidence of absence.'}
        while len(json.dumps(summary, ensure_ascii=False).encode()) > 6000 and summary['hits']:
            for h in summary['hits']:
                h['passage'] = h['passage'][:int(len(h['passage']) * 0.8)]
            if all(len(h['passage']) < 200 for h in summary['hits']):
                summary['hits'].pop()
        return ToolResult(data={'summary': summary})

    service.catalog.register(spec, handler)
    return ['library_search']
