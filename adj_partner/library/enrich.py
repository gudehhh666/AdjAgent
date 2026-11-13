"""Grow the literature library: fill missing text for filed papers and follow links from seed papers.

Enrichment fetches open-access full text (Europe PMC JATS) for papers with a PMCID, and abstracts for
title-only papers by DOI/PMID. Expansion walks one hop from seed papers (adjuvant-tagged full texts)
along Europe PMC references and citations and PubMed similar-article links, keeps candidates whose
title/abstract names an adjuvant, and records every link with how it was found. Each attempt is
recorded so a re-run resumes instead of refetching.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import re
import threading
import time

import httpx

from .store import _now

EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest'
ELINK = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi'
RELEVANT = re.compile(r'(?i)\badjuvant|\badjuvanted|\badjuvanticity')


class Fetcher:
    """Rate-limited GET with a few retries on transient statuses; never raises for HTTP errors."""

    def __init__(self, *, transport=None, timeout=30, interval=0.12, ncbi_key=None, email=None, retries=3):
        self.client = httpx.Client(transport=transport, timeout=timeout, follow_redirects=True,
                                   headers={'User-Agent': 'AdjAgent-literature-library/1.0'})
        self.interval, self.retries = interval, retries
        self.ncbi_key, self.email = ncbi_key, email
        self._lock, self._last = threading.Lock(), 0.0

    def get(self, url, params=None):
        for attempt in range(self.retries + 1):
            with self._lock:
                wait = self._last + self.interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            try:
                response = self.client.get(url, params=params)
            except httpx.HTTPError:
                response = None
            if response is not None and response.status_code not in (429, 500, 502, 503, 504):
                return response
            time.sleep(min(2 ** attempt, 20) if self.interval else 0)
        return response

    def json(self, url, params=None):
        response = self.get(url, params)
        if response is None or response.status_code != 200:
            return None
        try:
            return response.json()
        except ValueError:
            return None


def _record(result):
    """Europe PMC core result -> the library's metadata-record shape."""
    journal = (((result.get('journalInfo') or {}).get('journal') or {}).get('title'))
    return {key: value for key, value in {
        'pmid': result.get('pmid'), 'pmcid': result.get('pmcid'), 'doi': result.get('doi'),
        'title': result.get('title'), 'abstract': re.sub(r'<[^>]+>', '', result.get('abstractText') or '') or None,
        'year': result.get('pubYear'), 'journal': journal, 'authors': result.get('authorString'),
        'source': 'europe_pmc', 'is_open_access': result.get('isOpenAccess') == 'Y'}.items() if value}


class Enricher:
    def __init__(self, library, fetcher, *, workers=4, log=print):
        self.library, self.fetcher, self.workers, self.log = library, fetcher, workers, log

    # ---------- bookkeeping ----------

    def _pending(self, step, sql, params=()):
        with self.library._connect() as conn:
            return [dict(r) for r in conn.execute(
                f"{sql} AND paper_id NOT IN (SELECT paper_id FROM attempts WHERE step=?)", (*params, step))]

    def _mark(self, paper_id, step, status):
        with self.library._connect() as conn:
            conn.execute('INSERT OR REPLACE INTO attempts VALUES(?,?,?,?)', (paper_id, step, status, _now()))

    def _file_record(self, record, origin, query):
        content = json.dumps(record, sort_keys=True, ensure_ascii=False).encode('utf-8')
        url = f"https://europepmc.org/article/MED/{record['pmid']}" if record.get('pmid') else None
        return self.library.ingest_safely(content, 'application/json',
                                          {'provider': 'europe_pmc', 'query': query, 'url': url,
                                           **{k: record.get(k) for k in ('pmid', 'pmcid', 'doi')}}, origin=origin)

    def _core(self, clauses):
        """Batch Europe PMC lookups: clauses like EXT_ID:123 AND SRC:MED or DOI:"10.x"."""
        data = self.fetcher.json(f'{EPMC}/search', {'query': ' OR '.join(clauses), 'resultType': 'core',
                                                    'format': 'json', 'pageSize': 100})
        return [] if data is None else (data.get('resultList') or {}).get('result') or []

    # ---------- enrichment ----------

    def abstracts(self, *, limit=None, batch=20):
        rows = self._pending('abstract', "SELECT paper_id, pmid, doi FROM papers WHERE best_text IN"
                                         " ('identifier_only','title_only') AND (pmid IS NOT NULL OR doi IS NOT NULL)")
        rows = rows[:limit] if limit else rows
        filled = 0
        chunks = [rows[i:i + batch] for i in range(0, len(rows), batch)]

        def lookup(chunk):
            clauses = [f'(EXT_ID:{r["pmid"]} AND SRC:MED)' if r['pmid'] else f'DOI:"{r["doi"]}"' for r in chunk]
            return chunk, self._core(clauses)

        with ThreadPoolExecutor(self.workers) as pool:
            for index, (chunk, results) in enumerate(pool.map(lookup, chunks), 1):
                by_pmid = {r.get('pmid'): r for r in results if r.get('pmid')}
                by_doi = {(r.get('doi') or '').lower(): r for r in results if r.get('doi')}
                for row in chunk:
                    result = by_pmid.get(row['pmid']) or by_doi.get((row['doi'] or '').lower())
                    record = _record(result) if result else None
                    if record and (record.get('abstract') or record.get('pmid') or record.get('pmcid')):
                        self._file_record(record, {'kind': 'enrichment', 'ref': 'abstract', 'project': None},
                                          'enrichment: abstract by identifier')
                        filled += bool(record.get('abstract'))
                    self._mark(row['paper_id'], 'abstract', 'filled' if record and record.get('abstract') else 'none')
                if index % 50 == 0:
                    self.log(f'abstracts: {index}/{len(chunks)} batches, {filled} filled')
        return {'checked': len(rows), 'filled': filled}

    def fulltext(self, *, limit=None):
        rows = self._pending('fulltext', "SELECT paper_id, pmid, pmcid FROM papers WHERE pmcid IS NOT NULL"
                                         " AND best_text != 'full_text'")
        rows = rows[:limit] if limit else rows
        filled = 0

        def fetch(row):
            response = self.fetcher.get(f"{EPMC}/{row['pmcid']}/fullTextXML")
            ok = response is not None and response.status_code == 200 and b'<article' in response.content[:4000]
            return row, (response.content if ok else None)

        with ThreadPoolExecutor(self.workers) as pool:
            for index, (row, content) in enumerate(pool.map(fetch, rows), 1):
                if content is not None:
                    touched = self.library.ingest_safely(
                        content, 'application/xml',
                        {'provider': 'europe_pmc', 'pmcid': row['pmcid'], 'pmid': row['pmid'],
                         'url': f"{EPMC}/{row['pmcid']}/fullTextXML", 'text_availability': 'full_text'},
                        origin={'kind': 'enrichment', 'ref': 'fulltext', 'project': None})
                    filled += bool(touched)
                self._mark(row['paper_id'], 'fulltext', 'filled' if content is not None else 'not_open_access')
                if index % 200 == 0:
                    self.log(f'fulltext: {index}/{len(rows)}, {filled} filled')
        return {'checked': len(rows), 'filled': filled}

    # ---------- expansion ----------

    def seeds(self):
        return self._pending('expand', "SELECT p.paper_id, p.pmid FROM papers p WHERE p.best_text='full_text'"
                                       " AND p.pmid IS NOT NULL AND EXISTS(SELECT 1 FROM tags t WHERE"
                                       " t.paper_id=p.paper_id AND t.kind='adjuvant')")

    def _relevant(self, text):
        if RELEVANT.search(text or ''):
            return True
        padded = ' ' + ' '.join(re.findall(r'[^\W_]+', text or '')) + ' '
        lowered = padded.lower()
        return any(phrase in (padded if exact else lowered) for _, phrase, exact in self.library.adjuvant_names())

    def _links(self, pmid):
        found = []
        for relation, path, key in (('cites', 'references', 'referenceList'), ('cited_by', 'citations', 'citationList')):
            data = self.fetcher.json(f'{EPMC}/MED/{pmid}/{path}', {'format': 'json', 'pageSize': 1000})
            items = ((data or {}).get(key) or {}).get(key[:-4]) or []
            for item in items:
                if item.get('source') == 'MED' and item.get('id'):
                    found.append((relation, str(item['id']), item.get('title') or '', None))
        params = {'dbfrom': 'pubmed', 'db': 'pubmed', 'id': pmid, 'cmd': 'neighbor_score', 'retmode': 'json'}
        if self.fetcher.ncbi_key:
            params['api_key'] = self.fetcher.ncbi_key
        if self.fetcher.email:
            params['email'] = self.fetcher.email
        data = self.fetcher.json(ELINK, params)
        for linkset in (data or {}).get('linksets') or []:
            for db in linkset.get('linksetdbs') or []:
                if db.get('linkname') == 'pubmed_pubmed':
                    for link in db.get('links') or []:
                        if isinstance(link, dict) and str(link.get('id')) != str(pmid):
                            found.append(('similar', str(link['id']), None, float(link.get('score') or 0)))
        return found

    def expand(self, *, per_seed=20, max_new=2000, limit_seeds=None):
        seeds = self.seeds()[:limit_seeds] if limit_seeds else self.seeds()
        added = linked = 0
        with ThreadPoolExecutor(self.workers) as pool:
            for index, (seed, links) in enumerate(pool.map(lambda s: (s, self._links(s['pmid'])), seeds), 1):
                taken = {'cites': 0, 'cited_by': 0, 'similar': 0}
                similar = sorted((l for l in links if l[0] == 'similar'), key=lambda l: -l[3])
                ordered = [l for l in links if l[0] != 'similar'] + similar
                with self.library._connect() as conn:
                    known = {pmid: row['paper_id'] for pmid in {l[1] for l in ordered}
                             for row in [conn.execute("SELECT paper_id FROM identifiers WHERE scheme='pmid' AND value=?",
                                                      (pmid,)).fetchone()] if row}
                unknown = [l for l in ordered if l[1] not in known and (l[2] is None or self._relevant(l[2]))]
                details = {}
                wanted = list(dict.fromkeys(l[1] for l in unknown))[:per_seed * 3]
                for i in range(0, len(wanted), 20):
                    for result in self._core([f'(EXT_ID:{p} AND SRC:MED)' for p in wanted[i:i + 20]]):
                        if result.get('pmid'):
                            details[result['pmid']] = _record(result)
                rows = []
                for relation, pmid, _title, score in ordered:
                    if pmid in known:
                        rows.append((relation, known[pmid], score))
                        continue
                    record = details.get(pmid)
                    if (record is None or taken[relation] >= per_seed or added >= max_new
                            or not self._relevant(f"{record.get('title', '')} {record.get('abstract', '')}")):
                        continue
                    touched = self._file_record(record, {'kind': 'expansion', 'ref': f"{relation}:{seed['paper_id']}",
                                                         'project': None}, f"{relation} of {seed['paper_id']}")
                    if touched:
                        taken[relation] += 1
                        added += 1
                        known[pmid] = touched[0]
                        rows.append((relation, touched[0], score))
                with self.library._connect() as conn:
                    for relation, paper_id, score in rows:
                        if paper_id != seed['paper_id']:
                            linked += conn.execute('INSERT OR IGNORE INTO relations VALUES(?,?,?,?,?,?)',
                                                   (seed['paper_id'], paper_id, relation,
                                                    'pubmed_elink' if relation == 'similar' else 'europe_pmc',
                                                    score, _now())).rowcount
                self._mark(seed['paper_id'], 'expand', 'done')
                if index % 10 == 0:
                    self.log(f'expand: {index}/{len(seeds)} seeds, {added} new papers, {linked} links')
                if added >= max_new:
                    break
        return {'seeds': len(seeds), 'new_papers': added, 'links': linked}
