"""Host literature retrieval for grounded QA v2: PubMed/PMC full text plus Google Scholar and web hits.

Scholar and web search go through the Serper gateway; up to ``k_pages`` hit pages are read with the configured
page reader (Jina gateway, then direct fetch). All paragraphs share one BM25 pool ranked by the question
(weighted twice), the model's queries and the draft answer. Without gateway credentials only PubMed is used.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import time
from urllib.parse import urlsplit

from .grounding import GroundingRetriever, bm25

# Hosts that may republish benchmark question/answer pairs instead of source papers.
BLOCKED_HOSTS = ('huggingface.co', 'github.com', 'githubusercontent.com', 'kaggle.com', 'paperswithcode.com',
                 'quizlet.com', 'chegg.com', 'coursehero.com', 'brainly.com', 'studocu.com', 'scribd.com')


def blocked(url):
    host = (urlsplit(url).hostname or '').lower()
    return any(host == b or host.endswith('.' + b) for b in BLOCKED_HOSTS)


class SerperScholarWeb:
    """Serper ``/scholar`` and ``/search`` through the AI Hub gateway; transient failures are retried."""

    def __init__(self, client, *, attempts=3, sleep=time.sleep):
        self.client, self.attempts, self.sleep = client, attempts, sleep

    def search(self, endpoint, query, num=8):
        for attempt in range(self.attempts):
            try:
                _, raw = self.client.post(endpoint, {'q': query, 'num': num}, failure_prefix='web_search')
                break
            except Exception:
                if attempt == self.attempts - 1:
                    raise
                self.sleep(2 * (attempt + 1))
        hits = []
        for item in json.loads(raw).get('organic', []):
            url = item.get('link') or item.get('pdfUrl')
            if not isinstance(url, str) or blocked(url) or ('google.com' in url and '/goto' in url):
                continue
            hits.append(dict(title=str(item.get('title', ''))[:1000], url=url, snippet=str(item.get('snippet', ''))[:4000],
                             pdf=item.get('pdfUrl'), year=item.get('year')))
        return hits


class HybridRetriever:
    def __init__(self, pubmed, search=None, reader=None, *, k_pages=6, k_passages=12, max_chars=1400):
        self.pubmed, self.web, self.reader = pubmed, search, reader
        self.k_pages, self.k_passages, self.max_chars = k_pages, k_passages, max_chars

    @classmethod
    def from_environment(cls, project_root, env_file, pubmed=None):
        from .gateway import GatewayClient, settings
        from ..web_access import web_page_fetcher_from_environment
        if pubmed is None:
            values = settings(project_root, env_file, ('ADJ_NCBI_API_KEY', 'ADJ_NCBI_EMAIL'))
            pubmed = GroundingRetriever(api_key=values['ADJ_NCBI_API_KEY'] or None,
                                        email=values['ADJ_NCBI_EMAIL'] or None, timeout_seconds=45)
        values = settings(project_root, env_file, ('ADJ_AIHUB_GATEWAY_URL', 'ADJ_SERPER_GATEWAY_APP_ID',
                                                   'ADJ_SERPER_GATEWAY_APP_KEY'))
        search = reader = None
        if values['ADJ_SERPER_GATEWAY_APP_ID'] and values['ADJ_SERPER_GATEWAY_APP_KEY']:
            search = SerperScholarWeb(GatewayClient(values['ADJ_SERPER_GATEWAY_APP_ID'],
                                                    values['ADJ_SERPER_GATEWAY_APP_KEY'], 'serper',
                                                    base_url=values['ADJ_AIHUB_GATEWAY_URL'] or None,
                                                    timeout_seconds=40))
            reader = web_page_fetcher_from_environment(project_root, env_file)
        return cls(pubmed, search, reader)

    def _pubmed_pool(self, queries, k_papers=3, k_abstracts=3):
        pool, papers, seen, full, abstracts = [], [], set(), 0, 0
        for query in queries:
            for record in _quiet(self.pubmed.relaxed, query) or []:
                if record['pmid'] in seen:
                    continue
                seen.add(record['pmid'])
                paper = dict(title=record['title'], url=f"https://pubmed.ncbi.nlm.nih.gov/{record['pmid']}/",
                             year=record.get('year'), pmid=record['pmid'], pmcid=record.get('pmcid'), query=query)
                doc = _quiet(self.pubmed.fulltext, record['pmcid']) if record.get('pmcid') and full < k_papers else None
                if doc and doc['passages']:
                    full += 1
                    paper.update(title=doc['title'] or paper['title'], text_availability='full_text')
                    pool += [(paper, x['section'] or 'Body', x['text']) for x in doc['passages']]
                    papers.append(paper)
                elif abstracts < k_abstracts and (text := _quiet(self.pubmed.abstract, record['pmid'])):
                    abstracts += 1
                    paper['text_availability'] = 'abstract'
                    pool.append((paper, 'Abstract', text))
                    papers.append(paper)
        return pool, papers

    def _hits(self, question, queries):
        jobs = [(endpoint, q[:300]) for q in [question] + list(queries) for endpoint in ('/scholar', '/search')]
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda job: _quiet(self.web.search, *job), jobs))
        seen, scholar, web = set(), [], []
        for (endpoint, query), hits in zip(jobs, results):
            for hit in hits or []:
                if hit['url'] not in seen and not blocked(hit['url']):
                    seen.add(hit['url'])
                    (scholar if endpoint == '/scholar' else web).append(dict(hit, query=query, engine=endpoint[1:]))
        return [h for pair in zip(scholar, web) for h in pair] + scholar[len(web):] + web[len(scholar):]

    def _read(self, hit):
        from ..web_access import markdown_paragraphs
        for url in dict.fromkeys(u for u in (hit.get('pdf'), hit['url']) if u):
            page = _quiet(self.reader.fetch, url)
            if page is not None and page.media == 'text/markdown':
                return _quiet(markdown_paragraphs, page.body.decode('utf-8', 'replace')) or []
        return []

    def gather(self, question, queries, draft=''):
        if self.web is None:
            pool, papers = self._pubmed_pool(queries)
            hits = []
        else:
            with ThreadPoolExecutor(2) as ex:
                pubmed = ex.submit(self._pubmed_pool, queries)
                hits = self._hits(question, queries)
                pool, papers = pubmed.result()
        titles = {p['title'].lower() for p in papers}
        hits = [h for h in hits if h['title'].lower() not in titles]
        for hit in hits:
            if hit['snippet']:
                pool.append((hit, 'Search snippet', hit['snippet']))
        to_read = hits[:self.k_pages] if self.reader is not None else []
        with ThreadPoolExecutor(4) as ex:
            bodies = list(ex.map(self._read, to_read))
        for hit, body in zip(to_read, bodies):
            pool += [(hit, 'Web page', text) for text in body]
        for i, hit in enumerate(hits):
            papers.append(dict(title=hit['title'], url=hit['url'], year=hit.get('year'), query=hit['query'],
                               engine=hit['engine'],
                               # A read page may be an abstract page or the full article.
                               text_availability='unknown' if i < len(bodies) and bodies[i] else 'search_snippet'))
        scores = bm25(' '.join([question, question, ' '.join(queries), draft]), [t for _, _, t in pool])
        ranked = sorted(((s, i) for i, s in enumerate(scores) if s > 0), key=lambda z: -z[0])[:self.k_passages]
        passages = []
        for s, i in ranked:
            paper, section, text = pool[i]
            passages.append(dict(url=paper['url'], title=paper['title'], year=paper.get('year') or 'n.d.',
                                 pmid=paper.get('pmid'), pmcid=paper.get('pmcid'), section=section,
                                 text=text[:self.max_chars], score=round(s, 3)))
        return dict(papers=papers, passages=passages)


def _quiet(fn, *args):
    """A failed search, read or PubMed request is an evidence gap, not a failed answer."""
    try:
        return fn(*args)
    except Exception:
        return None
