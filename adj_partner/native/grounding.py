"""Host-driven paper retrieval for grounded QA: PubMed Best Match, PMC full text, BM25 passages.

The model only writes keyword queries; ranking, fetching and passage selection are deterministic.
Retrieval failure or an empty result is not evidence against an answer.
"""
import json
import math
import re
import time
import unicodedata
from collections import Counter

import httpx
from lxml import etree

from . import ncbi
from .jats import inert_jats_content

ESEARCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
ESUMMARY = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
EPMC_SEARCH = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
EPMC_FULLTEXT = 'https://www.ebi.ac.uk/europepmc/webservices/rest/{}/fullTextXML'
MAX_BYTES = 8 * 1024 * 1024
STOP = frozenset('''a an the of and or to in on for with by from as at is are was were be been being this that these those
it its their there which what how why when where who does do did can could may might will would should into than then also
such both each other more most less many much very about between among during after before under over via using used use
based role effect effects impact mechanism mechanisms compared study studies authors findings results observed show shown
suggest significant specific potential key main important enhance enhanced improve increase increased reduce reduced'''.split())


def words(text):
    return re.findall(r'[a-z0-9]+', unicodedata.normalize('NFKC', text).casefold())


def bm25(query, docs, k1=1.2, b=0.75):
    tokens = [words(d) for d in docs]
    if not tokens:
        return []
    avg = sum(map(len, tokens)) / len(tokens) or 1
    df = Counter(w for t in tokens for w in set(t))
    wanted = {w for w in words(query) if w not in STOP}
    out = []
    for t in tokens:
        tf = Counter(t)
        out.append(sum(math.log(1 + (len(tokens) - df[w] + .5) / (df[w] + .5)) * tf[w] * (k1 + 1)
                       / (tf[w] + k1 * (1 - b + b * len(t) / avg)) for w in wanted if tf[w]))
    return out


def _text(node):
    return re.sub(r'\s+', ' ', ''.join(node.itertext())).strip()


def jats_passages(body):
    """Abstract, body paragraphs with their section title, and figure/table captions."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False,
                             huge_tree=False, remove_comments=True, recover=False)
    root = etree.fromstring(inert_jats_content(body), parser)
    title = root.xpath("//*[local-name()='article-title']")
    out = [dict(section='Abstract', text=_text(a)) for a in root.xpath("//*[local-name()='front']//*[local-name()='abstract']")]

    def walk(node, section):
        for child in node:
            tag = etree.QName(child).localname if isinstance(child.tag, str) else ''
            if tag == 'sec':
                head = child.xpath("./*[local-name()='title']")
                walk(child, _text(head[0]) if head else section)
            elif tag == 'p' and len(_text(child)) > 80:
                out.append(dict(section=section, text=_text(child)))
            elif tag in ('fig', 'table-wrap'):
                caption = child.xpath("./*[local-name()='caption']")
                label = child.xpath("./*[local-name()='label']")
                if caption:
                    out.append(dict(section=_text(label[0]) if label else tag, text=_text(caption[0])))
            elif tag and tag not in ('ref-list', 'back', 'title'):
                walk(child, section)

    for body_node in root.xpath("//*[local-name()='body']"):
        walk(body_node, '')
    return dict(title=_text(title[0]) if title else '', passages=out)


class GroundingRetriever:
    """No caching across runs; each call is bounded by timeout, response size and the NCBI rate.

    ``max_year`` limits PubMed search to papers published up to that year (retrospective evaluation).
    """

    def __init__(self, *, transport=None, timeout_seconds=20, api_key=None, email=None, max_year=None):
        if max_year is not None and (type(max_year) is not int or not 1800 <= max_year <= 2100):
            raise ValueError('max_year must be an integer year')
        self.transport, self.timeout, self.max_year = transport, timeout_seconds, max_year
        self.common = {k: v for k, v in (('api_key', api_key), ('email', email)) if v}
        self.requests = []

    @property
    def secrets(self):
        return tuple(v for k, v in self.common.items() if k == 'api_key')

    def _get(self, url, params):
        if 'ncbi.nlm.nih.gov' in url:
            params = {**params, **self.common}
            ncbi.wait(self.common.get('api_key'))
        with httpx.Client(transport=self.transport, timeout=self.timeout, follow_redirects=False, trust_env=False) as client:
            for attempt in range(2):
                try:
                    response = client.get(url, params=params)
                except httpx.TransportError:
                    if attempt:
                        raise
                    time.sleep(1.5)
                    continue
                if response.status_code == 200:
                    if len(response.content) > MAX_BYTES:
                        return None
                    self.requests.append(str(response.request.url.copy_remove_param('api_key')))
                    return response.content
                if response.status_code != 429 and response.status_code < 500 or attempt:
                    return None
                time.sleep(1.5)
        return None

    def pubmed(self, query, size=5):
        cutoff = dict(datetype='pdat', mindate='1800', maxdate=str(self.max_year)) if self.max_year else {}
        body = self._get(ESEARCH, dict(db='pubmed', retmode='json', retmax=size, sort='relevance', term=query, **cutoff))
        ids = json.loads(body)['esearchresult'].get('idlist', []) if body else []
        if not ids:
            return []
        body = self._get(ESUMMARY, dict(db='pubmed', retmode='json', id=','.join(ids)))
        result = json.loads(body).get('result', {}) if body else {}
        return [dict(pmid=i, title=result.get(i, {}).get('title', ''), year=(result.get(i, {}).get('pubdate') or '')[:4],
                     pmcid=next((a['value'] for a in result.get(i, {}).get('articleids', []) if a.get('idtype') == 'pmc'), None))
                for i in ids]

    def relaxed(self, query, size=5):
        """PubMed ANDs every term, so one unmatched synonym empties a query; retry with one term dropped."""
        records = self.pubmed(query, size)
        terms = query.split()
        for i in range(len(terms) if not records and len(terms) >= 4 else 0):
            records = self.pubmed(' '.join(terms[:i] + terms[i + 1:]), size)
            if records:
                break
        return records

    def fulltext(self, pmcid):
        body = self._get(EPMC_FULLTEXT.format(pmcid), {})
        if not body:
            return None
        try:
            return jats_passages(body)
        except (ValueError, etree.XMLSyntaxError):
            return None

    def abstract(self, pmid):
        body = self._get(EPMC_SEARCH, dict(query=f'EXT_ID:{pmid} AND SRC:MED', format='json', resultType='core'))
        rows = json.loads(body).get('resultList', {}).get('result', []) if body else []
        text = rows[0].get('abstractText') if rows else None
        return re.sub(r'<[^>]+>', ' ', text).strip() if text else None

    def gather(self, question, queries, *, k_papers=3, k_abstracts=2, k_passages=8, max_chars=1400):
        self.requests = []
        papers, seen = [], set()
        full = abstracts = 0
        for query in queries:
            for record in self.relaxed(query):
                if record['pmid'] in seen:
                    continue
                seen.add(record['pmid'])
                doc = self.fulltext(record['pmcid']) if record['pmcid'] and full < k_papers else None
                if doc and doc['passages']:
                    full += 1
                    papers.append(dict(record, query=query, title=doc['title'] or record['title'],
                                       text_availability='full_text', passages=doc['passages']))
                elif abstracts < k_abstracts and (text := self.abstract(record['pmid'])):
                    abstracts += 1
                    papers.append(dict(record, query=query, text_availability='abstract',
                                       passages=[dict(section='Abstract', text=text)]))
        pool = [(p, x) for p in papers for x in p['passages']]
        scores = bm25(question + ' ' + ' '.join(queries), [x['text'] for _, x in pool])
        ranked = sorted(((s, i) for i, s in enumerate(scores) if s > 0), key=lambda z: -z[0])[:k_passages]
        return dict(papers=[{k: v for k, v in p.items() if k != 'passages'} for p in papers],
                    passages=[dict(pmid=pool[i][0]['pmid'], pmcid=pool[i][0]['pmcid'], title=pool[i][0]['title'],
                                   year=pool[i][0]['year'], section=pool[i][1]['section'],
                                   text=pool[i][1]['text'][:max_chars], score=round(s, 3)) for s, i in ranked],
                    requests=list(self.requests))
