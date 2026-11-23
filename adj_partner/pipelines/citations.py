"""Citation units and source matching for Markdown research reports.

Sources are the ones a run read: ``sources.json`` (from ``ResearchStore.list_sources``) next to the run's
``state/artifacts``; artifact bytes are checked against their hash before use.
"""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

SOURCE_CHARS = 90000
URL = r'https?://(?:[^\s()<>]|\([^\s()<>]*\))+'
LINK = re.compile(r'\[[^\]]*\]\((' + URL + r')\)')
BARE = re.compile(r'(?<![(\[<])\b(?:https?://(?:[^\s()<>\]|]|\([^\s()<>]*\))+)')


def clean_url(u):
    u = u.strip().rstrip('.,;:')
    while u.endswith(')') and u.count('(') < u.count(')'):
        u = u[:-1]
    return u


def urls_in(text):
    found = [clean_url(u) for u in LINK.findall(text)]
    stripped = LINK.sub(' ', text)
    found += [clean_url(u) for u in BARE.findall(stripped)]
    return sorted(set(found))


def units(article):
    """Paragraph-bound citation units; tables split into rows, multi-link lists into items, link-only blocks merged up."""
    out = []
    for block in re.split(r'\n\s*\n', article):
        lines = [l for l in block.split('\n') if l.strip()]
        if not lines or not urls_in(block):
            out.append({'text': block, 'urls': []})
            continue
        if sum(l.lstrip().startswith('|') for l in lines) >= max(2, len(lines) - 1):
            header = lines[0]
            for row in lines[1:]:
                if urls_in(row):
                    out.append({'text': header + '\n' + row, 'urls': urls_in(row)})
            continue
        items, lead = [], []
        for l in lines:
            if re.match(r'\s{0,3}([-*+]|\d+[.)])\s', l):
                items.append([l])
            elif items:
                items[-1].append(l)
            else:
                lead.append(l)
        if len([i for i in items if urls_in('\n'.join(i))]) >= 2:
            for i in items:
                t = '\n'.join(lead + i)
                if urls_in('\n'.join(i)):
                    out.append({'text': t, 'urls': urls_in('\n'.join(i))})
            continue
        residue = re.sub(r'\s+', ' ', LINK.sub('', BARE.sub('', block))).strip(' .;,()-–—*_>')
        if len(residue) < 40 and out:
            prev = out[-1]
            prev['text'] += '\n' + block
            prev['urls'] = sorted(set(prev['urls']) | set(urls_in(block)))
            continue
        out.append({'text': block, 'urls': urls_in(block)})
    rows = [u for u in out if u['urls']]
    return [{'paragraph_id': i + 1, 'text': u['text'], 'allowed_urls': u['urls']} for i, u in enumerate(rows)]


def identifiers(url):
    u = unquote(url)
    doi = re.search(r'(10\.\d{4,9}/[^\s?#]+)', u)
    pmid = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', u) or re.search(r'europepmc\.org/(?:abstract|article)/MED/(\d+)', u)
    pmc = re.search(r'(PMC\d+)', u, re.I)
    nature = re.search(r'nature\.com/articles/(s\d{5}-\d{3}-[\w-]+|[a-z]+\d+)', u)
    doi = doi.group(1).rstrip('/.') if doi else ('10.1038/' + nature.group(1) if nature else None)
    if doi and re.search(r'/(full|abstract|pdf|epdf|fulltext)$', doi):
        doi = re.sub(r'/(full|abstract|pdf|epdf|fulltext)$', '', doi)
    if doi and 'frontiersin.org' in u:
        doi = re.sub(r'/(full|abstract|pdf)$', '', doi)
    return {'doi': doi, 'pmid': pmid.group(1) if pmid else None, 'pmcid': pmc.group(1).upper() if pmc else None}


def trim(text):
    """Drop reference lists and cap length so the judge sees body text."""
    m = re.search(r'\n#{1,4}\s*(References|REFERENCES|Bibliography)\b|\n(References|REFERENCES)\n', text)
    if m and m.start() > len(text) * 0.3:
        text = text[:m.start()]
    return text[:SOURCE_CHARS]


def _texts(obj):
    found = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ('text', 'abstract') and isinstance(v, str):
                    found.append(v)
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(obj)
    return '\n'.join(found)


def norm_url(u):
    u = unquote(u or '').strip().lower().replace('http://', 'https://').rstrip('/')
    return u.replace('https://www.ncbi.nlm.nih.gov/pmc/articles/', 'https://pmc.ncbi.nlm.nih.gov/articles/')


def read_sources(src_dir):
    """Every source the harness holds text for (hash-verified), with URL/identifier aliases."""
    path = Path(src_dir) / 'sources.json'
    if not path.exists():
        return []
    out = []
    for s in json.loads(path.read_text())['sources']:
        p = s.get('provenance', {})
        avail = p.get('text_availability')
        art = Path(src_dir) / 'state/artifacts' / s['artifact_hash']
        text = ''
        if avail in ('derived_full_text', 'full_text', 'abstract') and art.exists():
            data = art.read_bytes()
            if hashlib.sha256(data).hexdigest() != s['artifact_hash']:
                raise ValueError('Source hash mismatch')
            try:
                text = _texts(json.loads(data))
            except (json.JSONDecodeError, UnicodeDecodeError):
                text = ''
        urls = {norm_url(x) for x in (p.get('url'), p.get('request_url'), p.get('requested_url')) if isinstance(x, str)}
        ids = {'doi': (p.get('doi') or '').lower() or None, 'pmid': str(p['pmid']) if p.get('pmid') else None,
               'pmcid': (p.get('pmcid') or '').upper() or None}
        for u in list(urls):
            for k, v in identifiers(u).items():
                if v and not ids.get(k):
                    ids[k] = v.lower() if k == 'doi' else v
        out.append({'title': s.get('title'), 'availability': avail if text else (avail or 'none') + ':no_text',
                    'has_text': bool(text), 'text': text, 'urls': urls, 'ids': ids, 'artifact': s['artifact_hash'],
                    'kind': s.get('kind')})
    return out


def match(sources, url):
    nu, ids = norm_url(url), identifiers(url)
    hits = []
    for s in sources:
        if nu in s['urls'] or any(v and s['ids'].get(k) and (v.lower() if k == 'doi' else v) == s['ids'][k]
                                  for k, v in ids.items()):
            hits.append(s)
    return hits
