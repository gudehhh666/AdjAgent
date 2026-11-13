"""Project-wide literature library: every paper the host retrieves, deduplicated and searchable.

Raw bytes live in a content-addressed blob store. Papers, identifiers, documents, passages,
sightings, tags and resource-pack links live in one SQLite file with an FTS5 passage index.
Ingestion is idempotent by content hash and is a side index: it never changes what a run
registered. Library passages are retrieved material, not verified claims.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3

TEXT_RANK = {'identifier_only': 0, 'title_only': 1, 'abstract': 2, 'full_text': 3}
RAW_MEDIA = ('application/xml', 'text/xml', 'application/pdf', 'application/x-bioc+json')
PROVENANCE_KEYS = ('provider', 'source', 'query', 'url', 'license', 'retrieved_at', 'text_availability',
                   'doi', 'pmid', 'pmcid')
MAX_PASSAGE = 6_000
GENERIC = {'vaccine', 'vaccines', 'adjuvant', 'adjuvants', 'system', 'formulation', 'the', 'of'}

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers(
  paper_id TEXT PRIMARY KEY, pmid TEXT, pmcid TEXT, doi TEXT, title TEXT, year TEXT, journal TEXT,
  authors TEXT, abstract TEXT, best_text TEXT NOT NULL, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS identifiers(
  scheme TEXT NOT NULL, value TEXT NOT NULL, paper_id TEXT NOT NULL, PRIMARY KEY(scheme, value));
CREATE TABLE IF NOT EXISTS documents(
  sha256 TEXT NOT NULL, paper_id TEXT NOT NULL, media_type TEXT, kind TEXT NOT NULL, provider TEXT,
  size INTEGER, blob TEXT, provenance TEXT, added_at TEXT NOT NULL, PRIMARY KEY(sha256, paper_id));
CREATE TABLE IF NOT EXISTS passages(
  id INTEGER PRIMARY KEY, paper_id TEXT NOT NULL, document_sha256 TEXT, section TEXT, locator TEXT,
  text TEXT NOT NULL, text_sha256 TEXT NOT NULL, UNIQUE(paper_id, text_sha256));
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(text, section, title, content='');
CREATE TABLE IF NOT EXISTS sightings(
  paper_id TEXT NOT NULL, origin_kind TEXT NOT NULL, origin_ref TEXT NOT NULL, project TEXT,
  provider TEXT, query TEXT, document_sha256 TEXT, seen_at TEXT NOT NULL,
  UNIQUE(paper_id, origin_ref, document_sha256));
CREATE TABLE IF NOT EXISTS tags(
  paper_id TEXT NOT NULL, kind TEXT NOT NULL, tag TEXT NOT NULL, basis TEXT, PRIMARY KEY(paper_id, kind, tag));
CREATE TABLE IF NOT EXISTS pack_links(
  paper_id TEXT NOT NULL, pack_sha256 TEXT, record_id TEXT NOT NULL, dataset TEXT, record_title TEXT,
  PRIMARY KEY(paper_id, record_id));
CREATE TABLE IF NOT EXISTS failures(origin_ref TEXT, provider TEXT, error TEXT, at TEXT);
CREATE TABLE IF NOT EXISTS relations(
  from_paper TEXT NOT NULL, to_paper TEXT NOT NULL, relation TEXT NOT NULL, source TEXT, score REAL,
  found_at TEXT NOT NULL, PRIMARY KEY(from_paper, to_paper, relation));
CREATE TABLE IF NOT EXISTS attempts(
  paper_id TEXT NOT NULL, step TEXT NOT NULL, status TEXT NOT NULL, at TEXT NOT NULL, PRIMARY KEY(paper_id, step));
CREATE INDEX IF NOT EXISTS sightings_project ON sightings(project);
CREATE INDEX IF NOT EXISTS tags_tag ON tags(kind, tag);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode('utf-8')).hexdigest()


def normalize_ids(*sources):
    """Merge pmid/pmcid/doi from records and provenance into canonical spellings."""
    ids = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ('pmid', 'pmcid', 'doi'):
            value = source.get(key)
            if value in (None, '') or key in ids:
                continue
            value = str(value).strip()
            if key == 'pmid' and re.fullmatch(r'\d{1,9}', value):
                ids['pmid'] = value
            elif key == 'pmcid' and re.fullmatch(r'(?i)(PMC)?\d{1,10}', value):
                ids['pmcid'] = 'PMC' + re.sub(r'(?i)^PMC', '', value)
            elif key == 'doi':
                value = re.sub(r'(?i)^(https?://(dx\.)?doi\.org/|doi:)', '', value).lower()
                if value.startswith('10.'):
                    ids['doi'] = value
    return ids


def _title_key(title):
    words = re.findall(r'[a-z0-9]+', (title or '').lower())
    return _sha(' '.join(words))[:20] if len(words) >= 4 else None


class LiteratureLibrary:
    def __init__(self, root, *, adjuvants=None):
        self.root = Path(root)
        self.blobs = self.root / 'blobs'
        self.path = self.root / 'library.sqlite3'
        self._adjuvants = adjuvants
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=60)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=60000')
        return conn

    # ---------- ingestion ----------

    def ingest(self, content, media_type, provenance, *, origin):
        """Ingest one retrieved artifact. Returns the paper ids it touched (possibly empty)."""
        provenance = provenance or {}
        provider = str(provenance.get('provider') or provenance.get('source') or '')
        sha = _sha(content)
        with self._connect() as conn:
            known = [row['paper_id'] for row in conn.execute('SELECT paper_id FROM documents WHERE sha256=?', (sha,))]
            if known:
                for paper_id in known:
                    self._sight(conn, paper_id, origin, provider, provenance, sha)
                return known
        items = self._interpret(content, media_type, provenance, provider)
        touched = []
        with self._connect() as conn:
            stored = None
            for item in items:
                paper_id = self._paper(conn, item['paper'], item['ids'], item['level'])
                if paper_id is None:
                    continue
                if stored is None:
                    stored = self._blob(content, media_type)
                conn.execute('INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?,?,?,?,?)',
                             (sha, paper_id, media_type, item['kind'], provider, len(content), stored,
                              json.dumps(self._provenance(provenance), sort_keys=True), _now()))
                title = conn.execute('SELECT title FROM papers WHERE paper_id=?', (paper_id,)).fetchone()['title']
                for passage in item['passages']:
                    self._passage(conn, paper_id, sha, passage, title)
                self._sight(conn, paper_id, origin, provider, provenance, sha)
                self._tag(conn, paper_id, origin)
                touched.append(paper_id)
        return touched

    def ingest_safely(self, content, media_type, provenance, *, origin):
        try:
            return self.ingest(content, media_type, provenance, origin=origin)
        except Exception as exc:  # side index: a library fault must not reach the caller
            try:
                with self._connect() as conn:
                    conn.execute('INSERT INTO failures VALUES(?,?,?,?)',
                                 (str(origin.get('ref')), str((provenance or {}).get('provider')),
                                  f'{type(exc).__name__}: {exc}'[:500], _now()))
            except Exception:
                pass
            return []

    def _interpret(self, content, media_type, provenance, provider):
        """Turn one artifact into [{paper, ids, level, kind, passages}] without touching the database."""
        media = (media_type or '').split(';')[0].strip().lower()
        if media in RAW_MEDIA:
            from ..native.documents import parse_document, _passages
            document = parse_document(content, media)
            passages = [p for p in _passages(document) if p.get('text')]
            paper = {'title': document.get('title') or provenance.get('title')}
            ids = normalize_ids(provenance)
            if media == 'application/pdf' and 'doi' not in ids:
                first = ' '.join(p['text'] for p in passages[:2])
                found = re.search(r'\b(10\.\d{4,9}/[^\s"<>]+[^\s"<>.,;)])', first)
                if found:
                    ids['doi'] = found.group(1).lower()
            if not paper['title'] and passages and not ids:
                return []
            return [{'paper': paper, 'ids': ids, 'level': 'full_text', 'kind': 'raw_fulltext', 'passages': passages}]
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return []
        if not isinstance(value, dict):
            return []
        if provider == 'document_parser' or value.get('document_type'):
            from ..native.documents import _passages
            passages = [p for p in _passages(value) if p.get('text')]
            level = 'full_text' if value.get('sections') or value.get('pages') else 'abstract'
            return [{'paper': {'title': value.get('title')}, 'ids': normalize_ids(value, provenance),
                     'level': level, 'kind': 'parsed_document', 'passages': passages}]
        if isinstance(value.get('paper'), dict):
            paper = value['paper']
            passages = value.get('passages') or ([value['passage']] if isinstance(value.get('passage'), dict) else [])
            level = 'full_text' if paper.get('text_availability') == 'full_text' or paper.get('full_text') else 'abstract'
            return [{'paper': paper, 'ids': normalize_ids(paper, provenance), 'level': level,
                     'kind': 'host_passages', 'passages': passages}]
        listed = value.get('records', value.get('candidates'))
        records = listed if isinstance(listed, list) else [value]
        items = []
        for record in records:
            if not isinstance(record, dict) or not (record.get('title') or normalize_ids(record)):
                continue
            abstract = record.get('abstract') or ''
            items.append({'paper': record, 'ids': normalize_ids(record, provenance if record is value else {}),
                          'level': 'abstract' if abstract else 'title_only', 'kind': 'metadata_record',
                          'passages': [{'section': 'Abstract', 'locator': 'abstract', 'text': abstract}] if abstract else []})
        return items

    def _paper(self, conn, paper, ids, level):
        title = paper.get('title') if isinstance(paper.get('title'), str) else None
        keys = [(scheme, ids[scheme]) for scheme in ('pmid', 'doi', 'pmcid') if scheme in ids]
        title_key = _title_key(title)
        if title_key:
            keys.append(('title', title_key))
        if not keys:
            return None
        found = []
        for scheme, value in keys:
            row = conn.execute('SELECT paper_id FROM identifiers WHERE scheme=? AND value=?', (scheme, value)).fetchone()
            if row and row['paper_id'] not in found:
                found.append(row['paper_id'])
        now = _now()
        if found:
            paper_id = found[0]
            for other in found[1:]:
                self._merge(conn, paper_id, other)
        else:
            paper_id = f'{keys[0][0]}:{keys[0][1]}'
            conn.execute('INSERT INTO papers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                         (paper_id, None, None, None, None, None, None, None, None, level, now, now))
        for scheme, value in keys:
            conn.execute('INSERT OR IGNORE INTO identifiers VALUES(?,?,?)', (scheme, value, paper_id))
        row = dict(conn.execute('SELECT * FROM papers WHERE paper_id=?', (paper_id,)).fetchone())
        authors = paper.get('authors')
        if isinstance(authors, list):
            authors = '; '.join(str(a) for a in authors[:30])
        update = {'pmid': ids.get('pmid'), 'pmcid': ids.get('pmcid'), 'doi': ids.get('doi'), 'title': title,
                  'year': str(paper['year'])[:4] if paper.get('year') else None,
                  'journal': paper.get('journal') if isinstance(paper.get('journal'), str) else None,
                  'authors': authors if isinstance(authors, str) else None,
                  'abstract': paper.get('abstract') if isinstance(paper.get('abstract'), str) else None}
        fields = {key: value for key, value in update.items() if value and not row[key]}
        if TEXT_RANK[level] > TEXT_RANK[row['best_text']]:
            fields['best_text'] = level
        fields['last_seen'] = now
        conn.execute(f"UPDATE papers SET {', '.join(f'{k}=?' for k in fields)} WHERE paper_id=?",
                     (*fields.values(), paper_id))
        return paper_id

    def _merge(self, conn, keep, other):
        row = conn.execute('SELECT * FROM papers WHERE paper_id=?', (other,)).fetchone()
        for table in ('identifiers', 'documents', 'sightings', 'tags', 'pack_links', 'passages', 'attempts'):
            conn.execute(f'UPDATE OR IGNORE {table} SET paper_id=? WHERE paper_id=?', (keep, other))
            conn.execute(f'DELETE FROM {table} WHERE paper_id=?', (other,))
        for column in ('from_paper', 'to_paper'):
            conn.execute(f'UPDATE OR IGNORE relations SET {column}=? WHERE {column}=?', (keep, other))
            conn.execute(f'DELETE FROM relations WHERE {column}=?', (other,))
        conn.execute('DELETE FROM relations WHERE from_paper=to_paper')
        if row:
            kept = dict(conn.execute('SELECT * FROM papers WHERE paper_id=?', (keep,)).fetchone())
            fields = {key: row[key] for key in ('pmid', 'pmcid', 'doi', 'title', 'year', 'journal', 'authors', 'abstract')
                      if row[key] and not kept[key]}
            if TEXT_RANK[row['best_text']] > TEXT_RANK[kept['best_text']]:
                fields['best_text'] = row['best_text']
            if fields:
                conn.execute(f"UPDATE papers SET {', '.join(f'{k}=?' for k in fields)} WHERE paper_id=?",
                             (*fields.values(), keep))
            conn.execute('DELETE FROM papers WHERE paper_id=?', (other,))

    def _passage(self, conn, paper_id, sha, passage, title):
        value = str(passage.get('text') or '').strip()[:MAX_PASSAGE]
        if len(value) < 20:
            return
        section = str(passage.get('section') or passage.get('section_title') or '')[:300]
        cursor = conn.execute('INSERT OR IGNORE INTO passages(paper_id, document_sha256, section, locator, text, text_sha256)'
                              ' VALUES(?,?,?,?,?,?)', (paper_id, sha, section, str(passage.get('locator') or '')[:200],
                                                       value, _sha(value)))
        if cursor.rowcount:
            conn.execute('INSERT INTO passages_fts(rowid, text, section, title) VALUES(?,?,?,?)',
                         (cursor.lastrowid, value, section, title or ''))

    def _sight(self, conn, paper_id, origin, provider, provenance, sha):
        conn.execute('INSERT OR IGNORE INTO sightings VALUES(?,?,?,?,?,?,?,?)',
                     (paper_id, origin.get('kind', 'run'), str(origin.get('ref', '')), origin.get('project'),
                      provider, str(provenance.get('query') or origin.get('query') or '')[:500] or None, sha, _now()))

    def _blob(self, content, media_type):
        sha = _sha(content)
        relative = Path(sha[:2]) / sha
        target = self.blobs / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix('.partial')
            partial.write_bytes(content)
            partial.replace(target)
        return str(relative)

    @staticmethod
    def _provenance(provenance):
        kept = {key: provenance[key] for key in PROVENANCE_KEYS if provenance.get(key) not in (None, '')}
        if isinstance(kept.get('url'), str):
            kept['url'] = re.sub(r'(?i)([?&](api_?key|key|token)=)[^&]+', r'\1REDACTED', kept['url'])
        return kept

    # ---------- organization ----------

    def adjuvant_names(self):
        if self._adjuvants is None:
            self._adjuvants = []
            from ..application_config import PROJECT_ROOT
            records = PROJECT_ROOT / 'data' / 'adjuvant_resources_v4' / 'records.jsonl'
            if records.exists():
                seen = set()
                for line in records.open():
                    record = json.loads(line)
                    title = record.get('title') or ''
                    if record.get('dataset') == 'vaxjo_master' and title.lower() not in seen:
                        seen.add(title.lower())
                        self._adjuvants.append(title)
        if not hasattr(self, '_phrases'):
            self._phrases = []
            for name in self._adjuvants:
                core = [t for t in re.findall(r'[^\W_]+', name) if t.lower() not in GENERIC]
                if not core or len(''.join(core)) < 3:
                    continue
                exact = len(core) == 1 and core[0].isupper() and core[0].isalpha()
                self._phrases.append((name, ' ' + ' '.join(core if exact else [t.lower() for t in core]) + ' ', exact))
        return self._phrases

    def _tag(self, conn, paper_id, origin):
        if origin.get('project'):
            conn.execute('INSERT OR IGNORE INTO tags VALUES(?,?,?,?)',
                         (paper_id, 'project', origin['project'], str(origin.get('ref', ''))[:300]))
        row = conn.execute('SELECT title, abstract FROM papers WHERE paper_id=?', (paper_id,)).fetchone()
        tokens = re.findall(r'[^\W_]+', f"{row['title'] or ''} {row['abstract'] or ''}")
        cased, lowered = ' ' + ' '.join(tokens) + ' ', ' ' + ' '.join(tokens).lower() + ' '
        for name, phrase, exact in self.adjuvant_names():
            if phrase in (cased if exact else lowered):
                conn.execute('INSERT OR IGNORE INTO tags VALUES(?,?,?,?)',
                             (paper_id, 'adjuvant', name, 'VaxJO adjuvant name phrase in title/abstract'))

    def retag(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM tags WHERE kind='adjuvant'")
            for row in conn.execute('SELECT paper_id FROM papers').fetchall():
                self._tag(conn, row['paper_id'], {})

    def link_pack(self, pack_root):
        """Link resource-pack records that cite PubMed IDs to library papers (identifier-only if unseen)."""
        pack_root = Path(pack_root)
        manifest = json.loads((pack_root / 'manifest.json').read_text())
        pack_sha = manifest.get('records_sha256') or manifest.get('sha256') or _sha((pack_root / 'records.jsonl').read_bytes())
        links = 0
        with self._connect() as conn:
            for line in (pack_root / 'records.jsonl').open():
                record = json.loads(line)
                cited = set(re.findall(r'(?:PMID[:_ ]*|pubmed/)(\d{6,9})', json.dumps(record.get('fields', {})), re.I))
                for pmid in sorted(cited):
                    paper_id = self._paper(conn, {}, {'pmid': pmid}, 'identifier_only')
                    conn.execute('INSERT OR IGNORE INTO pack_links VALUES(?,?,?,?,?)',
                                 (paper_id, pack_sha, record['id'], record.get('dataset'), record.get('title')))
                    conn.execute('INSERT OR IGNORE INTO tags VALUES(?,?,?,?)',
                                 (paper_id, 'cited_by_pack', record.get('dataset') or '', record['id']))
                    links += 1
        return links

    # ---------- reading ----------

    def search(self, query, *, limit=10, adjuvant=None, project=None, min_text=None, max_year=None):
        terms = [t for t in re.findall(r'[A-Za-z0-9]+', query or '') if len(t) > 1][:24]
        if not terms:
            return []
        rows = []
        for joiner in (' AND ', ' OR '):
            match = joiner.join(f'"{t}"' for t in terms)
            sql = ['SELECT p.id, p.paper_id, p.section, p.locator, p.text, bm25(passages_fts) AS score,'
                   ' a.title, a.year, a.pmid, a.pmcid, a.doi, a.best_text'
                   ' FROM passages_fts JOIN passages p ON p.id = passages_fts.rowid'
                   ' JOIN papers a ON a.paper_id = p.paper_id WHERE passages_fts MATCH ?']
            params = [match]
            if adjuvant:
                sql.append("AND p.paper_id IN (SELECT paper_id FROM tags WHERE kind='adjuvant' AND tag=?)")
                params.append(adjuvant)
            if project:
                sql.append("AND p.paper_id IN (SELECT paper_id FROM tags WHERE kind='project' AND tag=?)")
                params.append(project)
            if min_text:
                sql.append('AND a.best_text IN (%s)' % ','.join('?' * len([k for k, v in TEXT_RANK.items() if v >= TEXT_RANK[min_text]])))
                params.extend(k for k, v in TEXT_RANK.items() if v >= TEXT_RANK[min_text])
            if max_year:
                sql.append('AND a.year IS NOT NULL AND CAST(a.year AS INTEGER) <= ?')
                params.append(int(max_year))
            sql.append('ORDER BY score LIMIT ?')
            params.append(int(limit))
            with self._connect() as conn:
                rows = [dict(row) for row in conn.execute(' '.join(sql), params)]
            if rows:
                break
        return rows

    def paper(self, paper_id):
        with self._connect() as conn:
            row = conn.execute('SELECT * FROM papers WHERE paper_id=?', (paper_id,)).fetchone()
            if row is None:
                found = conn.execute('SELECT paper_id FROM identifiers WHERE value=?', (paper_id,)).fetchone()
                row = found and conn.execute('SELECT * FROM papers WHERE paper_id=?', (found['paper_id'],)).fetchone()
            if row is None:
                return None
            pid = row['paper_id']
            return {**dict(row),
                    'identifiers': [dict(r) for r in conn.execute('SELECT scheme, value FROM identifiers WHERE paper_id=?', (pid,))],
                    'tags': [dict(r) for r in conn.execute('SELECT kind, tag FROM tags WHERE paper_id=? ORDER BY kind, tag', (pid,))],
                    'documents': [dict(r) for r in conn.execute('SELECT sha256, media_type, kind, provider, size, blob FROM documents WHERE paper_id=?', (pid,))],
                    'pack_links': [dict(r) for r in conn.execute('SELECT record_id, dataset, record_title FROM pack_links WHERE paper_id=?', (pid,))],
                    'sightings': conn.execute('SELECT COUNT(*) FROM sightings WHERE paper_id=?', (pid,)).fetchone()[0],
                    'passages': conn.execute('SELECT COUNT(*) FROM passages WHERE paper_id=?', (pid,)).fetchone()[0]}

    def related(self, paper_id):
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                'SELECT r.relation, r.to_paper AS paper_id, r.source, r.score, a.title, a.year FROM relations r'
                ' LEFT JOIN papers a ON a.paper_id=r.to_paper WHERE r.from_paper=? ORDER BY r.relation, r.score DESC',
                (paper_id,))]

    def stats(self):
        with self._connect() as conn:
            one = lambda sql: conn.execute(sql).fetchone()[0]
            return {'papers': one('SELECT COUNT(*) FROM papers'),
                    'by_text': {r[0]: r[1] for r in conn.execute('SELECT best_text, COUNT(*) FROM papers GROUP BY best_text')},
                    'documents': one('SELECT COUNT(DISTINCT sha256) FROM documents'),
                    'documents_by_kind': {r[0]: r[1] for r in conn.execute('SELECT kind, COUNT(DISTINCT sha256) FROM documents GROUP BY kind')},
                    'passages': one('SELECT COUNT(*) FROM passages'),
                    'sightings': one('SELECT COUNT(*) FROM sightings'),
                    'projects': one("SELECT COUNT(DISTINCT tag) FROM tags WHERE kind='project'"),
                    'adjuvant_tagged_papers': one("SELECT COUNT(DISTINCT paper_id) FROM tags WHERE kind='adjuvant'"),
                    'pack_linked_papers': one('SELECT COUNT(DISTINCT paper_id) FROM pack_links'),
                    'relations': {r[0]: r[1] for r in conn.execute('SELECT relation, COUNT(*) FROM relations GROUP BY relation')},
                    'failures': one('SELECT COUNT(*) FROM failures')}

    def catalog(self, *, per_group=8):
        """Markdown catalog grouped by adjuvant and project, regenerated from the database."""
        stats = self.stats()
        lines = ['# 文献库目录', '', f"自动生成于 {_now()}，由 `python -m adj_partner.library catalog` 重建。", '',
                 f"- 论文 {stats['papers']} 篇：" + '，'.join(f'{k} {v}' for k, v in sorted(stats['by_text'].items())),
                 f"- 文档 {stats['documents']} 份，段落 {stats['passages']} 条，出现记录 {stats['sightings']} 条",
                 f"- 带佐剂标签 {stats['adjuvant_tagged_papers']} 篇；被资源包引用 {stats['pack_linked_papers']} 篇；"
                 f"项目 {stats['projects']} 个；入库失败 {stats['failures']} 次", '']
        with self._connect() as conn:
            for kind, heading in (('adjuvant', '按佐剂'), ('project', '按项目'), ('cited_by_pack', '按引用它的资源包数据集')):
                groups = conn.execute('SELECT tag, COUNT(*) n FROM tags WHERE kind=? GROUP BY tag ORDER BY n DESC, tag',
                                      (kind,)).fetchall()
                lines += [f'## {heading}（{len(groups)} 组）', '']
                for group in groups:
                    papers = conn.execute(
                        'SELECT a.paper_id, a.title, a.year, a.best_text FROM tags t JOIN papers a ON a.paper_id=t.paper_id'
                        ' WHERE t.kind=? AND t.tag=? ORDER BY a.best_text="full_text" DESC, a.year DESC LIMIT ?',
                        (kind, group['tag'], per_group)).fetchall()
                    lines.append(f"### {group['tag']}（{group['n']} 篇）")
                    lines += [f"- `{p['paper_id']}` {p['year'] or '—'} · {p['best_text']} · {(p['title'] or '（仅标识符）')[:160]}"
                              for p in papers]
                    lines.append('')
        return '\n'.join(lines) + '\n'
