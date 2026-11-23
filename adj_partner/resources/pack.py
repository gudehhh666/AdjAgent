"""Offline lookup of frozen source records; no scientific inference or network.

Hashes establish snapshot identity, not source truth. The manifest is host
configuration, never supplied by the model. Missing values stay missing.
"""
from copy import deepcopy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from ..tools.catalog import canonical_json


def _tokens(value):
    return set(re.findall(r'[\w]+', value.casefold(), re.UNICODE))


class ResourcePack:
    def __init__(self, root):
        self.root = Path(root).expanduser().absolute()
        if self.root.resolve() != self.root:
            raise ValueError('Resource pack cannot use symlink paths')
        raw = self._bytes('manifest.json', 1_000_000)
        self.manifest_hash = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        canonical_json(self.manifest)
        if self.manifest.get('schema_version') != 1:
            raise ValueError('Unsupported resource pack schema')
        self.datasets = {}
        for d in self.manifest.get('datasets', []):
            if d['id'] in self.datasets:
                raise ValueError('Duplicate dataset identity')
            for key in ['source_url', 'retrieved_at', 'license', 'evidence_scope']:
                if not isinstance(d.get(key), str) or not d[key]:
                    raise ValueError('Missing dataset provenance: '+key)
            if not re.fullmatch('[a-f0-9]{64}', d.get('raw_sha256', '')):
                raise ValueError('Missing raw source hash')
            for name, source in d.get('source_files', {}).items():
                if source.get('raw_filename') != name or not re.fullmatch('[a-f0-9]{64}', source.get('raw_sha256', '')):
                    raise ValueError('Invalid per-file source identity')
                if not all(isinstance(source.get(k), str) and source[k] for k in ('source_url', 'retrieved_at')):
                    raise ValueError('Missing per-file source provenance')
            self.datasets[d['id']] = d
        self.records = {}
        self._search = {}
        for record in self._rows('records.jsonl'):
            rid = record['id']
            if not isinstance(rid, str) or not rid or len(rid) > 256:
                raise ValueError('Invalid record identity')
            if rid in self.records:
                raise ValueError('Duplicate record identity: '+rid)
            if record['dataset'] not in self.datasets:
                raise ValueError('Unknown record dataset')
            if not isinstance(record.get('title'), str) or not isinstance(record.get('fields'), dict):
                raise ValueError('Missing record fields/title')
            if not isinstance(record.get('source_locator'), dict):
                raise ValueError('Missing source locator')
            source_files = self.datasets[record['dataset']].get('source_files')
            if source_files is not None and record['source_locator'].get('raw_filename') not in source_files:
                raise ValueError('Record has no bound source file')
            canonical_json(record)
            self.records[rid] = record
            text = canonical_json(record).casefold()
            exact = {rid.casefold(), record['title'].casefold(),
                     *[a.casefold() for a in record.get('aliases', [])]}
            self._search[rid] = (_tokens(text), exact, text)
        self._rank_tokens = {rid: _tokens(canonical_json({'id':r['id'],'title':r['title'], 'aliases':r.get('aliases',[]),'fields':r['fields']})) for rid,r in self.records.items()}
        self._term_frequency = Counter(t for tokens in self._rank_tokens.values() for t in tokens)
        self.edges = self._rows('relations.jsonl')
        for edge in self.edges:
            if edge.get('subject') not in self.records or edge.get('source_record') not in self.records:
                raise ValueError('Relation missing source record')
            if not all(isinstance(edge.get(k), str) and edge[k] for k in ['predicate', 'object', 'qualifier']):
                raise ValueError('Malformed relation')

    def _bytes(self, filename, maximum):
        path = self.root / filename
        if path.is_symlink() or path.resolve() != path:
            raise ValueError('Resource pack cannot use symlink paths')
        with path.open('rb') as handle:
            raw = handle.read(maximum+1)
        if len(raw) > maximum:
            raise ValueError('Resource pack size limit exceeded')
        return raw

    def _rows(self, filename):
        expected = self.manifest['files'][filename]
        raw = self._bytes(filename, 64_000_000)
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('Resource pack integrity mismatch: '+filename)
        rows = []
        for line in raw.splitlines():
            if len(line) > 100_000 or len(rows) >= 50_000:
                raise ValueError('Resource pack record limit exceeded')
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Resource record must be an object')
            canonical_json(row)
            rows.append(row)
        return rows

    @staticmethod
    def _limit(limit):
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError('limit must be an integer in 1..20')

    def search(self, query=None, *, kind=None, dataset=None, filters=None, limit=5, match_mode="all", focus=None):
        self._limit(limit)
        if focus is not None and (not isinstance(focus,str) or len(focus)>200):raise ValueError("Invalid focus")
        if match_mode not in {"all", "ranked"}:raise ValueError("Unknown match_mode")
        if dataset is not None and dataset not in self.datasets:
            raise ValueError('Unknown dataset')
        selected = [r for r in self.records.values() if dataset is None or r['dataset'] == dataset]
        allowed_kinds = {r['kind'] for r in selected}
        if kind is not None and kind not in allowed_kinds:
            raise ValueError('kind incompatible with dataset; allowed: '+', '.join(sorted(allowed_kinds)))
        if filters is not None:
            if dataset is None:
                raise ValueError('Exact filters require dataset')
            if not isinstance(filters, dict) or not 1 <= len(filters) <= 8:
                raise ValueError('filters must contain 1..8 exact scalar fields')
            available = {k for r in selected for k in r['fields']}
            for key, value in filters.items():
                if not isinstance(key, str) or key not in available:
                    raise ValueError('Unknown filter field: '+str(key))
                if type(value) not in (str, int, float, bool, type(None)) or (isinstance(value, str) and len(value) > 500) or (type(value) is float and not math.isfinite(value)):
                    raise ValueError('Filter values must be finite JSON scalars of at most 500 characters')
        if query is None and not filters:
            raise ValueError('query or exact filters required')
        if query is not None and (not isinstance(query, str) or not query.strip() or len(query) > 500):
            raise ValueError('query must contain 1..500 characters')
        query = query.casefold().strip() if query is not None else ''
        tokens = _tokens(query)
        if query and not tokens:
            raise ValueError('query needs searchable text')
        matches = []; retrieval_matches = {}
        rank_tokens = tokens - {'how','what','which','is','are','the','a','an','of','for','to','and','or','in','on','with','does','do','can','by','from','this','that'}
        for rid, r in self.records.items():
            if (kind is not None and r['kind'] != kind) or (dataset is not None and r['dataset'] != dataset):
                continue
            if filters and any(k not in r['fields'] or type(r['fields'][k]) is not type(v) or r['fields'][k] != v for k, v in filters.items()):
                continue
            haystack, exact, text = self._search[rid]
            if not query:
                score = 100
            elif query in exact:
                score = 100
            elif match_mode == 'ranked':
                overlap = rank_tokens & self._rank_tokens[rid]
                if not overlap:continue
                title_tokens = _tokens(r['title']+' '+ ' '.join(r.get('aliases',[])))
                score = sum(math.log(1 + len(self.records)/(1+self._term_frequency[t])) * (2 if t in title_tokens else 1) for t in overlap)
            elif tokens <= haystack:
                score = 10 + int(query in text)
            else:
                continue
            if focus:
                score += sum(math.log(1+len(self.records)/(1+self._term_frequency[t]))*5 for t in _tokens(focus) & self._rank_tokens[rid])
            if match_mode == 'ranked':
                retrieval_matches[rid] = {'matched_terms':sorted(rank_tokens & self._rank_tokens[rid]),'query_terms':sorted(rank_tokens),'score':round(score,6)}
            matches.append((score, rid))
        matches.sort(key=lambda x: (-x[0], x[1]))
        result = {'hit_count': len(matches), 'returned': min(len(matches), limit),
                'applied_filters': deepcopy(filters or {}),
                'pack_sha256': self.manifest_hash,
                'results': [self.read(rid) for _, rid in matches[:limit]],
                'limitations': ['Lexical lookup of a frozen pack, not exhaustive literature search or efficacy ranking.']}

        if match_mode == 'ranked':
            result.update(match_mode=match_mode,retrieval_matches={rid:retrieval_matches[rid] for _,rid in matches[:limit]})
            result['limitations'].append('Ranked partial lexical matches are discovery candidates, not evidence that the question is answered; read exact fields and check context.')
        return result

    def read(self, record_id):
        record = deepcopy(self.records[record_id])
        dataset = deepcopy(self.datasets[record['dataset']])
        source_files = dataset.pop('source_files', None)
        if source_files is not None:
            dataset.update(source_files[record['source_locator']['raw_filename']])
        record['provenance'] = {**dataset, 'pack_sha256': self.manifest_hash,
                                'text_availability': 'database_record',
                                'source_locator': record['source_locator']}
        return record

    def neighbors(self, record_id, *, direction='outgoing', predicate=None, limit=10):
        self._limit(limit)
        if record_id not in self.records:
            raise KeyError(record_id)
        if direction not in {'incoming', 'outgoing'}:
            raise ValueError('Invalid relation direction')
        key = 'subject' if direction == 'outgoing' else 'object'
        edges = [e for e in self.edges if e[key] == record_id and (predicate is None or e['predicate'] == predicate)]
        return {'record_id': record_id, 'hit_count': len(edges), 'relations': deepcopy(edges[:limit]),
                'pack_sha256': self.manifest_hash,
                'limitations': ['Explicit ontology or source-membership relations and qualifiers only. No OWL reasoning, causal inference or efficacy validation.']}
