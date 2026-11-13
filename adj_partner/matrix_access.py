"""Bounded, permission-checked GEO sample metadata access, without assay inference.

The reader exposes source strings and generic grouping counts. It has no study
IDs, preferred biological fields, cohort assumptions or reference answers.
"""

from collections import Counter
import csv
import gzip
import io
from pathlib import Path

from .contracts import ToolResult
from .native.safe_files import read_input
from .tools import ToolSpec
from .tools.catalog import canonical_json


MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 16 * 1024 * 1024
SUMMARY_BYTES = 1800
SCOPE = ('Sample metadata only; exact decoded cells (credential redaction may apply). '
         'Groups are metadata pairing candidates, not expression values, raw data, QC or causal validation.')


def parse_matrix(content):
    """Parse only GEO sample metadata; repeated rows have distinct stable IDs."""
    if len(content) > MAX_DECOMPRESSED_BYTES:
        raise ValueError('Matrix exceeds decompressed byte limit')
    text = content.decode('utf-8-sig')
    rows, body, metadata_rows = [], False, 0
    for row in csv.reader(io.StringIO(text), delimiter='\t', strict=True):
        if not row:
            continue
        if row[0] == '!series_matrix_table_begin':
            body = True
            break
        if not row[0].startswith(('!Series_', '!Sample_')) or len(row) < 2:
            raise ValueError('Expected GEO series-matrix metadata rows')
        metadata_rows += 1
        if metadata_rows > 1000:
            raise ValueError('Metadata exceeds 1000 rows')
        if row[0].startswith('!Sample_'):
            rows.append(row)
    accession_rows = [row[1:] for row in rows if row[0] == '!Sample_geo_accession']
    if (len(accession_rows) != 1 or not accession_rows[0]
            or any(not value.strip() for value in accession_rows[0])):
        raise ValueError('Expected one nonempty sample accession row')
    accessions = accession_rows[0]
    if len(accessions) > 50000:
        raise ValueError('Metadata exceeds 50000 sample columns')
    fields, columns, occurrences = [], {}, Counter()
    for i, row in enumerate(rows):
        if len(row) != len(accessions) + 1:
            raise ValueError('Sample metadata width does not match accessions')
        name, values = row[0], row[1:]
        occurrences[name] += 1
        nonempty = [x for x in values if x.strip()]
        # Only a uniform literal prefix is a label. Mixed characteristic rows
        # remain one original field; do not silently merge or reinterpret cells.
        prefixes = {x.split(':', 1)[0] for x in nonempty if ':' in x}
        label = (next(iter(prefixes)) if nonempty and len(prefixes) == 1
                 and all(':' in x for x in nonempty) and '' not in prefixes else None)
        identifier = f'f{i}'
        fields.append({'id': identifier, 'name': name, 'occurrence': occurrences[name],
                       'label': label, 'type': 'string', 'nonempty': len(nonempty),
                       'unique_nonempty': len(set(nonempty))})
        columns[identifier] = values
    return {'sample_count': len(accessions), 'duplicate_accessions': len(accessions)-len(set(accessions)),
            'matrix_body_present': body, 'fields': fields, 'columns': columns}


def matrix_page(matrix, sha256, *, mode='schema', fields=None, offset=0, limit=20):
    """Return whole entries under a byte cap; never cut scientific field values."""
    if mode not in ('schema', 'rows', 'counts'):
        raise ValueError('Unknown metadata mode')
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError('Invalid metadata page bounds')
    fields = fields or []
    if (not isinstance(fields, list) or len(set(fields)) != len(fields)
            or len(fields) > 4 or any(f not in matrix['columns'] for f in fields)):
        raise ValueError('Select up to four distinct existing field IDs from schema')
    if mode == 'schema':
        if fields:
            raise ValueError('Schema mode does not select fields')
        entries = matrix['fields']
    else:
        if not fields:
            raise ValueError('Read schema, then select existing fields')
        rows = list(zip(*(matrix['columns'][f] for f in fields)))
        entries = ([list(row) for row in rows] if mode == 'rows' else
                   [{'values': list(values), 'count': count}
                    for values, count in sorted(Counter(rows).items())])
    if offset > len(entries):
        raise ValueError('Offset exceeds total entries')
    page = {'format': 'geo_sample_metadata', 'sha256': sha256, 'mode': mode,
            'sample_count': matrix['sample_count'], 'duplicate_accessions': matrix['duplicate_accessions'],
            'field_count': len(matrix['fields']), 'selected_fields': fields,
            'total_entries': len(entries), 'offset': offset, 'returned': 0,
            'next_offset': None, 'items': [], 'expression_values_inspected': False,
            'matrix_body_present': matrix['matrix_body_present'], 'scope': SCOPE}
    for entry in entries[offset:offset+limit]:
        candidate = {**page, 'items': page['items']+[entry], 'returned': page['returned']+1}
        end = offset + candidate['returned']
        candidate['next_offset'] = end if end < len(entries) else None
        if len(canonical_json(candidate).encode()) > SUMMARY_BYTES:
            break
        page = candidate
    if offset < len(entries) and not page['items']:
        raise ValueError('One complete entry cannot fit the metadata page; select fewer fields or use code_interpreter')
    return page


def register_matrix_access(service):
    def inspect(context, path, mode='schema', fields=None, offset=0, limit=20, expected_sha256=None):
        context.check_cancelled()
        bound = service.native._bind(context)
        item = read_input(context, path, maximum_bytes=MAX_INPUT_BYTES)
        if expected_sha256 is not None and item.sha256 != expected_sha256:
            raise ValueError('Input SHA256 differs from expected snapshot')
        if item.content.startswith(b'\x1f\x8b'):
            with gzip.GzipFile(fileobj=io.BytesIO(item.content)) as compressed:
                content = compressed.read(MAX_DECOMPRESSED_BYTES + 1)
            if len(content) > MAX_DECOMPRESSED_BYTES:
                raise ValueError('Matrix exceeds decompressed byte limit')
        else:
            content = item.content
        context.check_cancelled()
        # Redact full strings before any projection so known secrets cannot be
        # exposed by a shortened prefix. SHA always identifies original bytes.
        content = context.store.redact(content.decode('utf-8-sig')).encode('utf-8')
        matrix = parse_matrix(content)
        summary = matrix_page(matrix, item.sha256, mode=mode, fields=fields, offset=offset, limit=limit)
        provenance = {'inputs': [item.manifest()], 'text_availability': 'userfile',
                      'parser': 'geo_matrix_metadata_v1', 'mode': mode,
                      'selected_fields': summary['selected_fields'], 'offset': offset,
                      'returned': summary['returned'], 'limitations': [SCOPE]}
        # Preserve exactly the observed page, not an unbounded hidden expression
        # matrix. The original input hash/path remain available in provenance.
        artifact = context.store.put_artifact(canonical_json(summary), 'geo-metadata-page.json',
                    'application/json', provenance, session_id=context.session_id, run_id=context.run_id)
        source = bound.research.register_source(context.session_id, context.run_id, artifact_id=artifact['id'],
                    kind='userfile', title=Path(item.path).name+' metadata', provenance=provenance,
                    idempotency_key='matrix-page:'+context.call_id)
        result = ToolResult(data={'source_id': source['id'], 'summary': summary})
        if len(canonical_json(result.to_dict()).encode()) >= 2400:
            raise ValueError('Metadata page envelope exceeds model-visible limit')
        return result

    schema = {'type': 'object', 'additionalProperties': False, 'required': ['path'], 'properties': {
        'path': {'type': 'string', 'minLength': 1, 'maxLength': 4096},
        'mode': {'enum': ['schema', 'rows', 'counts']},
        'fields': {'type': 'array', 'items': {'type': 'string', 'pattern': '^f[0-9]+$'},
                   'maxItems': 4, 'uniqueItems': True},
        'offset': {'type': 'integer', 'minimum': 0},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 50},
        'expected_sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}}}
    service.catalog.register(ToolSpec('geo_matrix_metadata',
        'Read permitted local GEO series-matrix txt/gzip SAMPLE METADATA. Start with schema to discover exact field IDs, names, labels and counts; follow next_offset for remaining fields. Then select up to four existing fields for rows or counts (cross-tab groups); paginate to next_offset=null. Pass expected_sha256 to pin a snapshot. Exact strings retained, repeated keys distinct. Metadata pairing candidates only: no expression values, QC, biological or causal validation.',
        schema, version='1'), inspect)
