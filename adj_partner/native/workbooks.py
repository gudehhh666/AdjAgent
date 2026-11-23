"""Bounded, read-only XLSX/ZIP structure entry; no scientific selection or execution."""
import hashlib
import io
import math
from pathlib import Path
import zipfile

from ..contracts import ToolResult
from ..tools import ToolSpec
from ..tools.catalog import canonical_json, validator
from .safe_files import read_input
from .sequence_inventory import _member_path

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_WORKBOOK_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 10000
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['path'], 'properties': {
    'path': {'type': 'string', 'minLength': 1, 'maxLength': 4096},
    'member': {'type': 'string', 'minLength': 1, 'maxLength': 4096,
               'description': 'Exact ZIP member path to one XLSX; omit to list archive members.'},
    'member_prefix': {'type': 'string', 'maxLength': 1024,
                      'description': 'Literal prefix for archive listing only; not a glob.'},
    'expected_sha256': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
    'sheet': {'type': 'string', 'minLength': 1, 'maxLength': 256},
    'row_start': {'type': 'integer', 'minimum': 1, 'maximum': 5000},
    'column_start': {'type': 'integer', 'minimum': 1, 'maximum': 256},
    'rows': {'type': 'integer', 'minimum': 1, 'maximum': 20},
    'columns': {'type': 'integer', 'minimum': 1, 'maximum': 256},
}}
LIMITATIONS = ['Literal bounded preview, not complete table/sample coverage or scientific validation.',
               'No header, biological group or missing-value rule is inferred. Formulas are not executed; caches may be absent or stale.']


def register_workbooks(catalog):
    catalog.register(ToolSpec('workbook_inspect',
        'Inspect authorized XLSX or ZIP-of-XLSX structure without extraction. Omit member to list ZIP; '
        'select an exact member and sheet for coordinate previews. Ignores unreliable declared dimensions. '
        'row_start/column_start select header/data windows; rows and columns bound the preview. '
        'Full selected cells and source/member hashes go to an artifact; next_cell continues a shortened preview. '
        'For calculations use code_interpreter with zipfile and io.BytesIO, or its private tmp/; '
        'read copied inputs/, write outputs/analysis_result.json. No task-specific filtering is performed.',
        SCHEMA), workbook_inspect)


def _zip(content, context, maximum):
    """Validate all member identities and CRCs before exposing any selection."""
    z = zipfile.ZipFile(io.BytesIO(content))
    try:
        infos = z.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ValueError('ZIP exceeds member count limit')
        seen, total, entries = set(), 0, []
        for info in infos:
            context.check_cancelled()
            name = _member_path(info, seen)
            total += info.file_size
            if total > maximum:
                raise ValueError('ZIP exceeds expanded byte limit')
            if info.is_dir():
                continue
            digest, size = hashlib.sha256(), 0
            with z.open(info) as handle:
                while chunk := handle.read(65536):
                    context.check_cancelled()
                    size += len(chunk)
                    if size > info.file_size or size > maximum:
                        raise ValueError('ZIP member exceeds declared byte limit')
                    digest.update(chunk)
            if size != info.file_size:
                raise ValueError('ZIP member size mismatch')
            entries.append({'path': name, 'bytes': size, 'sha256': digest.hexdigest()})
        return z, entries
    except BaseException:
        z.close()
        raise


def _value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Non-finite workbook cell')
        return value
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return str(value)


def _preview(content, context, sheet, row_start, column_start, rows, columns):
    # XLSX is itself ZIP; bound expansion before the XML reader materializes strings/styles.
    z, entries = _zip(content, context, min(MAX_EXPANDED_BYTES, MAX_WORKBOOK_BYTES))
    try:
        names = {m['path'] for m in entries}
        if not {'[Content_Types].xml', 'xl/workbook.xml'} <= names:
            raise ValueError('Selected member is not an XLSX workbook')
        if any(n.lower().endswith('vbaproject.bin') for n in names):
            raise ValueError('Macro-enabled workbooks are unsupported')
        # openpyxl may use lxml for some parts despite DEFUSEDXML availability.
        # Validate every XML part with explicit DTD/entity denial first.
        from defusedxml.ElementTree import iterparse
        nodes = 0
        for name in sorted(names):
            if not name.lower().endswith(('.xml', '.rels')):
                continue
            with z.open(name) as handle:
                for _, element in iterparse(handle, events=('end',), forbid_dtd=True,
                                             forbid_entities=True, forbid_external=True):
                    nodes += 1
                    if nodes > 1000000:
                        raise ValueError('Workbook exceeds XML node limit')
                    if nodes % 1024 == 0: context.check_cancelled()
                    element.clear()
    finally:
        z.close()
    from openpyxl import load_workbook
    from openpyxl.xml import DEFUSEDXML
    if not DEFUSEDXML:
        raise ValueError('Hardened XML reader is unavailable')
    wb = cached = None
    try:
        wb = load_workbook(io.BytesIO(content), read_only=True, data_only=False, keep_links=False)
        cached = load_workbook(io.BytesIO(content), read_only=True, data_only=True, keep_links=False)
        if len(wb.sheetnames) > 128 or any(len(n) > 256 for n in wb.sheetnames):
            raise ValueError('Workbook exceeds sheet count/name limit')
        name = sheet if sheet is not None else wb.sheetnames[0]
        if name not in wb.sheetnames:
            raise ValueError('Unknown sheet; inspect returned sheet names')
        ws, cs = wb[name], cached[name]
        dimensions = [ws.max_row, ws.max_column]
        ws.reset_dimensions(); cs.reset_dimensions()
        window = dict(min_row=row_start, max_row=row_start + rows - 1,
                      min_col=column_start, max_col=column_start + columns - 1)
        cells, mismatch = [], False
        from openpyxl.utils import get_column_letter
        for row_num, (row, cache) in enumerate(zip(ws.iter_rows(**window), cs.iter_rows(**window), strict=True), row_start):
            context.check_cancelled()
            for col_num, (cell, cv) in enumerate(zip(row, cache, strict=True), column_start):
                v = _value(cell.value)
                item = {'coordinate': f'{get_column_letter(col_num)}{row_num}', 'value': v, 'type': cell.data_type}
                if isinstance(v, str) and len(v) > 1024:
                    item.update(value=v[:1024], value_truncated=True, original_characters=len(v))
                if cell.data_type == 'f':
                    item.update(cached_value=_value(cv.value), cache_present=cv.value is not None)
                cells.append(item)
                if v is not None and all(d is not None for d in dimensions):
                    mismatch |= row_num > dimensions[0] or col_num > dimensions[1]
        return {'mode': 'workbook_preview', 'sheet_names': wb.sheetnames, 'sheet': name,
                'declared_dimensions': dimensions, 'observed_nonempty_outside_declared_dimensions': mismatch,
                'whole_sheet_inspected': False, 'window': window, 'cells': cells}
    finally:
        if wb is not None: wb.close()
        if cached is not None: cached.close()


def workbook_inspect(context, path, member=None, member_prefix='', expected_sha256=None,
                     sheet=None, row_start=1, column_start=1, rows=3, columns=40):
    arguments = dict(path=path, member_prefix=member_prefix, row_start=row_start,
                     column_start=column_start, rows=rows, columns=columns)
    arguments.update({k:v for k,v in dict(member=member, expected_sha256=expected_sha256, sheet=sheet).items() if v is not None})
    validator(SCHEMA).validate(arguments)
    if rows * columns > 1024 or column_start + columns - 1 > 256 or row_start + rows - 1 > 5000:
        raise ValueError('Preview exceeds coordinate/cell limit')
    item = read_input(context, path, maximum_bytes=MAX_INPUT_BYTES)
    if expected_sha256 is not None and item.sha256 != expected_sha256:
        raise ValueError('Source hash changed; inspect stable input again')
    suffix = Path(item.path).suffix.lower()
    if suffix == '.zip':
        z, entries = _zip(item.content, context, MAX_EXPANDED_BYTES)
        try:
            if member is None:
                if sheet is not None:
                    raise ValueError('Select an exact workbook member before a sheet')
                for entry in entries: entry['selected'] = entry['path'].startswith(member_prefix)
                doc = {'mode':'archive_inventory', 'member_prefix':member_prefix, 'members':entries}
            else:
                if member_prefix:
                    raise ValueError('member_prefix applies only to archive listing')
                selected = next((m for m in entries if m['path'] == member), None)
                if selected is None or Path(member).suffix.lower() != '.xlsx':
                    raise ValueError('Select an existing exact XLSX member')
                doc = _preview(z.read(member), context, sheet, row_start, column_start, rows, columns)
                doc['member'] = selected
        finally:
            z.close()
    elif suffix == '.xlsx' and member is None and not member_prefix:
        doc = _preview(item.content, context, sheet, row_start, column_start, rows, columns)
    else:
        raise ValueError('Use XLSX or ZIP with an exact XLSX member')
    doc.update(input=item.manifest(), limitations=LIMITATIONS)
    doc = context.store.redact(doc)
    encoded = canonical_json(doc)
    if len(encoded.encode()) > 2 * 1024 * 1024:
        raise ValueError('Selected preview exceeds artifact limit; request fewer cells')
    context.check_cancelled()
    artifact = context.store.put_artifact(encoded, 'workbook-inspection.json', 'application/json',
        {'provider': 'workbook_inspect', 'inputs': [item.manifest()], 'limitations': LIMITATIONS},
        session_id=context.session_id, run_id=context.run_id)
    summary = {'mode': doc['mode'], 'source_sha256': item.sha256}
    if doc['mode'] == 'archive_inventory':
        selected = [m for m in doc['members'] if m['selected']]
        summary.update(total_members=len(doc['members']), selected_members=len(selected),
                       members=[{'path':m['path'][:160], 'path_truncated':len(m['path'])>160,
                                 'bytes':m['bytes']} for m in selected[:8]])
        preview = summary['members']
    else:
        summary.update(sheet=doc['sheet'][:80], sheet_name_truncated=len(doc['sheet'])>80,
                       sheet_names=[n[:40] for n in doc['sheet_names'][:4]],
                       declared_dimensions=doc['declared_dimensions'],
                       dimension_mismatch_observed=doc['observed_nonempty_outside_declared_dimensions'],
                       whole_sheet_inspected=False,
                       cells=[[c['coordinate'], c['value'], c['type']] for c in doc['cells']])
        preview = summary['cells']
        # Formula caches and long strings remain available in the bounded full artifact.
        for cell in preview:
            if isinstance(cell[1],str) and len(cell[1])>120: cell[1]=cell[1][:120]+'[truncated]'
    minimal = {k:artifact[k] for k in ('id','name','sha256','session_id','run_id')}
    result = ToolResult(data={'summary':summary,'artifact_id':artifact['id']}, artifacts=[minimal])
    while True:
        if doc['mode'] == 'archive_inventory':
            summary['omitted_members'] = len(selected)-len(preview)
        else:
            summary['omitted_cells'] = len(doc['cells'])-len(preview)
            summary['next_cell'] = doc['cells'][len(preview)]['coordinate'] if summary['omitted_cells'] else None
        if len(canonical_json(result.to_dict()).encode()) <= 2100:
            return result
        if not preview:
            raise ValueError('Metadata too large for compact response')
        preview.pop()
