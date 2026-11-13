"""Current-run sequence snapshots: inspect literal fields and reselect without rereading inputs."""
from collections import Counter
from copy import deepcopy
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import PurePosixPath

from ..contracts import ToolResult
from ..tools import ToolSpec
from ..tools.catalog import canonical_json, validator
from .sequence_inventory import SCHEMA as INVENTORY_SCHEMA, _stages, _add, STAGES

MAX_INVENTORY_BYTES = 32 * 1024 * 1024
QUERY_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['artifact_id', 'member_glob', 'scope_note'],
    'properties': {
        'artifact_id': {'type': 'string', 'pattern': '^artifact_[A-Za-z0-9_-]+$', 'maxLength': 128},
        **{key: deepcopy(INVENTORY_SCHEMA['properties'][key]) for key in ('member_glob','file_stems','scope_note','unresolved_scope')},
        'source_indices': {'type':'array','maxItems':8,'uniqueItems':True,
                           'items':{'type':'integer','minimum':0,'maximum':7},
                           'description':'Original zero-based source indices; omit for all, [] for none.'},
    },
}


def register_sequence_query(catalog):
    catalog.register(ToolSpec('sequence_inventory_query',
        'Reselect actual records from a current-run sequence_inventory artifact; no files must be copied again. '
        'Directory and filename_stem examples are literal fields, distinct from FASTA record_id. '
        'Use member_glob for path conditions and file_stems for exact filename stems. '
        'Snapshot data only, no biological inference; always reuse the original inventory artifact_id.',
        QUERY_SCHEMA), sequence_inventory_query)


def directory_facets(inputs):
    groups = {}
    for source in inputs:
        for member in source['members']:
            if member['kind'] != 'fasta':
                continue
            p = PurePosixPath(member['path']); key = str(p.parent)
            group = groups.setdefault(key, {'path':key,'files':0,'records':0,'stems':set(),'ids':set()})
            group['files'] += 1; group['records'] += len(member['records'])
            group['stems'].add(p.stem)
            group['ids'].update(r['record_id'] for r in member['records'])
    result = []
    for path, group in sorted(groups.items()):
        result.append({k:group[k] for k in ('path','files','records')} |
                      {'filename_stem_examples':sorted(group['stems'])[:2],
                       'record_id_examples':sorted(group['ids'])[:2]})
    return result


def compact_result(data, artifact):
    """Keep literal choices and totals visible; full artifact retains omitted details."""
    data = deepcopy(data)
    result = ToolResult(data=data, artifacts=[{k:artifact[k] for k in ('id','name','sha256','session_id','run_id')}])
    data.setdefault('directories_omitted', 0)
    while len(canonical_json(result.to_dict()).encode()) > 2200:
        if data.get('inputs'):
            data['inputs'].pop();data['inputs_omitted'] += 1
        elif data.get('directories'):
            data['directories'].pop();data['directories_omitted'] += 1
        else:
            raise ValueError('Sequence summary exceeds bounded response size')
    return result


def sequence_inventory_query(context, artifact_id, member_glob, scope_note,
                             file_stems=None, source_indices=None, unresolved_scope=None):
    context.check_cancelled()
    args = dict(artifact_id=artifact_id,member_glob=member_glob,scope_note=scope_note)
    for key,value in [('file_stems',file_stems),('source_indices',source_indices),('unresolved_scope',unresolved_scope)]:
        if value is not None:args[key]=value
    validator(QUERY_SCHEMA).validate(args)
    meta = next((x for x in context.store.artifacts(context.session_id,context.run_id) if x['id']==artifact_id),None)
    if meta is None or meta['provenance'].get('provider') != 'sequence_inventory':
        raise PermissionError('Only a host-created sequence_inventory from this run is queryable')
    if meta['size'] > MAX_INVENTORY_BYTES:
        raise ValueError('Inventory snapshot exceeds query byte limit')
    raw = context.store.read_artifact(artifact_id,max_bytes=MAX_INVENTORY_BYTES)
    if hashlib.sha256(raw).hexdigest() != meta['sha256']:
        raise ValueError('Inventory snapshot hash mismatch')
    doc = json.loads(raw)
    if doc.get('format_version') != 1:
        raise ValueError('Unsupported inventory snapshot version')
    valid_indices = {s['source_index'] for s in doc['inputs']}
    if source_indices is not None and not set(source_indices) <= valid_indices:
        raise ValueError('Source index absent from original inventory')
    sources = [s for s in doc['inputs'] if source_indices is None or s['source_index'] in source_indices]
    totals, selected, ids = _stages(), [], Counter()
    stems = None if file_stems is None else set(file_stems)
    for source in sources:
        for index, member in enumerate(source['members']):
            if index % 128 == 0:context.check_cancelled()
            if member['kind'] != 'fasta':continue
            path = PurePosixPath(member['path']); match = fnmatchcase(member['path'],member_glob)
            keep = match and (stems is None or path.stem in stems)
            records = member['records']
            counts = dict(files=1,records=len(records),**{k:sum(r[k] for r in records) for k in ('residues','gaps','stops')})
            for stage, include in zip(STAGES,(True,match,keep)):
                if include:_add(totals[stage],counts)
            if keep:
                for r in records:
                    selected.append({k:r[k] for k in ('ordinal','record_id','residues','gaps','stops')} |
                                    dict(source_index=source['source_index'],member_path=member['path'],
                                         filename_stem=path.stem,member_sha256=member['sha256']))
                    ids[r['record_id']] += 1
    duplicates = dict(duplicate_record_ids=sum(n>1 for n in ids.values()),
                      records_with_duplicate_ids=sum(n for n in ids.values() if n>1))
    facets = directory_facets(sources)
    derived = dict(format_version=1,parent=dict(artifact_id=artifact_id,sha256=meta['sha256']),
                   scope=args,counts=totals,selected_records=selected,directories=facets,**duplicates,
                   limitations=['Snapshot re-selection only; input files are not reread. No biological scope inference.',
                                'file_stems selects filenames, not FASTA headers. Repeated IDs remain separate records.'])
    context.check_cancelled()
    artifact = context.store.put_artifact(canonical_json(derived),'sequence-selection.json','application/json',
        {'provider':'sequence_inventory_query','parent_artifact_id':artifact_id,'parent_sha256':meta['sha256']},
        session_id=context.session_id,run_id=context.run_id)
    return compact_result(dict(counts=totals,artifact_id=artifact['id'],inventory_artifact_id=artifact_id,
        directories=facets[:6],directories_omitted=max(0,len(facets)-6),**duplicates,
        counting_unit='FASTA records; file_stems uses filenames',unresolved_scope_count=len(unresolved_scope or [])),artifact)
