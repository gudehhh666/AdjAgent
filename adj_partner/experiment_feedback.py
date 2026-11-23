"""Host CLI for local result ingestion and evidence-bound design updates. No API calls."""
import argparse
import hashlib
import json
from pathlib import Path
from .research import ResearchStore
from .storage import Store
from .tools.policy import _credential_path

FIELDS={'result_key','design_id','checkpoint_id','measurement_level','quality','context_status',
        'outcome','interpretation','context','reported_by','result_file','selector'}


def _read(path,maximum):
    path=Path(path).expanduser().absolute()
    if path.resolve()!=path or _credential_path(path) or not path.is_file():
        raise ValueError('Result input requires an explicit regular, non-credential, non-symlink file')
    with path.open('rb') as f:raw=f.read(maximum+1)
    if len(raw)>maximum:raise ValueError('Result input exceeds size limit')
    return path,raw


def import_result(research,session_id,run_id,bundle):
    if not isinstance(bundle,dict) or set(bundle) not in (FIELDS,FIELDS|{'supersedes'}):raise ValueError('Unexpected or missing experiment bundle fields')
    path,raw=_read(bundle['result_file'],20_000_000)
    from .research.records import extract
    extract(raw,bundle['selector'])  # Validate literal selection before replay or writes.
    sha=hashlib.sha256(raw).hexdigest()
    arguments={k:v for k,v in bundle.items() if k not in ('result_file','selector')}
    for existing in research.list_design_records(session_id,run_id,'experiment_result'):
        p=existing['payload']
        if p['result_key']!=bundle['result_key']:continue
        evidence=[research.get_evidence(session_id,run_id,eid) for eid in p['evidence_ids']]
        matching=(all(p[k]==v for k,v in arguments.items()) and len(evidence)==1 and
            evidence[0]['source_hash']==sha and isinstance(bundle['selector'],dict) and
            all(evidence[0]['locator'].get(k)==v for k,v in bundle['selector'].items()))
        if not matching:raise ValueError('Conflicting immutable result key; use a new key for a distinct report')
        issues=research.validate_design_record(session_id,run_id,existing['id'])
        if issues:raise ValueError('Stale experimental result bindings: '+'; '.join(issues))
        return existing
    # Validate target before publishing source material. Result classification is
    # still a reporter assertion; this adapter never infers assay success.
    target=research.get_design_record(session_id,run_id,bundle['design_id'])
    if target['kind']!='design_revision':raise ValueError('Expected design revision')
    provenance={'experiment_data':True,'text_availability':'userfile','input_sha256':sha,
                'reported_by':bundle['reported_by'],'original_filename':path.name,
                'limitations':['Local file and reporter assertions; assay execution, quality and biological interpretation not independently verified.']}
    media='application/json' if path.suffix.casefold()=='.json' else 'text/plain'
    if media=='application/json':json.loads(raw)
    else:raw.decode('utf-8')
    artifact=research.store.put_artifact(raw,path.name,media,provenance,session_id=session_id,run_id=run_id)
    src=research.register_source(session_id,run_id,artifact_id=artifact['id'],kind='userfile',
        title=path.name,provenance=provenance,idempotency_key='experiment:'+run_id+':'+sha+':'+hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest())
    evidence=next((e for e in research.list_evidence(session_id,run_id)
        if e['source_id']==src['id'] and e['source_hash']==sha and
        all(e['locator'].get(k)==v for k,v in bundle['selector'].items())),None)
    evidence=evidence or research.select_evidence(session_id,run_id,src['id'],bundle['selector'])
    return research.record_experiment_result(session_id,run_id,evidence_ids=[evidence['id']],**arguments)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,required=True);parser.add_argument('--session-id',required=True)
    parser.add_argument('--run-id',required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    imp=sub.add_parser('import');imp.add_argument('--input',type=Path,required=True)
    state=sub.add_parser('state');state.add_argument('--design-id',required=True)
    revise=sub.add_parser('revise');revise.add_argument('--design-id',required=True)
    revise.add_argument('--result-ids',nargs='+',required=True);revise.add_argument('--design',type=Path,required=True)
    args=parser.parse_args(argv)
    research=ResearchStore(Store(args.state))
    try:
        if args.command=='import':
            path,raw=_read(args.input,100_000);bundle=json.loads(raw)
            if not Path(bundle['result_file']).is_absolute():bundle['result_file']=str(path.parent/bundle['result_file'])
            value=import_result(research,args.session_id,args.run_id,bundle)
        elif args.command=='state':value=research.experiment_state(args.session_id,args.run_id,design_id=args.design_id)
        else:
            _,raw=_read(args.design,1_000_000)
            value=research.revise_design_from_experiments(args.session_id,args.run_id,design_id=args.design_id,
                                                        result_ids=args.result_ids,design=json.loads(raw))
        print(json.dumps(value,ensure_ascii=False,indent=2));return 0
    except (KeyError,ValueError,TypeError,PermissionError,OSError) as exc:
        print(json.dumps({'error':research.store.redact(str(exc))},ensure_ascii=False));return 1


if __name__=='__main__':raise SystemExit(main())
