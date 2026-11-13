"""Publish reviewed plans without relabeling host edits as model responses."""
import json
from pathlib import Path
import shutil

from .research.molecule_inputs import sha
from .research.molecule_protocol import normalize_function_design, PROTOCOL_VERSION
from .research.molecule_reports import validate_plan, render_molecule, render_molecule_html, render_index


def publish_reviewed_delivery(source, output):
    from .molecule_reports import verify_delivery, _write, workflow_hash, sha_bytes
    source=Path(source).resolve();out=Path(output).resolve()
    if out==source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Publication must use a separate directory')
    if out.exists():raise ValueError('Publication exists; never overwrite prior versions')
    if not verify_delivery(source)['accepted']:raise ValueError('Incomplete or changed original delivery')
    origin=json.loads((source/'RUN_MANIFEST.json').read_text())
    if origin.get('publication'):raise ValueError('Cannot repeatedly relabel a derived publication')
    bundle=json.loads((source/'bundle.json').read_text());profiles=bundle['profiles']
    # Keep every original response, failed attempt and source receipt. Runtime
    # remains the authoritative local origin; its independently audited export
    # is included rather than pretending a second set of API runs happened.
    shutil.copytree(source,out/'generation',ignore=shutil.ignore_patterns('runtime'))
    _write(out/'bundle.json',bundle)
    identity=json.loads((source/'RUN_IDENTITY.json').read_text())
    identity.update(workflow_sha256=workflow_hash(),publication_kind='host_review_publication',protocol_version=PROTOCOL_VERSION)
    _write(out/'RUN_IDENTITY.json',identity)
    plans={};receipts={}
    for profile in profiles:
        alias=profile['alias']
        raw_directory='model-plans' if origin['receipts'][alias].get('protocol_version') else 'plans'
        raw=json.loads((source/raw_directory/f'{alias}.json').read_text())
        plan,review=normalize_function_design(validate_plan(raw,profile));plan=validate_plan(plan,profile)
        report=render_molecule(profile,plan)
        for directory,value in [('inputs',profile),('model-plans',raw),('plans',plan),('normalization',review)]:
            _write(out/directory/f'{alias}.json',value)
        _write(out/'molecules'/f'{alias}.md',report)
        _write(out/'molecules'/f'{alias}.html',render_molecule_html(profile,plan))
        plans[alias]=plan
        receipts[alias]={'alias':alias,'compound':profile['compound'],'status':'succeeded',
                         'stage':'host_review_publication','origin':origin['receipts'][alias],
                         'report_sha256':sha_bytes(report),'model_plan_sha256':review['raw_plan_sha256'],
                         'published_plan_sha256':review['published_plan_sha256'],'protocol_version':PROTOCOL_VERSION,
                         'wet_experiment_status':'not_performed'}
    md,html=render_index(profiles,plans);_write(out/'INDEX.md',md);_write(out/'index.html',html)
    manifest={**origin,**identity,'receipts':receipts,'review_status':'host_normalized_pending_independent_review',
              'publication':{'kind':'host_review_publication','protocol_version':PROTOCOL_VERSION,
                             'generation_manifest_sha256':sha(source/'RUN_MANIFEST.json'),
                             'new_model_requests':0,'actual_generation_attempts':origin['attempts'],
                             'actual_generation_failed_attempts':origin['failed_attempts'],
                             'origin_runtime':str(source/'runtime')}}
    _write(out/'RUN_MANIFEST.json',manifest)
    result=verify_delivery(out)
    if not result['accepted']:raise ValueError('Derived publication verification failed: '+str(result['issues']))
    return manifest
