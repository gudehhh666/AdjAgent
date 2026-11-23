"""Frozen vaccine-adjuvant context slices. Annotations are not measured efficacy."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import parse_qs, urlsplit

from .pack import ResourcePack
from .support import support_records
from ..tools.catalog import canonical_json

ANNOTATIONS={'FUNCTION','CATALYTIC ACTIVITY','COFACTOR','TISSUE SPECIFICITY','SUBCELLULAR LOCATION'}


def record(identity,dataset,kind,title,fields,filename,pointer,aliases=()):
    return {'id':identity,'dataset':dataset,'kind':kind,'title':title,'aliases':sorted(set(aliases)),
            'fields':fields,'source_locator':{'raw_filename':filename,'json_pointer':pointer}}


def protein_records(payload,filename):
    rows=[];edges=[]
    for i,p in enumerate(payload['results']):
        accession=p['primaryAccession'];taxon=p['organism']['taxonId']
        if taxon not in (9606,10090) or p['entryType']!='UniProtKB reviewed (Swiss-Prot)':
            raise ValueError('Protein outside reviewed human/mouse selection')
        gene=p['genes'][0]['geneName']['value'];rid='uniprot:'+accession
        comments=[c for c in p.get('comments',[]) if c['commentType'] in ANNOTATIONS]
        name=p['proteinDescription']['recommendedName']['fullName']['value']
        rows.append(record(rid,'uniprot_adjuvant','protein_annotation',gene+' | '+p['organism']['scientificName'],
            {'accession':accession,'gene':gene,'taxon_id':taxon,'organism':p['organism']['scientificName'],
             'protein_name':name,'entry_audit':p['entryAudit'],'protein_existence':p.get('proteinExistence'),
             'genes':p['genes'],'annotations':comments,
             'scope':'Selected database annotations with original ECO evidence; not protein or enzyme measurements in the user sample.'},
            filename,f'/results/{i}',[accession,gene,name]))
        for c in comments:
            if c['commentType']!='CATALYTIC ACTIVITY':continue
            for x in c['reaction'].get('reactionCrossReferences',[]):
                if x['database']=='Rhea':
                    edges.append({'subject':rid,'predicate':'annotated_catalytic_reaction','object':'rhea:'+x['id'].split(':')[-1],
                        'source_record':rid,'qualifier':'uniprot_annotation_not_context_specific_activity'})
    return rows,edges


def trial_records(pages):
    def page_number(item):
        m=re.fullmatch(r'clinicaltrials-adjuvant(?:-([0-9]+))?\.json',item[0])
        if not m:raise ValueError('Unexpected trial page name')
        return int(m[1] or 1)
    pages=sorted(pages,key=page_number)
    if not pages or [page_number(p) for p in pages]!=list(range(1,len(pages)+1)):
        raise ValueError('Duplicate or incomplete pagination')
    total=pages[0][1]['totalCount'];seen=set();rows=[]
    for page_index,(name,payload) in enumerate(pages):
        if payload.get('totalCount',total)!=total or bool(payload.get('nextPageToken'))!=(page_index<len(pages)-1):
            raise ValueError('incomplete pagination or changed result total')
        for i,s in enumerate(payload['studies']):
            p=s['protocolSection'];identity=p['identificationModule'];nct=identity['nctId']
            if not re.fullmatch('NCT[0-9]{8}',nct) or nct in seen:raise ValueError('Duplicate or invalid trial ID')
            seen.add(nct);status=p.get('statusModule',{});arms=p.get('armsInterventionsModule',{})
            fields={'nct_id':nct,'status':status.get('overallStatus'),
                'has_results':s.get('hasResults'), 'brief_title':identity['briefTitle'],
                'official_title':identity.get('officialTitle'),
                'conditions':p.get('conditionsModule',{}),'design':p.get('designModule',{}),
                'interventions':arms.get('interventions',[]),'arms':arms.get('armGroups',[]),
                'outcomes':p.get('outcomesModule',{}),'eligibility':p.get('eligibilityModule',{}),
                'summary':p.get('descriptionModule',{}),'status_dates':status,
                'references':p.get('referencesModule',{}),
                'selection_scope':'Frozen vaccine AND adjuvant-related intervention-name query hit; not manually adjudicated adjuvant use, efficacy or safety.'}
            rows.append(record('ctgov:'+nct,'ctgov_adjuvant','clinical_trial_registry',identity['briefTitle'],
                               fields,name,f'/studies/{i}',[nct]))
    if len(rows)!=total:raise ValueError('incomplete pagination count')
    return rows


def reaction_records(payload,filename):
    rows=[]
    for i,b in enumerate(payload['results']['bindings']):
        identity=b['accession']['value']
        if not re.fullmatch(r'RHEA:[0-9]+',identity) or b['rhea']['value']!='http://rdf.rhea-db.org/'+identity.split(':')[1]:
            raise ValueError('Reaction identity mismatch')
        rows.append(record('rhea:'+identity.split(':')[1],'rhea_adjuvant','curated_reaction',identity+' '+b['equation']['value'],
            {'accession':identity,'equation':b['equation']['value'],'bindings':b},filename,f'/results/bindings/{i}',[identity]))
    return rows


def verified(raw_dir,name):
    path=raw_dir/name
    if path.is_symlink() or path.stat().st_size>20_000_000:raise ValueError('Unsupported source path/size')
    raw=path.read_bytes();meta=json.loads((raw_dir/(name+'.provenance.json')).read_text())
    if hashlib.sha256(raw).hexdigest()!=meta['sha256'] or len(raw)!=meta['bytes']:
        raise ValueError('Raw source integrity mismatch: '+name)
    return json.loads(raw),{'source_url':meta['url'],'retrieved_at':meta['retrieved_at'],
        'raw_sha256':meta['sha256'],'raw_bytes':len(raw),'raw_filename':name}


def build(base,raw,output,*,immune=None):
    base=ResourcePack(base);raw=Path(raw);output=Path(output)
    if output.exists():raise ValueError('Use a fresh output path')
    data={};sources={}
    names=['uniprot-panel.json','rhea-panel.json']+sorted(p.name for p in raw.glob('clinicaltrials-adjuvant*.json') if not p.name.endswith('.provenance.json'))
    for name in names:data[name],sources[name]=verified(raw,name)
    pages=[(n,data[n]) for n in names if n.startswith('clinicaltrials-')]
    trials=trial_records(pages)
    ordered=sorted(pages,key=lambda x:1 if x[0]=='clinicaltrials-adjuvant.json' else int(x[0].split('-')[-1].split('.')[0]))
    first_query=parse_qs(urlsplit(sources[ordered[0][0]]['source_url']).query)
    for index,(n,payload) in enumerate(ordered):
        query=parse_qs(urlsplit(sources[n]['source_url']).query);token=query.pop('pageToken',None)
        if query!=first_query or token!=([ordered[index-1][1]['nextPageToken']] if index else None):
            raise ValueError('Trial pagination provenance mismatch')
    proteins,edges=protein_records(data['uniprot-panel.json'],'uniprot-panel.json')
    reactions=reaction_records(data['rhea-panel.json'],'rhea-panel.json')
    rows=list(base.records.values())+proteins+reactions+trials;edges=base.edges+edges
    added=[
      ('uniprot_adjuvant',['uniprot-panel.json'],'UniProt CC-BY-4.0; preserve UniProt and original evidence attribution.',
       'Selected reviewed human/mouse annotations for BHMT, TLR4/7/9, MYD88, STING1 and NLRP3; not current sample measurements.'),
      ('rhea_adjuvant',['rhea-panel.json'],'Rhea CC-BY-4.0; retain Rhea/SIB attribution.',
       'Explicit reaction equations referenced by selected UniProt entries; no flux or contextual activity inference.'),
      ('ctgov_adjuvant',[n for n,_ in ordered],'ClinicalTrials.gov public registry; retain attribution and sponsor/submitter records. Reuse subject to ClinicalTrials.gov terms.',
       'All pages of the frozen intervention-name query. Registry protocol metadata and result-availability flag only, not clinical efficacy conclusions.')]
    datasets=deepcopy(base.manifest['datasets'])
    for identity,files,license,scope in added:
        datasets.append({'id':identity,**sources[files[0]],'source_files':{n:sources[n] for n in files},
                         'license':license,'evidence_scope':scope,
                         'limitations':['Coverage is restricted to the frozen query/panel; absence is not negative biological evidence.',
                           'Keep organism, experimental context and original evidence type; database annotations are not independent replication.']})
    if immune is not None:
        extra,relations,provenance=support_records(immune)
        rows+=extra;edges+=relations;datasets+=provenance
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate record ID')
    if any(len(canonical_json(r).encode())>100000 for r in rows):raise ValueError('Record exceeds pack bound')
    manifest={'schema_version':1,'builder':'adj_partner.resources.expansion:v4','created_at':datetime.now(timezone.utc).isoformat(),
              'base_pack_sha256':base.manifest_hash,'datasets':datasets,'files':{},'counts':dict(Counter(r['kind'] for r in rows)),
              'relation_count':len(edges)}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.adjuvant-v4-',dir=output.parent) as temporary:
        stage=Path(temporary)/'pack';stage.mkdir()
        for name,values in [('records.jsonl',rows),('relations.jsonl',edges)]:
            content=('\n'.join(canonical_json(v) for v in values)+'\n').encode();(stage/name).write_bytes(content)
            manifest['files'][name]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
        (stage/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        ResourcePack(stage.resolve());stage.rename(output)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','raw','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--immune',type=Path)
    a=p.parse_args();m=build(a.base,a.raw,a.output,immune=a.immune)
    print(json.dumps({'counts':m['counts'],'relation_count':m['relation_count']}))
