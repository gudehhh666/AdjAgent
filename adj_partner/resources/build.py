"""Deterministic import of public ontology/assay snapshots, without execution.

Run with --raw workspace/adjuvant-infrastructure/raw --output data/adjuvant_resources.
The output is local research data; third-party code is not copied or executed.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from ..tools.catalog import canonical_json
from .pack import ResourcePack

NS={'owl':'http://www.w3.org/2002/07/owl#','rdfs':'http://www.w3.org/2000/01/rdf-schema#',
    'rdf':'http://www.w3.org/1999/02/22-rdf-syntax-ns#','obo':'http://purl.obolibrary.org/obo/',
    'oboInOwl':'http://www.geneontology.org/formats/oboInOwl#'}
ABOUT='{'+NS['rdf']+'}about'; RESOURCE='{'+NS['rdf']+'}resource'


def term_id(iri):
    return iri[len(NS['obo']):] if iri.startswith(NS['obo']) else iri


def ontology_records(path):
    # This parser reads only a host-downloaded bounded XML snapshot. It does
    # not resolve ontology imports or perform OWL entailment.
    raw=Path(path).read_bytes()
    if len(raw)>50_000_000 or b'<!ENTITY' in raw or b'<!DOCTYPE' in raw:
        raise ValueError('Unsupported ontology size or XML declarations')
    root=ET.fromstring(raw)
    classes={c.get(ABOUT):c for c in root.findall('owl:Class',NS) if c.get(ABOUT)}
    seed=NS['obo']+'VO_0000580'
    if seed not in classes:
        raise ValueError('Missing vaccine adjuvant root')
    selected={seed}
    while True:
        expanded=selected|{iri for iri,c in classes.items()
            if any(p.get(RESOURCE) in selected for p in c.findall('rdfs:subClassOf',NS))}
        if selected==expanded: break
        selected=expanded
    adjuvants=set(selected)
    edges=[]
    for iri in sorted(adjuvants):
        c=classes[iri]; rid=term_id(iri)
        for parent in c.findall('rdfs:subClassOf',NS):
            if parent.get(RESOURCE):
                target=parent.get(RESOURCE)
                edges.append({'subject':rid,'predicate':'subClassOf','object':term_id(target),
                              'source_record':rid,'qualifier':'asserted_named_parent'})
                if target in classes: selected.add(target)
            for restriction in parent.findall('owl:Restriction',NS):
                prop=restriction.find('owl:onProperty',NS)
                if prop is None or not prop.get(RESOURCE): continue
                for qualifier in ['someValuesFrom','allValuesFrom','hasValue']:
                    obj=restriction.find('owl:'+qualifier,NS)
                    if obj is not None and obj.get(RESOURCE):
                        target=obj.get(RESOURCE)
                        edges.append({'subject':rid,'predicate':term_id(prop.get(RESOURCE)),
                            'object':term_id(target),'source_record':rid,'qualifier':qualifier})
                        if target in classes: selected.add(target)
    records=[]
    for iri in sorted(selected):
        c=classes[iri]; annotations={}
        for child in c:
            if len(child)==0 and child.text and child.text.strip():
                key=child.tag[1:].replace('}', '') if child.tag.startswith('{') else child.tag
                annotations.setdefault(key,[]).append(child.text.strip())
        aliases=[]
        for key,values in annotations.items():
            if any(key.endswith(x) for x in ['IAO_0000118','hasExactSynonym','hasRelatedSynonym']):
                aliases.extend(values)
        records.append({'id':term_id(iri),'dataset':'vo','kind':'ontology',
            'title':c.findtext('rdfs:label',term_id(iri),NS),'aliases':sorted(set(aliases)),
            'fields':{'iri':iri,'in_adjuvant_subgraph':iri in adjuvants,'annotations':annotations,
                'scope':'Explicit named subclass closure plus directly referenced context terms; complex OWL expressions are not interpreted.'},
            'source_locator':{'iri':iri}})
    return records,edges


def screening_records(path):
    records=[]
    with Path(path).open(newline='',encoding='utf-8-sig') as handle:
        reader=csv.DictReader(handle)
        if not {'Catalog','PME Code','SMILES'} <= set(reader.fieldnames or []):
            raise ValueError('Missing screening identity columns')
        for row_number,row in enumerate(reader,2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed screening CSV row')
            records.append({'id':f'tang2023:row:{row_number}','dataset':'tang2023','kind':'assay',
                'title':row['Catalog']+' '+row['PME Code'],'aliases':[row['Catalog'],row['PME Code']],
                'fields':row,'source_locator':{'row':row_number,
                    'row_convention':'One-based parsed CSV records including header; not independent subjects or replicates.'}})
    return records


def source(raw_dir,name,identity,license,scope,limitations):
    path=raw_dir/name
    meta=json.loads((raw_dir/(name+'.provenance.json')).read_text())
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta['sha256'] or len(raw)!=meta['bytes']:
        raise ValueError('Raw source integrity mismatch: '+name)
    return {'id':identity,'source_url':meta['url'],'retrieved_at':meta['retrieved_at'],
        'raw_sha256':meta['sha256'],'raw_bytes':meta['bytes'],'raw_filename':name,
        **{k:meta[k] for k in ('archive_sha256','archive_url','member') if k in meta},
        'license':license,'evidence_scope':scope,'limitations':limitations}


def build(raw_dir,output,*,extended=False,vaxjo=False):
    raw_dir=Path(raw_dir); output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory; existing pack is not overwritten')
    if vaxjo and not extended:raise ValueError('Vaxjo pack requires extended=True')
    datasets=[
        source(raw_dir,'vo.owl','vo','CC-BY-4.0; credit Vaccine Ontology authors; version pinned in source URL',
            'Curated ontology annotation, not experimental validation in the user setting.',
            ['An asserted target or immune-profile annotation does not prove activity in a particular species, cell, dose or formulation.',
             'Named subclass closure is incomplete with respect to logical OWL inference.']),
        source(raw_dir,'tang-screening-results.csv','tang2023',
            'Author-published research data; repository redistribution license not identified. Local research use; do not assume code is permissively licensed.',
            'Author-reported in-vitro immunomodulator screening measurements, not independently reproduced vaccine outcomes.',
            ['Raw column names and string values retained, including blanks; viability masks and Abs/Lum require the original method before interpreting thresholds.',
             'Different agonists and screening stages are distinct conditions; duplicate catalog IDs are not independent replication.',
             'No dose, species or assay mapping is inferred from a compound name.'])]
    ontology,edges=ontology_records(raw_dir/'vo.owl')
    assays=screening_records(raw_dir/'tang-screening-results.csv')
    records=ontology+assays
    if extended:
        from .tabular import viget_records, workbook_records
        datasets.append(source(raw_dir,'viget-metadata.csv','viget',
            'CC0 per ImmPort VIGET July 2023 data documentation; retain VIGET/ImmPort attribution.',
            'Public vaccine study and biosample metadata only; expression matrix not included.',
            ['Repeated biosample/subject identifiers and longitudinal rows are preserved; row counts are not independent subjects.',
             'A study match does not establish treatment efficacy or transferability to the user experiment.']))
        extra,relations=viget_records(raw_dir/'viget-metadata.csv');records+=extra;edges+=relations
        for name,identity in [
            ('21726997-20260731_optimization.xlsx','agent_optimization'),
            ('21726997-20250514_NHP_ELISA.xlsx','agent_nhp'),
            ('21726997-20260320_storage_ELISA.xlsx','agent_storage')]:
            datasets.append(source(raw_dir,name,identity,'CC-BY-4.0 per Zenodo record 21726997; credit data authors.',
                'Author-provided vaccine formulation/ELISA worksheet cells, not interpreted assay outcomes.',
                ['Literal cell values, formula strings and available cached formula values retained separately; no recalculation.',
                 'Blank, zero, error and OVER values remain distinct. No plate-to-sample mapping or units inferred.',
                 'Full methods and sample/control mapping required before cross-condition comparison or efficacy conclusions.']))
            extra,relations=workbook_records(raw_dir/name,identity);records+=extra;edges+=relations
    if vaxjo:
        from .vaxjo import vaxjo_records
        for number,identity,scope in [
            (1,'vaxjo_master','Author-curated adjuvant catalog assembled from prior databases and LLM candidates; preserve the per-entry Source field.'),
            (5,'vaxjo_audit','Author evaluation of 50 selected mechanism extractions plus aggregate worksheet rows; not our independent validation.'),
            (6,'vaxjo_matrix','Author literature-extraction mechanism-family matrix; flags are annotations, not measured effects or causal evidence.')]:
            name=f'vaxjo-supplementary-file-{number}.xlsx'
            datasets.append(source(raw_dir,name,identity,
                'Author-public GitHub supplement, local research copy. README limits code to academic/research use; separate supplement redistribution grant not confirmed. Do not inherit Zenodo license across snapshots.',
                scope,['Retain source spelling, missing cells, reference strings and formula text; formulas are not executed.',
                       'Catalog, ontology and literature extraction share sources; not independent experimental replications.',
                       'Targets and immune profiles do not establish efficacy in a particular species, cell, antigen, dose or formulation.']))
            extra,relations=vaxjo_records(raw_dir/name,identity);records+=extra;edges+=relations
    output.mkdir(parents=True,exist_ok=True)
    manifest={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),
        'datasets':datasets,'files':{},'counts':dict(Counter(r['kind'] for r in records)),
        'relation_count':len(edges),'builder':'adj_partner.resources.build:v3' if vaxjo else 'adj_partner.resources.build:v2' if extended else 'adj_partner.resources.build:v1'}
    for name,rows in [('records.jsonl',records),('relations.jsonl',edges)]:
        raw=('\n'.join(canonical_json(r) for r in rows)+'\n').encode()
        (output/name).write_bytes(raw)
        manifest['files'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    ResourcePack(output)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extended',action='store_true',help='Include verified VIGET metadata and all three AGENT workbooks')
    parser.add_argument('--vaxjo',action='store_true',help='Also include author Vaxjo supplements; requires --extended')
    args=parser.parse_args()
    print(json.dumps(build(args.raw,args.output,extended=args.extended,vaxjo=args.vaxjo),ensure_ascii=False,indent=2))
