"""Author Vaxjo supplements: separate curated entries, extraction and evaluation.

Read literal workbook values. Mechanism matrix flags are author annotations,
not causal findings. No embedded code, formula or reference URL is executed.
"""
import re
from urllib.parse import quote
from .tabular import workbook_records

HEADERS={
 'vaxjo_master':['Source','Vaxjo ID','Vaxjo Link','Label in Vaxjo','VO ID','Label in VO',
    'Vaccine adjuvant type','target receptor','induces immune profile','definition source'],
 'vaxjo_audit':['Adjuvant Name','Mechanism Subtypes','Correctly Identified Mechanism Subtypes','Total Mechanism Subtypes',None],
 'vaxjo_matrix':['Adjuvant','Adjuvant synergy / immune modulation','Antigen presentation / APCs',
    'Apoptosis / Cell Death Induction','B cell / antibody production','Complement / depot / formulation',
    'Cytokine signaling / production','Dendritic cell activation','Inflammatory response',
    'Macrophage / innate immune activation','Mucosal immunity','NLRP3 inflammasome activation',
    'Other / Unclassified','Pattern recognition / PRR sensing','STING / TRIF / MyD88 / RIG-I signaling',
    'T cell activation / polarization','TLR signaling']}


def vaxjo_records(path,dataset):
    if dataset not in HEADERS:raise ValueError('Unknown Vaxjo supplement')
    raw,_=workbook_records(path,dataset)
    sheets={r['source_locator']['sheet'] for r in raw if 'sheet' in r['source_locator']}
    if len(sheets)!=1:raise ValueError('Expected one Vaxjo worksheet')
    rows={r['source_locator']['row']:r for r in raw if 'row' in r['source_locator']}
    header=rows.get(3,{}).get('fields',{})
    columns=[chr(65+i) for i in range(len(HEADERS[dataset]))]
    if [header.get(c) for c in columns]!=HEADERS[dataset] or len([k for k in header if not k.startswith('_')])!=len(columns):
        raise ValueError('Unexpected Vaxjo supplement header')
    names={c:h if h is not None else '_unlabelled_'+c for c,h in zip(columns,HEADERS[dataset])}
    records=[];edges=[];seen_ids=set()
    for number,row in sorted(rows.items()):
        if number<=3:continue
        f=row['fields'];fields={name:f[col] for col,name in names.items()}
        fields['_cell_types']={names[c]:t for c,t in f['_cell_types'].items()}
        fields['_cached_formula_values']={names[c]:v for c,v in f['_cached_formula_values'].items()}
        rid=f'{dataset}:row:{number}';aliases=[]
        if dataset=='vaxjo_master':
            identity=fields['Vaxjo ID'];title=fields['Label in Vaxjo'];kind='curated_adjuvant'
            if type(identity) is not int or identity in seen_ids:
                raise ValueError('Missing/duplicate Vaxjo entry identity')
            if not isinstance(title,str) or not title.strip():
                label=fields['Label in VO']
                title=label if isinstance(label,str) and label.strip() else f'Vaxjo:{identity}'
                fields['_import_notes']=['Label in Vaxjo is not a nonempty string in the workbook; original value retained. Display title uses VO label or record identity.']
            seen_ids.add(identity)
            aliases=[v for v in [fields['Label in VO'],fields['VO ID'],f'Vaxjo:{identity}'] if isinstance(v,str) and v]
            vo=fields['VO ID']
            if isinstance(vo,str) and re.fullmatch(r'VO:\d{7}',vo):
                edges.append({'subject':rid,'predicate':'reported_vo_mapping','object':vo.replace(':','_'),
                    'source_record':rid,'qualifier':'author_reported_vo_mapping'})
        elif dataset=='vaxjo_matrix':
            title=fields['Adjuvant'];kind='extracted_mechanism'
            if not isinstance(title,str) or not title.strip():raise ValueError('Missing matrix adjuvant name')
            for family in HEADERS[dataset][1:]:
                value=fields[family]
                if type(value) is not int or value not in (0,1):raise ValueError('Unexpected mechanism matrix flag')
                if value==1:edges.append({'subject':rid,'predicate':'has_reported_mechanism_family',
                    'object':'vaxjo_matrix:family:'+quote(family,safe=''),'source_record':rid,
                    'qualifier':'author_literature_matrix_not_causal_validation'})
        else:
            title=fields['Adjuvant Name']
            kind='curation_audit' if isinstance(title,str) and title.strip() else 'curation_summary'
            title=title if kind=='curation_audit' else f'Author evaluation summary, row {number}'
        records.append({'id':rid,'dataset':dataset,'kind':kind,'title':title,'aliases':aliases,
            'fields':fields,'source_locator':{**row['source_locator'],'header_row':3}})
    if dataset=='vaxjo_matrix':
        for col,family in zip(columns[1:],HEADERS[dataset][1:]):
            records.append({'id':'vaxjo_matrix:family:'+quote(family,safe=''),'dataset':dataset,
                'kind':'mechanism_family','title':family,'aliases':[],
                'fields':{'scope':'Author grouping of literature-extracted mechanisms; not an experimental causal classification.'},
                'source_locator':{'sheet':next(iter(sheets)),'row':3,'column':col}})
    return records,edges
