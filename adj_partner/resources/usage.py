"""Resource-specific use contracts and bounded literal leaf selection.

These recipes describe available evidence, not biological conclusions. They never
modify source values or authorize network/filesystem access.
"""
import re
from collections import Counter
from ..tools.catalog import canonical_json

USAGE = {
 'vo': ('ontology', 'Resolve adjuvant names and explicit ontology relations.', 'Named ontology assertions are not experimental mechanisms; no OWL entailment.', ['annotations','iri','scope']),
 'tang2023': ('screen', 'Read original screening rows; retain catalog, stage and stimulant/readout.', 'String measurements and missingness are literal. Different screens/stages are not interchangeable or vaccine efficacy.', ['Catalog','PME Code','SMILES','Viability Mask 1','MPLA Lum','CpG Abs']),
 'viget': ('expression', 'Locate vaccine studies and sample metadata with accession, material and time.', 'No expression matrix installed. Sample count is not independent subject count; preserve study context.', ['gse','gsm','vaccine','biosample_type','immport_study_accession','immport_subject_accession','immport_vaccination_time']),
 'agent_optimization': ('table', 'Read workbook index, sheet/row coordinates and original formulation/assay cells.', 'Methods and plate/sample mapping are required before interpreting cells. Keep null, OVER, formula and cached values separate.', ['A','B','C','D','_cached_formula_values','_scope']),
 'agent_nhp': ('table', 'Read NHP ELISA workbook cells with exact sheet/row coordinates and original labels.', 'Do not infer animals, dilutions or titers from row position alone. Original formula text is not recomputation.', ['A','B','C','H','_cached_formula_values','_scope']),
 'agent_storage': ('table', 'Read storage-study workbook cells, including saturation and source formula caches.', 'OVER is saturation text, not zero. Storage labels alone do not establish formulation stability or immunity.', ['A','B','H','L','_cached_formula_values','_scope']),
 'vaxjo_master': ('catalog', 'Locate an external Vaxjo ID with a typed filter, then read original curated labels.', 'External Vaxjo ID is not worksheet row number. Preserve Source; curated/LLM-assisted entries are not independent experiments.', ['Vaxjo ID','Label in Vaxjo','Source','target receptor','induces immune profile','VO ID']),
 'vaxjo_audit': ('audit', 'Inspect author audit rows separately from aggregate worksheet rows.', 'Author label/extraction audit is not our independent mechanism validation or efficacy evidence.', ['Adjuvant Name','Mechanism Subtypes','Correctly Identified Mechanism Subtypes','Total Mechanism Subtypes']),
 'vaxjo_matrix': ('annotation', 'Inspect original mechanism-family flags for a named adjuvant.', 'Extracted flags are annotations, not measured effects, synergy or causal links; same source is not independent evidence.', ['Adjuvant','TLR signaling','NLRP3 inflammasome activation','T cell activation / polarization']),
 'uniprot_adjuvant': ('mechanism', 'Match target and organism, then read function/catalytic annotation leaves with evidence codes.', 'Curated and similarity-inferred annotations do not measure current-sample protein, enzyme activity or vaccine benefit.', ['gene','taxon_id','organism','annotations','protein_existence','entry_audit']),
 'rhea_adjuvant': ('mechanism', 'Read exact reaction accession/equation; follow explicit protein cross-reference and direction.', 'Reaction membership does not measure flux, thermodynamics in the current context or causal dependence.', ['accession','equation']),
 'ctgov_adjuvant': ('clinical', 'Read exact NCT record, registered objectives, population, planned outcomes and result-availability flag.', 'Registration/planned outcomes are not trial success. has_results=false means not present in this snapshot, not a negative trial.', ['nct_id','summary','outcomes','has_results','status','eligibility']),
 'ncl_protocols': ('assay', 'Find relevant protocol pages by readout and interference term; read focus-centred text before interpretation.', 'Published method is not candidate efficacy. Check protocol version, model and interference controls; document missing applicability.', ['text','protocol_id','experimental_gate','limitations','version','pages']),
 'immgen_studies': ('expression', 'Find immune study/cell context and public accession, then inspect local data availability.', 'Installed catalog has no expression values. Missing access or RNA is not zero expression, absent protein or absent enzyme activity.', ['Name','CellTypes','GeoAccession','expression_available_locally','scope','Description']),
}


def lookup_example(pack,dataset,kind,example,fields):
    """Executable source-specific example, not a task result or an access grant."""
    args={'dataset':dataset,'kind':kind,'query':example['id'],'fields':fields[:2]}
    if dataset=='uniprot_adjuvant':
        args={'dataset':dataset,'filters':{'gene':example['fields']['gene'],'taxon_id':example['fields']['taxon_id']},'fields':['annotations'],'focus':'function betaine homocysteine'}
    elif dataset=='ncl_protocols' and kind=='assay_protocol_page':
        args={'dataset':dataset,'kind':kind,'query':'dendritic','focus':'principles assessing','fields':['text']}
    return args


def _type(v):
    return 'null' if v is None else 'boolean' if type(v) is bool else 'integer' if type(v) is int else 'number' if type(v) is float else 'string' if isinstance(v,str) else 'array' if isinstance(v,list) else 'object'


def resource_guide(pack,dataset,*,kind=None,offset=0):
    if dataset not in pack.datasets or dataset not in USAGE:raise ValueError('Unknown or unconfigured resource')
    if type(offset) is not int or offset<0:raise ValueError('Invalid field offset')
    all_rows=[r for r in pack.records.values() if r['dataset']==dataset]
    counts=Counter(r['kind'] for r in all_rows)
    kind=kind or counts.most_common(1)[0][0]
    if kind not in counts:raise ValueError('Unknown resource kind')
    rows=[r for r in all_rows if r['kind']==kind];recipe,purpose,boundary,preferred=USAGE[dataset]
    fields=set().union(*(r['fields'] for r in rows))
    ordered=[f for f in preferred if f in fields]+sorted(fields-set(preferred))
    chosen=ordered[offset:offset+6]
    schema=[{'name':f,'types':sorted({_type(r['fields'][f]) for r in rows if f in r['fields']}),'present':sum(f in r['fields'] for r in rows)} for f in chosen]
    example=rows[0]
    return {'dataset':dataset,'recipe':recipe,'purpose':purpose,'boundary':boundary,
        'kinds':dict(counts),'selected_kind':kind,'records_of_kind':len(rows),'fields':schema,
        'next_offset':offset+6 if offset+6<len(ordered) else None,'example_record_id':example['id'],
        'example_read':{'record_id':example['id'],'fields':[f for f in chosen if f in example['fields']][:2]},
        'lookup_example':lookup_example(pack,dataset,kind,example,[f for f in chosen if f in example['fields']]),
        'lookup_tip':'Adapt example to task. Protein: filter gene AND taxon_id; free text can match another protein mentioning the gene. Protocol: focus on principles or interference, not just the title. Missing fields remain gaps.',
        'pack_sha256':pack.manifest_hash}


def _escape(s):return str(s).replace('~','~0').replace('/','~1')


def _leaves(value,pointer):
    if isinstance(value,dict):
        for k,v in value.items():yield from _leaves(v,pointer+'/'+_escape(k))
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from _leaves(v,pointer+'/'+str(i))
    else:yield pointer,value


def excerpt_fields(record,fields,focus,*,limit=2,width=240):
    if type(limit) is not int or not 1<=limit<=6 or type(width) is not int or not 80<=width<=400:raise ValueError('Invalid excerpt bounds')
    if not isinstance(focus,str) or len(focus)>200:raise ValueError('Invalid focus')
    if not fields or len(fields)>8 or any(f not in record['fields'] for f in fields):raise ValueError('Unknown or excessive fields')
    terms=set(re.findall(r'\w+',focus.casefold()))-{'the','and','of','in','for','to','a','is','what','how','does','with'}
    candidates=[]
    for field in fields:
        for pointer,value in _leaves(record['fields'][field],'/record/fields/'+_escape(field)):
            text=value if isinstance(value,str) else canonical_json(value)
            hits=[m for m in re.finditer(r'\w+',text) if m.group().casefold() in terms]
            matched={m.group().casefold() for m in hits}
            # Place the window around the densest occurrence cluster, not a page header.
            anchor=max(hits,key=lambda m:len({h.group().casefold() for h in hits if m.start()<=h.start()<m.start()+width-40})).start() if hits else 0
            start=max(0,anchor-30) if len(text)>width else 0;end=min(len(text),start+width)
            item={'field':field,'selector':{'pointer':pointer},'text':text[start:end],
                  'start':start,'end':end,'total_chars':len(text),'matched_terms':sorted(matched)}
            if not isinstance(value,str) or len(text)<=width:item['value']=value
            score=len(matched)*10+min(len(text),300)/1000 if terms else 0
            candidates.append((score,item))
    candidates.sort(key=lambda p:-p[0])
    return [item for _,item in candidates[:limit]]
