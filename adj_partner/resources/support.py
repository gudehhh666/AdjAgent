"""Published assay methods and immune-study metadata, never experimental results."""
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader


def _check(path,sha,size):
    if path.is_symlink() or path.stat().st_size>5_000_000:raise ValueError('Unsupported support file')
    content=path.read_bytes()
    if len(content)!=size or hashlib.sha256(content).hexdigest()!=sha:raise ValueError('Support source integrity mismatch')
    return content


def support_records(directory):
    from .expansion import record
    directory=Path(directory);rows=[];edges=[];files={}
    acquisition={r['file']:r for r in json.loads((directory/'resolved-acquisition.json').read_text())}
    protocols=json.loads((directory/'ncl-protocol-metadata.json').read_text())
    for p in protocols:
        name=p['source_file'];a=acquisition[name]
        if a['sha256']!=p['source_sha256'] or a['url']!=p['source_url']:raise ValueError('Protocol provenance mismatch')
        content=_check(directory/name,a['sha256'],a['bytes'])
        if not content.startswith(b'%PDF-'):raise ValueError('Expected PDF, not redirect HTML')
        book=PdfReader(directory/name)
        if len(book.pages)!=p['pages'] or not 1<=len(book.pages)<=100:raise ValueError('Protocol page count mismatch')
        texts=[page.extract_text() for page in book.pages]
        if not all(isinstance(t,str) and t.strip() for t in texts):raise ValueError('Protocol text extraction incomplete')
        if p['doi'].lower() not in '\n'.join(texts[:2]).lower():raise ValueError('Protocol DOI not found in original pages')
        files[name]={'source_url':a['url'],'raw_filename':name,'raw_sha256':a['sha256'],'raw_bytes':a['bytes'],
            'retrieved_at':a['retrieved_utc'],'retrieval_time_basis':a.get('retrieval_time_basis','HTTP acquisition record')}
        identity='ncl:'+p['protocol_id']
        fields={k:p[k] for k in ['protocol_id','version','published_date','doi','pages','biological_context','experimental_gate','limitations']}
        r=record(identity,'ncl_protocols','assay_protocol',p['title'],fields,name,'',[p['protocol_id'],p['doi'],p['experimental_gate']])
        r['source_locator']={'raw_filename':name,'pages':[1,2],'extraction':'Metadata curated from the original PDF; page text separately available.'};rows.append(r)
        for i,text in enumerate(texts,1):
            rid=identity+':page:'+str(i)
            r=record(rid,'ncl_protocols','assay_protocol_page',p['protocol_id']+' page '+str(i),
                {'protocol_id':p['protocol_id'],'page':i,'text':text,'scope':'Published method, not an executed experiment or evidence of adjuvant efficacy.'},name,'',[p['protocol_id']])
            r['source_locator']={'raw_filename':name,'page':i,'extraction':'pypdf text; check original PDF for layout/tables'};rows.append(r)
            edges.append({'subject':rid,'predicate':'page_of_protocol','object':identity,'source_record':rid,'qualifier':'document_membership_not_biological_relation'})
    datasets=[{'id':'ncl_protocols',**next(iter(files.values())),'source_files':files,
        'license':'Official public NCI/Frederick National Laboratory protocols; preserve attribution and embedded third-party notices; local research copy.',
        'evidence_scope':'Published detection protocols with original PDF pages, not measurements or proof of vaccine-adjuvant efficacy.',
        'limitations':['Nanomaterial method applicability and assay interference require evaluation in the actual formulation/cell context.']}]
    a=json.loads((directory/'catalog-acquisition.json').read_text());name=a['file'];content=_check(directory/name,a['sha256'],a['bytes'])
    source={'source_url':a['url'],'raw_filename':name,'raw_sha256':a['sha256'],'raw_bytes':a['bytes'],'retrieved_at':a['retrieved_utc']}
    for i,value in enumerate(json.loads(content)['data']):
        # GeoAccession may describe a superseries or be repeated; keep catalog identity separate.
        r=record('immgen:catalog:'+str(i+1),'immgen_studies','immune_study_metadata',value['Name'],
            {**value,'expression_available_locally':False,'scope':'Study catalog only; no expression values imported.'},name,f'/data/{i}',[v for v in [value['GeoAccession']] if v])
        rows.append(r)
    datasets.append({'id':'immgen_studies',**source,'source_files':{name:source},
        'license':'Public ImmGen research catalog; attribute ImmGen and individual studies. Separate unrestricted redistribution grant not confirmed.',
        'evidence_scope':'Mouse immune study metadata only, not gene expression measurements or vaccine efficacy evidence.',
        'limitations':['Expression API returned server errors; incomplete matrix download excluded. No absent or zero expression inferred.']})
    return rows,edges,datasets
