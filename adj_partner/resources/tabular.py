"""Literal study metadata and worksheet coordinates, with no assay inference."""
from collections import defaultdict
import csv
from datetime import date, datetime, time
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile


def viget_records(path):
    records=[]; edges=[]; studies=defaultdict(list)
    with Path(path).open(newline='',encoding='utf-8-sig') as handle:
        reader=csv.DictReader(handle)
        required={'immport_study_accession','study_title','biosample_id','vaccine','gse','gsm'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Missing VIGET identity columns')
        if len(set(reader.fieldnames))!=len(reader.fieldnames):
            raise ValueError('Duplicate VIGET columns')
        for number,row in enumerate(reader,2):
            if None in row or any(v is None for v in row.values()) or not row['immport_study_accession']:
                raise ValueError('Malformed VIGET row or missing study accession')
            rid=f'viget:row:{number}';study=row['immport_study_accession']
            records.append({'id':rid,'dataset':'viget','kind':'sample_metadata',
                'title':' | '.join(row[k] for k in ('immport_study_accession','biosample_id','vaccine','gsm')),
                'aliases':[row[k] for k in ('biosample_id','gse','gsm') if row[k]],
                'fields':row,'source_locator':{'row':number,'row_convention':'One-based parsed CSV record including header; not an independent subject.'}})
            studies[study].append((number,row))
            edges.append({'subject':rid,'predicate':'in_study','object':'viget:study:'+study,
                          'source_record':rid,'qualifier':'reported_study_membership'})
    for study,rows in sorted(studies.items()):
        # Preserve all categorical values, including blanks; do not infer age
        # ranges, resolve inconsistent titles, or merge repeated subjects.
        summary={k:sorted({row[k] for _,row in rows}) for k in rows[0][1]
                 if k not in {'biosample_id','immport_biosample_accession','immport_subject_accession','gsm'}}
        summary.update(row_count=len(rows),unique_biosample_count=len({r['biosample_id'] for _,r in rows if r['biosample_id']}),
                       scope='Metadata only; row counts are not independent subjects, and expression measurements are not included.')
        records.append({'id':'viget:study:'+study,'dataset':'viget','kind':'study_metadata',
            'title':study+' | '+' / '.join(summary['study_title']),
            'aliases':[study,*[v for v in summary['gse'] if v]],'fields':summary,
            'source_locator':{'study_accession':study,'csv_rows':[n for n,_ in rows]}})
    return records,edges


def _cell_value(value):
    if isinstance(value,(datetime,date,time)):
        return value.isoformat()
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    raise ValueError('Unsupported worksheet cell type')


def workbook_records(path,dataset):
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter
    path=Path(path)
    if path.stat().st_size>10_000_000:
        raise ValueError('Workbook exceeds import size limit')
    with ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist())>100_000_000:
            raise ValueError('Workbook expanded size exceeds import limit')
    workbook=load_workbook(path,read_only=True,data_only=False,keep_links=False)
    cached=None
    try:
        cached=load_workbook(path,read_only=True,data_only=True,keep_links=False)
        records=[];edges=[];sheets=[];dataset_id=dataset+':workbook'
        for sheet in workbook:
            if not sheet.max_row or not sheet.max_column or sheet.max_row>10000 or sheet.max_column>200:
                raise ValueError('Unsupported worksheet dimensions')
            sheets.append({'name':sheet.title,'max_row':sheet.max_row,'max_column':sheet.max_column})
            columns=[get_column_letter(i) for i in range(1,sheet.max_column+1)]
            values=cached[sheet.title].iter_rows(min_row=1,max_row=sheet.max_row,max_col=sheet.max_column)
            cells=sheet.iter_rows(min_row=1,max_row=sheet.max_row,max_col=sheet.max_column)
            for number,(row,cache) in enumerate(zip(cells,values,strict=True),1):
                fields={col:_cell_value(cell.value) for col,cell in zip(columns,row,strict=True)}
                fields['_cell_types']={col:cell.data_type for col,cell in zip(columns,row,strict=True) if cell.value is not None}
                fields['_cached_formula_values']={col:_cell_value(c.value) for col,v,c in zip(columns,row,cache,strict=True) if v.data_type=='f'}
                fields['_scope']='Literal worksheet row. Formulas are not recalculated; cached values may be stale. Blank, zero, errors and OVER remain distinct. Sample, unit and assay mappings require source methods.'
                rid=f'{dataset}:{quote(sheet.title,safe="")}:row:{number}'
                records.append({'id':rid,'dataset':dataset,'kind':'experimental_row',
                    'title':f'{path.name} | {sheet.title} | row {number}', 'aliases':[],
                    'fields':fields,'source_locator':{'sheet':sheet.title,'row':number,
                        'range':f'A{number}:{columns[-1]}{number}'}})
                edges.append({'subject':rid,'predicate':'in_workbook','object':dataset_id,
                              'source_record':rid,'qualifier':'literal_worksheet_membership'})
        records.append({'id':dataset_id,'dataset':dataset,'kind':'experiment_dataset',
            'title':path.name,'aliases':[path.name], 'fields':{'sheets':sheets,
            'scope':'Author-provided workbook. Plate labels, doses, endpoints and efficacy are not inferred from filenames or cell positions.'},
            'source_locator':{'workbook':path.name}})
        return records,edges
    finally:
        workbook.close()
        if cached is not None:cached.close()
