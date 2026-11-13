"""Audited frozen-matrix adapter; all treatments are covered before selection."""
import csv
import hashlib
import json
from pathlib import Path

from .molecule_reports import validate_dose, VERSION
from .validation import identifier

PANEL = ['Cdkn1a','Phlda3','Ifng','Il2','Il1b','Nlrp3','Gzmb','Bhmt','Mmp9',
         'Mpo','Elane','H2-Ab1','Cth','Slc1a4','Hmox1','Slc7a11','Trp53','Cdkn2a']
MATRIX_FILES = ['expression_fpkm.tsv.gz','vendor_log2fc.tsv.gz','vendor_qval.tsv.gz','vendor_flags.tsv.gz','treatment_identity.csv']
CONTEXT = ('有疫苗背景的老年鼠脾细胞与抗原共同体外处理，小分子相对匹配DMSO。'
           '原始剂量/时间、抗原身份、DMSO浓度、动物/混样和生物重复结构未知。'
           '现有输入为供应商汇总差异表，不能估计独立重复方差；共同髓系/炎症程序可能受组成或共享对照影响。')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_profiles(discovery, doses):
    import numpy as np
    import pandas as pd
    root=Path(discovery).resolve(); inp=root/'input'
    manifest=json.loads((inp/'manifest.json').read_text())
    names=MATRIX_FILES
    for name in names:
        if sha(inp/name) != manifest['files'][name]: raise ValueError('Frozen input hash mismatch: '+name)
    identities=list(csv.DictReader((inp/'treatment_identity.csv').open()))
    aliases=[i['alias'] for i in identities]; compounds=[i['compound'] for i in identities]
    if len(aliases) != manifest['n_treatments'] or len(set(aliases)) != len(aliases) or len(set(compounds)) != len(compounds):
        raise ValueError('Treatment identity coverage mismatch')
    if set(doses) != set(compounds): raise ValueError('Dose strategy coverage mismatch')
    for alias in aliases: identifier(alias)
    expr=pd.read_csv(inp/names[0],sep='\t').set_index('gene_id')
    fc=pd.read_csv(inp/names[1],sep='\t').set_index('gene_id')
    q=pd.read_csv(inp/names[2],sep='\t').set_index('gene_id')
    flags=pd.read_csv(inp/names[3],sep='\t').set_index('gene_id')
    for table in (expr,fc,q,flags):
        if len(table) != manifest['n_genes'] or not table.index.is_unique or not table.index.equals(expr.index):
            raise ValueError('Matrix gene IDs do not align')
        if not set(aliases) <= set(table.columns): raise ValueError('Missing treatment matrix column')
    if not np.isfinite(expr[['Vehicle',*aliases]].to_numpy()).all(): raise ValueError('Invalid FPKM')
    if (expr[['Vehicle',*aliases]] < 0).any().any(): raise ValueError('Negative FPKM')
    relative=fc[aliases].sub(fc[aliases].median(axis=1),axis=0)
    eligible=expr[aliases].ge(1).apply(lambda column: column | expr['Vehicle'].ge(1))
    common=(flags[aliases].astype(bool) & eligible).sum(axis=1).ge(min(20,len(aliases)))
    sources={s['source_name']:s for s in manifest['source_records']}
    profiles=[]
    for identity in identities:
        alias=identity['alias']; compound=identity['compound']
        source=sources[identity['source_file']]
        path=Path(manifest['input_root'])/source['source_name']
        if not path.is_file() or sha(path)!=source['source_sha256']:
            raise ValueError('Original XLSX missing or changed: '+compound)
        mask=eligible[alias]
        # Representative fixed markers plus large treatment-specific residuals.
        top=relative.loc[mask & ~common,alias].abs().nlargest(5).index.tolist()
        large=fc.loc[mask,alias].abs().nlargest(3).index.tolist()
        markers=expr.index[expr['gene_name'].isin(PANEL)].tolist()
        selected=list(dict.fromkeys([*markers,*top,*large]))
        genes=[]
        for gid in selected:
            rawq=q.at[gid,alias];rawfc=fc.at[gid,alias]
            genes.append({'gene_id':gid,'gene_name':str(expr.at[gid,'gene_name']),
                          'vehicle_fpkm':float(expr.at[gid,'Vehicle']),'treatment_fpkm':float(expr.at[gid,alias]),
                          'vendor_log2fc':float(rawfc) if np.isfinite(rawfc) else None,
                          'vendor_qval':float(rawq) if np.isfinite(rawq) else None,
                          'vendor_flag':bool(flags.at[gid,alias]),
                          'panel_relative_log2fc':float(relative.at[gid,alias]) if np.isfinite(relative.at[gid,alias]) else None})
        profile={'alias':alias,'compound':compound,'source':source,'context':CONTEXT,
                 'counts':{'genes':len(expr),'vendor_flagged':int(flags[alias].sum()),
                           'eligible_flagged':int((flags[alias]&mask).sum()),'common_core':int(common.sum())},
                 'genes':genes,'dose_strategy':doses[compound]}
        validate_dose(profile);profiles.append(profile)
    return profiles, {'schema_version':VERSION,'input_manifest_sha256':sha(inp/'manifest.json'),
                      'input_files':{n:sha(inp/n) for n in names},'source_records':manifest['source_records'],
                      'method':'FPKM>=1 in either member; vendor fields unchanged; panel-relative=vendor log2FC minus all-treatment median. Fixed marker panel plus 5 largest non-common residuals and 3 largest absolute effects.',
                      'full_gene_coverage':len(expr),'treatment_coverage':len(profiles),'wet_experiment_status':'not_performed'}


def prepare_bundle(discovery, dose_file, output):
    doses=json.loads(Path(dose_file).read_text())
    profiles, provenance=load_frozen_profiles(discovery,doses)
    bundle={'schema_version':VERSION,'provenance':provenance,'profiles':profiles}
    path=Path(output).resolve()
    if path.exists(): raise ValueError('Refusing to overwrite frozen input bundle')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n')
    return {'bundle':str(path),'sha256':sha(path),'molecules':len(profiles)}
