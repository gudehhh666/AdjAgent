"""Fixed, project-owned evidence skills; no model-selected paths or permissions."""
import hashlib
from pathlib import Path
from ..profiles import load_trusted_guidance

SKILLS = {
    'paper': (), 'ontology': ('vo',), 'mechanism': ('uniprot_adjuvant','rhea_adjuvant'),
    'assay': ('ncl_protocols',), 'expression': ('immgen_studies','viget'),
    'clinical': ('ctgov_adjuvant',),
    'table': ('agent_optimization','agent_nhp','agent_storage','tang2023','vaxjo_master','vaxjo_audit','vaxjo_matrix'),
}
ROOT = Path(__file__).resolve().parents[2] / 'config' / 'skills'


def read_skill(name, datasets):
    if name not in SKILLS:raise ValueError('Unknown evidence skill')
    text=load_trusted_guidance(ROOT / ('adjuvant-'+name+'-evidence') / 'SKILL.md')
    body=text.split('---',2)[2].strip()
    if len(body.encode())>1500:raise ValueError('Evidence skill exceeds model-visible budget')
    return {'skill':name,'instructions':body,'sha256':hashlib.sha256(text.encode()).hexdigest(),
            'available_datasets':[d for d in SKILLS[name] if d in datasets],
            'missing_datasets':[d for d in SKILLS[name] if d not in datasets],
            'permission_scope':'Guidance only; existing tool, file and network permissions apply.'}
