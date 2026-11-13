"""Immutable public task identity and explicit delivery semantics."""
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
import hashlib
import json
from ..contracts import Budget
from ..tools.catalog import canonical_json

MODES = ('qa', 'design', 'discovery_design')
DEFAULT_BUDGET = dict(max_steps=96, max_input_tokens=1200000, max_output_tokens=100000, wall_time_seconds=3600.)


def implementation_hash():
    root=Path(__file__).resolve().parents[1]
    h=hashlib.sha256()
    # Persist the implementation identity; loading an old TaskSpec preserves its hash.
    for folder in ('orchestration','task_modes','runtime','research','native'):
        for path in sorted((root/folder).glob('*.py')):
            h.update(str(path.relative_to(root)).encode());h.update(path.read_bytes())
    return h.hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    task_type: str
    question: str
    dataset_sha256: str = ''
    row_id: str = ''
    model: str = 'gpt-5.6-luna'
    critic_model: str = ''
    provider_sha256: str = ''
    workflow_version: str = 'unified-experiment-v1'
    implementation_sha256: str = field(default_factory=implementation_hash)
    parameters: dict = field(default_factory=dict)
    budget: dict = field(default_factory=lambda: dict(DEFAULT_BUDGET))
    allowed_roots: tuple = ()
    allow_network: bool = True
    resource_manifest_sha256: str = ''

    def __post_init__(self):
        if self.task_type not in MODES: raise ValueError('Unknown task mode')
        if not isinstance(self.question, str) or not self.question.strip() or len(self.question)>20000:
            raise ValueError('Public question must contain 1..20000 characters')
        if not all(isinstance(x,str) for x in (self.model,self.critic_model,self.workflow_version,self.dataset_sha256,self.row_id)):
            raise ValueError('Task identity fields must be strings')
        if not self.model or not self.workflow_version: raise ValueError('Model and workflow version required')
        if self.provider_sha256 and (not isinstance(self.provider_sha256,str) or len(self.provider_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.provider_sha256)):raise ValueError('Invalid provider identity')
        if self.dataset_sha256 and (len(self.dataset_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.dataset_sha256)):
            raise ValueError('Invalid dataset hash')
        if type(self.allow_network) is not bool: raise ValueError('Invalid network policy')
        if self.resource_manifest_sha256 and (not isinstance(self.resource_manifest_sha256,str) or len(self.resource_manifest_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.resource_manifest_sha256)):
            raise ValueError('Invalid resource manifest identity')
        if any(not isinstance(p,str) for p in self.allowed_roots): raise ValueError('Invalid roots')
        local_design=self.task_type=='design' and self.workflow_version=='evidence-grounded-design-v2'
        if self.task_type!='discovery_design' and not local_design and self.allowed_roots: raise ValueError('Benchmark modes have no local data roots')
        if set(self.parameters)-{'top_n','n_permutations','max_candidates','max_revision_rounds','retrieval_steps','no_progress_limit'}:
            raise ValueError('Unknown workflow parameter; hidden benchmark fields are forbidden')
        for key,value in self.parameters.items():
            if type(value) is not int or value <= 0: raise ValueError('Parameters must be positive integers')
        if not isinstance(self.parameters,dict) or not isinstance(self.budget,dict):raise ValueError('Parameters and budget must be objects')
        Budget(**self.budget)
        if len(self.implementation_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.implementation_sha256):raise ValueError('Invalid implementation hash')
        object.__setattr__(self,'parameters',MappingProxyType(dict(self.parameters)))
        object.__setattr__(self,'budget',MappingProxyType(dict(self.budget)))
        object.__setattr__(self,'allowed_roots',tuple(self.allowed_roots))
        canonical_json(self.to_dict())

    def to_dict(self):
        value={**{k:getattr(self,k) for k in self.__dataclass_fields__},'parameters':dict(self.parameters),'budget':dict(self.budget),'allowed_roots':list(self.allowed_roots)}
        if not self.resource_manifest_sha256:value.pop('resource_manifest_sha256')
        return value

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value,dict) or set(value)-set(cls.__dataclass_fields__):
            raise ValueError('Unknown or hidden TaskSpec fields')
        return cls(**value)

    @property
    def fingerprint(self):
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()


def result_record(spec, execution_status, delivery=None, *, review_status='not_run', **extra):
    delivery=delivery or {}
    report=delivery.get('kind') in {'accepted_report','incomplete_report'}
    return {'schema_version':1,'task_fingerprint':spec.fingerprint,'task_type':spec.task_type,
            'question':spec.question,'execution_status':execution_status,
            'delivery_status':'report' if report else 'answer' if delivery.get('content') else 'none',
            'content_status':('incomplete' if delivery.get('research_design_status')=='incomplete_design' else
                              'complete' if delivery.get('content') else 'not_assessed'),
            'review_status':review_status,'empirical_validation_status':'not_performed',
            'delivery':delivery,**extra}
