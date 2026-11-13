"""Shared data structures. The EvidenceRecord is the universal unit that every
retrieval source emits and the fusion / reliability layers consume."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class EvidenceRecord:
    claim: str
    source_type: str                       # literature | biodb | web | local_corpus | local_data
    source_id: str                         # PMID:xxx | UniProt:Pxxxxx | URL | probe_id
    snippet: str = ""
    entities: list[str] = field(default_factory=list)
    date: str = ""
    retrieved_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    authority_weight: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReliabilityReport:
    hypothesis: str
    grounding: list[dict[str, Any]] = field(default_factory=list)
    consistency: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    reviewer_verdict: dict[str, Any] = field(default_factory=dict)
    confidence: dict[str, Any] = field(default_factory=dict)
    falsification: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
