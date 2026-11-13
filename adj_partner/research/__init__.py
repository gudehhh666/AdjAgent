"""Shared, session-scoped scientific records; no coordinator or hidden model."""

from .ledger import Ledger
from .design_records import DesignRecords
from .experiments import ExperimentRecords
from .records import Records
from .assessments import Assessments, HostAssessor
from .reports import Reports
from .acceptance import ReportNotAccepted
from .tools import register_research_tools
from .views import StateViews


class ResearchStore(ExperimentRecords, DesignRecords, Records, Assessments, Reports, StateViews, Ledger):
    """Host service over one Store's versioned research tables."""


__all__ = ["ResearchStore", "HostAssessor", "ReportNotAccepted", "register_research_tools"]
