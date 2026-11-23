"""Literal citation packets over a synthetic record; no model calls."""
from adj_partner.resources.citations import citation_packet


RECORD = {'id': 'ctgov:NCT0', 'fields': {'design': {'phases': ['PHASE2'], 'enrollmentInfo': {'count': 42}}}}


def test_explicit_pointer_wins_over_redundant_focus():
    both = citation_packet(RECORD, 'design', pointer='/record/fields/design', start=0, focus='design', length=600)
    plain = citation_packet(RECORD, 'design', pointer='/record/fields/design', start=0, length=600)
    assert both == plain
