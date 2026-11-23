"""Legitimate external JATS declarations are inert; custom entities remain rejected."""
import httpx
import pytest
from lxml import etree

from adj_partner.native.documents import parse_jats
from adj_partner.native.fulltext import FullTextFetcher
from adj_partner.tools.executor import ToolFailure


ARTICLE = b'<article><front><article-meta><title-group><article-title>Fixture</article-title></title-group></article-meta></front><body><sec><title>Results</title><p>A &amp; B are observations.</p></sec></body></article>'
DECLARATION = b'<!DOCTYPE article\n PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving v1.4//EN" "JATS-archivearticle1-4-mathml3.dtd">'


@pytest.mark.parametrize('prefix', [b'\n', b'\xef\xbb\xbf', b'<?xml version="1.0" encoding="UTF-8"?>\n'])
def test_fetch_and_parse_external_jats_declaration_without_changing_source_bytes(prefix):
    raw = prefix + DECLARATION + b'\n' + ARTICLE
    fetcher = FullTextFetcher(transport=httpx.MockTransport(lambda _: httpx.Response(
        200, headers={'content-type':'application/xml'}, content=raw)),
        resolver=lambda _: ['93.184.216.34'])
    response = fetcher.fetch({'candidates':[{'acquisition':[{'kind':'jats',
        'url':'https://example.org/fulltext.xml', 'provider':'fixture', 'is_oa':True}]}]})
    assert response.body == raw
    doc = parse_jats(response.body)
    assert doc['sections'][0]['paragraphs'][0]['text'] == 'A & B are observations.'


def test_external_dtd_is_never_resolved(monkeypatch):
    visited = []
    original = etree.XMLParser
    class RejectAll(etree.Resolver):
        def resolve(self, url, pubid, context):
            visited.append(url)
            raise AssertionError('External DTD or entity accessed')
    def parser(**kwargs):
        assert kwargs['resolve_entities'] is False and kwargs['no_network'] is True and kwargs['load_dtd'] is False
        value = original(**kwargs)
        value.resolvers.add(RejectAll())
        return value
    monkeypatch.setattr('adj_partner.native.documents.etree.XMLParser', parser)
    doc = parse_jats(b'<!DOCTYPE article SYSTEM "https://invalid.example/never-read.dtd">'+ARTICLE)
    assert doc['title'] == 'Fixture'
    assert visited == []


@pytest.mark.parametrize('declaration', [
    b'<!DOCTYPE article [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>',
    b'<!DOCTYPE article SYSTEM "https://example.org/dtd" [<!ENTITY x "expanded">]>',
    b'<!DOCTYPE article [<!ENTITY % external SYSTEM "https://example.org/entities"> %external;]>',
    b'<!DOCTYPE article []>',
    DECLARATION + DECLARATION,
])
def test_custom_internal_or_duplicate_declarations_still_rejected(declaration):
    with pytest.raises(ToolFailure) as failure:
        parse_jats(declaration + ARTICLE)
    assert failure.value.code == 'document_xml_unsafe'


def test_xml_error_envelope_is_not_a_jats_article():
    with pytest.raises(ToolFailure) as failure:
        parse_jats(b'<?xml version="1.0"?><error>Not available</error>')
    assert failure.value.code == 'document_xml'


@pytest.mark.parametrize('encoding', ['utf-16', 'utf-32'])
def test_entity_checks_cannot_be_bypassed_by_null_encoded_xml(encoding):
    hostile = '<!DOCTYPE article [<!ENTITY x "hidden">]><article><body><p>&x;</p></body></article>'
    with pytest.raises(ToolFailure) as failure:
        parse_jats(hostile.encode(encoding))
    assert failure.value.code == 'document_xml_unsafe'
