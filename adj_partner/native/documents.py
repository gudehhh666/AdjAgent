"""Pure bounded JATS/PDF parsing with stable text locators."""

from io import BytesIO
import json
import re
import unicodedata

from lxml import etree
from pypdf import PdfReader

from ..tools.executor import ToolFailure
from .scholarly import normalize_doi
from .jats import inert_jats_content


MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
MAX_XML_NODES = 200_000
MAX_TEXT_CHARACTERS = 4_000_000
MAX_PDF_PAGES = 500
MAX_PAGE_STREAM_BYTES = 16 * 1024 * 1024


def _text(node, maximum=200_000):
    if node is None:
        return ""
    value = " ".join("".join(node.itertext()).split())
    if len(value) > maximum:
        raise ToolFailure("document_text_size", "Extracted document field exceeds its text limit")
    return value


def _first(node, expression):
    values = node.xpath(expression)
    return values[0] if values else None


def parse_jats(content):
    if not isinstance(content, bytes) or not content or len(content) > MAX_DOCUMENT_BYTES:
        raise ToolFailure("document_size", "JATS input is empty or exceeds 32 MiB")
    try:
        content = inert_jats_content(content)
    except ValueError as exc:
        raise ToolFailure("document_xml_unsafe", str(exc)) from exc
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False,
                             huge_tree=False, remove_comments=True, recover=False)
    try:
        root = etree.fromstring(content, parser=parser)
    except (etree.XMLSyntaxError, ValueError) as exc:
        raise ToolFailure("document_xml", "JATS XML is malformed or exceeds parser limits") from exc
    if etree.QName(root).localname != "article":
        raise ToolFailure("document_xml", "JATS XML root must be an article, not an error or other document")
    nodes = root.xpath("descendant-or-self::*")
    if len(nodes) > MAX_XML_NODES:
        raise ToolFailure("document_nodes", "JATS document exceeds the XML node limit")
    title = _text(_first(root, ".//*[local-name()='article-title']"))
    abstract = []
    for index, paragraph in enumerate(root.xpath(".//*[local-name()='abstract']//*[local-name()='p']"), 1):
        value = _text(paragraph)
        if value:
            abstract.append({"locator": f"abstract:paragraph:{index}", "text": value})
    sections = []
    section_nodes = root.xpath(".//*[local-name()='body']//*[local-name()='sec']")
    if not section_nodes:
        body = _first(root, ".//*[local-name()='body']")
        section_nodes = [body] if body is not None else []
    for section_index, section in enumerate(section_nodes, 1):
        section_title = _text(_first(section, "./*[local-name()='title']")) or f"Section {section_index}"
        paragraphs = []
        # Direct paragraphs keep nested subsections from duplicating passages.
        for paragraph_index, paragraph in enumerate(section.xpath("./*[local-name()='p']"), 1):
            value = _text(paragraph)
            if value:
                paragraphs.append({"locator": f"section:{section_index}/paragraph:{paragraph_index}",
                                   "text": value})
        ancestors = section.xpath("ancestor::*[local-name()='sec']/*[local-name()='title']")
        section_path = [_text(node, 2_000) for node in ancestors] + [section_title]
        sections.append({"locator": f"section:{section_index}", "title": section_title,
                         "section_path": section_path, "paragraphs": paragraphs})
    figures = []
    for index, figure in enumerate(root.xpath(".//*[local-name()='fig']"), 1):
        graphic = _first(figure, ".//*[local-name()='graphic']")
        href = ""
        if graphic is not None:
            href = graphic.get("{http://www.w3.org/1999/xlink}href") or graphic.get("href") or ""
        figures.append({"locator": f"figure:{index}", "label": _text(_first(figure, "./*[local-name()='label']")),
                        "caption": _text(_first(figure, "./*[local-name()='caption']")),
                        "graphic_href": href[:4096],
                        "text_availability": "caption_and_metadata_only"})
    tables = []
    for index, table_wrap in enumerate(root.xpath(".//*[local-name()='table-wrap']"), 1):
        rows = []
        for row in table_wrap.xpath(".//*[local-name()='tr']")[:1000]:
            cells = [_text(cell, 20_000) for cell in row.xpath("./*[local-name()='th' or local-name()='td']")[:100]]
            rows.append(cells)
        tables.append({"locator": f"table:{index}", "label": _text(_first(table_wrap, "./*[local-name()='label']")),
                       "caption": _text(_first(table_wrap, "./*[local-name()='caption']")), "rows": rows})
    references = []
    for index, reference in enumerate(root.xpath(".//*[local-name()='ref-list']/*[local-name()='ref']")[:5000], 1):
        item = {"locator": f"reference:{index}", "title":
                _text(_first(reference, ".//*[local-name()='article-title']"))}
        doi_node = _first(reference, ".//*[local-name()='pub-id' and translate(@pub-id-type,'DOI','doi')='doi']")
        doi = normalize_doi(_text(doi_node, 1000))
        if doi:
            item["doi"] = doi
        references.append(item)
    document = {"document_type": "jats", "title": title, "abstract": abstract,
                "sections": sections, "figures": figures, "tables": tables,
                "references": references,
                "limitations": ["Figure content was not visually interpreted; only captions and graphic metadata were extracted."]}
    if sum(len(item["text"]) for item in abstract) + sum(len(p["text"]) for s in sections for p in s["paragraphs"]) > MAX_TEXT_CHARACTERS:
        raise ToolFailure("document_text_size", "Extracted JATS text exceeds the aggregate character limit")
    return document


def parse_pdf(content):
    if not isinstance(content, bytes) or not content or len(content) > MAX_DOCUMENT_BYTES:
        raise ToolFailure("document_size", "PDF input is empty or exceeds 32 MiB")
    if not content.startswith(b"%PDF-"):
        raise ToolFailure("document_pdf", "PDF input lacks a PDF signature")
    try:
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted:
            raise ToolFailure("document_pdf_encrypted", "Encrypted PDFs are not supported")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ToolFailure("document_pages", "PDF exceeds the page limit")
        pages, total = [], 0
        for index, page in enumerate(reader.pages, 1):
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > MAX_PAGE_STREAM_BYTES:
                raise ToolFailure("document_page_size", "PDF page content exceeds the stream limit")
            extracted = page.extract_text() or ""
            extracted = "\n".join(line.rstrip() for line in extracted.splitlines()).strip()
            total += len(extracted)
            if total > MAX_TEXT_CHARACTERS:
                raise ToolFailure("document_text_size", "Extracted PDF text exceeds the character limit")
            pages.append({"locator": f"page:{index}", "page": index, "text": extracted,
                          "text_availability": "extracted_text" if extracted else "image_only_or_empty"})
    except ToolFailure:
        raise
    except Exception as exc:
        raise ToolFailure("document_pdf", f"PDF parsing failed: {type(exc).__name__}") from exc
    return {"document_type": "pdf", "pages": pages,
            "limitations": ["PDF text extraction does not perform OCR; empty pages may contain images or scanned text."]}


def parse_document(content, media_type):
    if media_type in {"application/xml", "text/xml", "application/jats+xml"}:
        return parse_jats(content)
    if media_type == "application/bioc+json":
        from .bioc import parse_bioc
        return parse_bioc(content)
    if media_type == "application/pdf":
        return parse_pdf(content)
    raise ToolFailure("document_media", "Unsupported full-text media type")


def _passages(document):
    passages = []
    if document.get("document_type") == "pdf":
        for index, page in enumerate(document.get("pages", [])):
            pointer = f"/pages/{index}"
            info = {"page": page["page"], "text_availability": page["text_availability"]}
            if document.get("mode") != "tables":
                passages.append({**info, "locator": page["locator"], "text": page["text"],
                                 "selector": {"pointer": pointer + "/text"}})
            for ti, table in enumerate(page.get("tables", [])):
                for ri, row in enumerate(table.get("rows", [])):
                    row_path = f"{pointer}/tables/{ti}/rows/{ri}"
                    passages.append({**info, "locator": row["locator"], "text": row["text"],
                                     "selector": {"pointer": row_path + "/text"}})
                    for ci, cell in enumerate(row.get("cells", [])):
                        if cell["text"]:
                            passages.append({**info, **cell,
                                "selector": {"pointer": f"{row_path}/cells/{ci}/text"}})
        return passages
    for index, paragraph in enumerate(document.get("abstract", [])):
        passages.append({**paragraph, "selector": {"pointer": f"/abstract/{index}/text"}})
    for section_index, section in enumerate(document.get("sections", [])):
        for paragraph_index, paragraph in enumerate(section.get("paragraphs", [])):
            if paragraph.get("bioc_type", "").startswith("title"):
                continue
            passages.append({**paragraph, "section": section.get("title", ""),
                "section_path": section.get("section_path", [section.get("title", "")]),
                "selector": {"pointer": f"/sections/{section_index}/paragraphs/{paragraph_index}/text"}})
    for figure_index, figure in enumerate(document.get("figures", [])):
        if figure.get("caption"):
            passages.append({"locator": figure["locator"], "text": figure["caption"],
                             "selector": {"pointer": f"/figures/{figure_index}/caption"},
                             "text_availability": "caption_only"})
    for table_index, table in enumerate(document.get("tables", [])):
        value = "\n".join([table.get("caption", "")] + [" | ".join(row) for row in table.get("rows", [])]).strip()
        if value:
            passages.append({"locator": table["locator"], "text": value,
                "selector": {"pointer": f"/tables/{table_index}/caption"} if value == table.get("caption", "") else None,
                "selection_note": "Use a returned selector only when non-null; table-row cells remain available by JSON pointer."})
    for reference_index, reference in enumerate(document.get("references", [])):
        value = " ".join(part for part in (reference.get("title", ""), reference.get("doi", "")) if part)
        if value:
            passages.append({"locator": reference["locator"], "text": value,
                "selector": {"pointer": f"/references/{reference_index}/title"} if value == reference.get("title", "") else None,
                "selection_note": "Use a returned selector only when non-null; reference fields remain available by JSON pointer."})
    return passages


def query_document(document, *, query="", page=None, section=None, cursor=0, limit=8):
    if not isinstance(document, dict) or type(cursor) is not int or cursor < 0 or type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("Invalid document query")
    if not isinstance(query, str) or len(query) > 1000:
        raise ValueError("Document query exceeds 1000 characters")
    passages = _passages(document)
    ignored_filters = []
    if page is not None:
        if type(page) is not int or page < 1:
            raise ValueError("Page must be a positive integer")
        if document.get("document_type") == "pdf":
            passages = [item for item in passages if item.get("page") == page]
        else:
            ignored_filters.append("page")
    if section is not None:
        if not isinstance(section, str) or len(section) > 200:
            raise ValueError("Section filter is invalid")
        if document.get("document_type") in {"jats", "bioc"}:
            heading = unicodedata.normalize("NFKC", section).strip().casefold()
            def same_heading(value):
                return unicodedata.normalize("NFKC", value).strip().casefold() == heading

            if heading and any(same_heading(s.get("title", "")) for s in document.get("sections", [])):
                passages = [item for item in passages if any(same_heading(part)
                    for part in item.get("section_path", []))
                    or heading == "abstract" and item["locator"].startswith("abstract:")]
            else:
                # Retain the existing multi-section search when no exact heading was given.
                keys = re.findall(r"\w+", heading)
                passages = [item for item in passages if any(
                    key in unicodedata.normalize("NFKC", " ".join(item.get("section_path", []))).casefold()
                    or key == "abstract" and item.get("locator", "").startswith("abstract:") for key in keys)]
        else:
            ignored_filters.append("section")
    terms = list(dict.fromkeys(term for term in re.findall(
        r"\w+", unicodedata.normalize("NFKC", query).casefold()) if len(term) >= 2))
    if terms:
        ranked = []
        for index, item in enumerate(passages):
            normalized = unicodedata.normalize("NFKC", item["text"]).casefold()
            text_terms = set(re.findall(r"\w+", normalized))
            matched = [term for term in terms if term in text_terms]
            if matched:
                ranked.append((-len(matched), index, {**item, "match_score": len(matched),
                                                       "matched_terms": matched}))
        passages = [item for _, _, item in sorted(ranked)]
    selected = passages[cursor:cursor + limit]
    next_cursor = cursor + len(selected) if cursor + len(selected) < len(passages) else None
    result = {"matches": selected, "returned": len(selected), "total_matches": len(passages),
              "next_cursor": next_cursor}
    if ignored_filters:
        result["ignored_filters"] = ignored_filters
    return result


class DocumentReader:
    def parse(self, content, media_type):
        return parse_document(content, media_type)

    def query(self, document, **arguments):
        return query_document(document, **arguments)


def prepare_readable_document(content, media_type, provenance, reader):
    """Normalize readable material without upgrading abstract evidence to full text."""
    availability=provenance.get('text_availability')
    if availability=='full_text':
        return reader.parse(content,media_type),False,'derived_full_text'
    if media_type.split(';',1)[0].strip()!='application/json' or len(content)>32*1024*1024:
        raise ToolFailure('document_media','Parsed documents and abstracts require bounded ingested JSON')
    try:
        document=json.loads(content)
    except (ValueError,UnicodeError) as exc:
        raise ToolFailure('document_json','Invalid document JSON') from exc
    if not isinstance(document,dict):
        raise ToolFailure('document_json','Document must be an object')
    if availability=='derived_full_text':
        if provenance.get('provider')!='document_parser' or document.get('document_type') not in {'jats','pdf','bioc'}:
            raise ToolFailure('document_provenance','Expected a host-parsed full-text document')
        return document,True,'derived_full_text'
    if availability!='abstract':
        raise ToolFailure('document_media','No readable text: titles and search snippets are not documents')
    if provenance.get('provider')=='document_parser' and document.get('document_type')=='abstract':
        return document,True,'abstract'
    abstract=document.get('abstract')
    if isinstance(abstract,str):texts=[abstract] if abstract.strip() else []
    elif isinstance(abstract,list):texts=[p['text'] for p in abstract if isinstance(p,dict) and isinstance(p.get('text'),str) and p['text'].strip()]
    else:texts=[]
    if not texts:raise ToolFailure('document_no_abstract','This source has no readable abstract text')
    return {'document_type':'abstract','title':document.get('title',''),
            'abstract':[{'text':t,'locator':f'abstract:{i}','text_availability':'abstract'} for i,t in enumerate(texts,1)],
            'sections':[],'figures':[],'tables':[],'references':[],
            'limitations':list(provenance.get('limitations',[]))+['Abstract only; the full paper was not read or verified.']},False,'abstract'
