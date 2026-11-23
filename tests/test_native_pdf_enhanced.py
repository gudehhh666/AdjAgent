"""Offline regression checks for page-scoped PDF tables and OCR."""

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from adj_partner.tools.executor import ToolFailure


def _require_backend(package):
    from adj_partner.native.pdf_enhanced import _worker_python

    python = _worker_python()
    probe = subprocess.run([python, "-I", "-c",
                            "from importlib.util import find_spec; import sys; "
                            "sys.exit(find_spec(sys.argv[1]) is None)", package], check=False)
    if probe.returncode:
        pytest.skip(f"Optional package {package} is unavailable")
    if package == "ocrmypdf":
        path = os.pathsep.join([str(Path(python).parent), str(Path(sys.executable).parent),
                                os.environ.get("PATH", os.defpath)])
        if not shutil.which("tesseract", path=path):
            pytest.skip("Tesseract CLI is unavailable")


def _table_pdf(commands=None, height=140):
    # Page 2 has a ruled 2 x 2 table with an ordinary Helvetica text layer.
    writer = PdfWriter()
    writer.add_blank_page(width=220, height=140)
    page = writer.add_blank_page(width=220, height=height)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    drawing = DecodedStreamObject()
    drawing.set_data(commands if commands is not None else b"""
10 20 m 210 20 l S 10 70 m 210 70 l S 10 120 m 210 120 l S
10 20 m 10 120 l S 110 20 m 110 120 l S 210 20 m 210 120 l S
BT /F1 14 Tf 20 95 Td (Group) Tj 100 0 Td (Value) Tj ET
BT /F1 14 Tf 20 45 Td (Treated) Tj 100 0 Td (17.25) Tj ET
""")
    page[NameObject("/Contents")] = writer._add_object(drawing)
    target = io.BytesIO()
    writer.write(target)
    return target.getvalue()


def test_tables_keep_page_row_cell_text_and_coordinates():
    from adj_partner.native.pdf_enhanced import parse_pdf_page
    _require_backend("pdfplumber")

    document = parse_pdf_page(_table_pdf(), 2, "tables")
    assert document["document_type"] == "pdf" and document["mode"] == "tables"
    assert len(document["pages"]) == 1
    page = document["pages"][0]
    assert page["page"] == 2 and page["locator"] == "page:2"
    row = page["tables"][0]["rows"][1]
    assert row["text"] == "Treated | 17.25"
    cell = row["cells"][1]
    assert cell["text"] == "17.25" and cell["column"] == 2
    assert cell["locator"] == "page:2/table:1/row:2/cell:2"
    assert cell["bbox"] == pytest.approx([110, 70, 210, 120])
    blank = parse_pdf_page(_table_pdf(), 1, "tables")["pages"][0]
    assert blank["tables"] == [] and blank["text"] == ""
    assert blank["text_availability"] == "no_tables_detected"


def test_empty_grid_has_geometry_without_claiming_recognized_text():
    from adj_partner.native.pdf_enhanced import parse_pdf_page
    _require_backend("pdfplumber")

    grid = b"10 10 100 100 re S 10 60 m 110 60 l S 60 10 m 60 110 l S"
    page = parse_pdf_page(_table_pdf(grid), 2, "tables")["pages"][0]
    assert page["tables"][0]["rows"][0]["cells"][0]["bbox"]
    assert page["text"] == "" and page["text_availability"] == "table_structure_only"


def test_dense_grid_cannot_exceed_cell_limit():
    from adj_partner.native.pdf_enhanced import parse_pdf_page
    _require_backend("pdfplumber")

    lines = []
    for i in range(47):
        pos = i * 4 + 10
        lines.extend([f"{pos} 0 m {pos} 200 l S", f"0 {pos} m 200 {pos} l S"])
    with pytest.raises(ToolFailure) as error:
        parse_pdf_page(_table_pdf("\n".join(lines).encode(), height=210), 2, "tables")
    assert error.value.code == "document_table_limit"


def _scan_pdf():
    from PIL import Image, ImageDraw, ImageFont

    images = []
    for words in ("CONTROL 999", "SAMPLE 427"):
        image = Image.new("RGB", (1200, 400), "white")
        ImageDraw.Draw(image).text((80, 110), words, font=ImageFont.load_default(size=84), fill="black")
        images.append(image)
    output = io.BytesIO()
    images[0].save(output, format="PDF", resolution=200, save_all=True, append_images=images[1:],
                   creationDate=None, modDate=None)
    return output.getvalue()


def test_ocr_reads_only_requested_raster_page():
    from adj_partner.native.pdf_enhanced import parse_pdf_page
    _require_backend("ocrmypdf")
    raw = _scan_pdf()
    assert not PdfReader(io.BytesIO(raw)).pages[1].extract_text().strip()
    document = parse_pdf_page(raw, page=2, mode="ocr")
    assert document["backend"] == "OCRmyPDF/Tesseract"
    assert len(document["pages"]) == 1
    page = document["pages"][0]
    assert page["locator"] == "page:2" and page["text_availability"] == "ocr_text"
    assert "SAMPLE 427" in page["text"] and "999" not in page["text"]


@pytest.mark.parametrize(("mode", "package", "fixture", "expected", "pointer"), [
    ("tables", "pdfplumber", _table_pdf, "17.25", "/pages/0/tables/0/rows/1/cells/1/text"),
    ("ocr", "ocrmypdf", _scan_pdf, "SAMPLE 427", "/pages/0/text"),
], ids=["table-cell", "scan-page"])
def test_agent_service_evidence_readback(tmp_path, mode, package, fixture, expected, pointer):
    from adj_partner.application import AgentService
    from adj_partner.contracts import ToolCall
    from adj_partner.models import ScriptedPort

    _require_backend(package)
    raw = fixture()
    service = AgentService(tmp_path / "state", project_root=tmp_path, model=ScriptedPort([]))
    try:
        sid = service.create_session(allowed_tools=["tools_enable", "document_read", "evidence_read"])["id"]
        rid = service.submit(sid, "Read one PDF page")["id"]
        for name in ("document_read", "evidence_read"):
            result = service.executor.execute(ToolCall("enable-" + name, "tools_enable", {"name": name}), sid, rid)
            assert result.status == "ok", result.to_dict()
        artifact = service.store.put_artifact(raw, mode + ".pdf", "application/pdf", {}, session_id=sid, run_id=rid)
        original = service.research.register_source(sid, rid, artifact_id=artifact["id"], kind="literature",
            title="Generated " + mode, provenance={"text_availability": "full_text", "document_scope": "article"},
            idempotency_key=mode)
        located = service.executor.execute(ToolCall("locate", "document_read",
            {"source_id": original["id"], "page": 2, "mode": mode, "query": expected}), sid, rid)
        assert located.status == "ok", located.to_dict()
        match = next(item for item in located.data["matches"] if item["text"] == expected)
        assert match["selector"] == {"pointer": pointer}
        derived = service.research.get_source(sid, rid, located.data["source_id"])
        assert derived["id"] != original["id"] and derived["provenance"]["parent_source_id"] == original["id"]
        read = service.executor.execute(ToolCall("read", "evidence_read",
            {"source_id": derived["id"], "selector": match["selector"]}), sid, rid)
        assert read.status == "ok", read.to_dict()
        assert read.data["text"] == expected and read.data["artifact_hash"] == derived["artifact_hash"]
        assert read.data["parse_mode"] == mode and read.data["page"] == 2 and read.data["document_scope"] == "article"
        assert service.store.read_artifact(original["artifact_id"]) == raw
    finally:
        service.close()


@pytest.mark.parametrize(("raw", "page", "mode", "code"), [
    (b"bad", 1, "tables", "document_pdf"),
    (_table_pdf(), 3, "ocr", "document_page"),
    (_table_pdf(), 1, "text", "document_mode"),
    (_table_pdf(), True, "tables", "document_page"),
], ids=["signature", "page-out-of-range", "mode", "boolean-page"])
def test_target_validation(raw, page, mode, code):
    from adj_partner.native.pdf_enhanced import parse_pdf_page

    with pytest.raises(ToolFailure) as error:
        parse_pdf_page(raw, page, mode)
    assert error.value.code == code


def test_document_with_too_many_pages_is_refused_before_ocr():
    from adj_partner.native.documents import MAX_PDF_PAGES
    from adj_partner.native.pdf_enhanced import parse_pdf_page

    writer = PdfWriter()
    for _ in range(MAX_PDF_PAGES + 1):
        writer.add_blank_page(width=10, height=10)
    out = io.BytesIO()
    writer.write(out)
    with pytest.raises(ToolFailure) as error:
        parse_pdf_page(out.getvalue(), 1, "ocr")
    assert error.value.code == "document_pages"


def test_selected_page_must_fit_exact_evidence_scalar_limit(tmp_path, monkeypatch):
    from adj_partner.native import pdf_enhanced

    def wide_text(number, page):
        page["text"] = "汉" * 40_000
        return "fixture", []

    (tmp_path / "input.pdf").write_bytes(_table_pdf())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pdf_enhanced, "_ocr_page", wide_text)
    with pytest.raises(pdf_enhanced._PageError, match="single-page"):
        pdf_enhanced._extract("ocr", 2, 500, 16 * 1024 * 1024)


def test_cancel_interrupts_work_in_progress():
    from adj_partner.native.pdf_enhanced import parse_pdf_page

    calls = []

    def check():
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("request cancelled")

    with pytest.raises(RuntimeError, match="request cancelled"):
        parse_pdf_page(_table_pdf(), 1, "tables", check_cancelled=check)


def test_enhancement_deadline_is_enforced(monkeypatch):
    from adj_partner.native import pdf_enhanced

    monkeypatch.setattr(pdf_enhanced, "TIMEOUT_SECONDS", 0)
    with pytest.raises(ToolFailure) as error:
        pdf_enhanced.parse_pdf_page(_table_pdf(), 1, "ocr")
    assert error.value.code == "document_pdf_timeout"
