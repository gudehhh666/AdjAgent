"""On-demand, local extraction of a single PDF page in a bounded worker."""

import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


TIMEOUT_SECONDS = 40
MAX_CELLS = 2048
MAX_PAGE_TEXT = 120_000
# Match the whole-document and scalar limits of the host's JSON pointer reader.
MAX_RESULT_BYTES = 1_000_000
MAX_SCALAR_BYTES = 100_000
MAX_WORKER_MEMORY = 768 * 1024 * 1024
MAX_RENDER_EDGE = 3456  # At most 11,943,936 grayscale pixels.
OCR_DPI = 250
INSTALL_HINT = ("Install adj_partner/native/requirements-pdf.txt into the active interpreter "
                "or its pdf-tools virtual environment. OCR requires Tesseract with English data on PATH.")


class _PageError(Exception):
    """Only this worker's controlled code and message cross the process boundary."""


def _worker_python():
    companion = Path(sys.prefix) / "pdf-tools" / "bin" / "python"
    return str(companion) if companion.is_file() else sys.executable


def parse_pdf_page(content: bytes, page: int, mode: str, check_cancelled=None):
    """Return page text and, for tables, citeable row/cell values from original page N."""
    from ..tools.executor import ToolFailure
    from .documents import MAX_DOCUMENT_BYTES, MAX_PAGE_STREAM_BYTES, MAX_PDF_PAGES
    if sys.platform == "darwin":
        from ..analysis.sandbox import _owned_process_metrics

    if check_cancelled is not None:
        check_cancelled()
    if not isinstance(content, bytes) or not content or len(content) > MAX_DOCUMENT_BYTES:
        raise ToolFailure("document_size", "PDF input is empty or exceeds 32 MiB")
    if not content.startswith(b"%PDF-"):
        raise ToolFailure("document_pdf", "PDF input lacks a PDF signature")
    if type(page) is not int or not 1 <= page <= MAX_PDF_PAGES:
        raise ToolFailure("document_page", "Select one positive page within the PDF page limit")
    if mode not in ("tables", "ocr"):
        raise ToolFailure("document_mode", "PDF enhancement mode must be tables or ocr")

    with tempfile.TemporaryDirectory(prefix="adj-pdf-") as directory:
        work = Path(directory)
        (work / "input.pdf").write_bytes(content)
        interpreter = _worker_python()
        # Keep credentials, Python startup customizations and external plugin settings
        # out of the worker. Venv binaries also work when the caller did not activate it.
        path = os.pathsep.join([str(Path(interpreter).parent), str(Path(sys.executable).parent),
                                os.environ.get("PATH", os.defpath)])
        env = {"PATH": path, "TMPDIR": directory, "HOME": directory, "OMP_THREAD_LIMIT": "1",
               "XDG_CACHE_HOME": directory, "LANG": "en_US.UTF-8"}
        if os.environ.get("TESSDATA_PREFIX"):
            env["TESSDATA_PREFIX"] = os.environ["TESSDATA_PREFIX"]
        deadline = time.monotonic() + TIMEOUT_SECONDS
        process = subprocess.Popen([interpreter, "-I", str(Path(__file__).resolve()), mode, str(page),
                                    str(MAX_PDF_PAGES), str(MAX_PAGE_STREAM_BYTES)],
                                   cwd=work, env=env, start_new_session=True,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            while process.poll() is None:
                if check_cancelled is not None:
                    check_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ToolFailure("document_pdf_timeout", "Single-page PDF extraction reached its time limit")
                if sys.platform == "darwin" and _owned_process_metrics(process.pid)[1] > MAX_WORKER_MEMORY:
                    raise ToolFailure("document_pdf_memory", "PDF extraction exceeded its worker memory limit")
                try:
                    process.wait(timeout=min(remaining, 0.25))
                except subprocess.TimeoutExpired:
                    continue
            if check_cancelled is not None:
                check_cancelled()
        finally:
            if process.returncode != 0:
                # Include OCR grandchildren if the worker crashed or the caller cancelled.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
        output = work / "result.json"
        if process.returncode or not output.is_file() or output.stat().st_size > MAX_RESULT_BYTES:
            raise ToolFailure("document_pdf_worker", "Single-page PDF extraction failed or exceeded its resource limit")
        result = json.loads(output.read_bytes())
        if "error" in result:
            raise ToolFailure(*result["error"])
        return result


def _box(coords):
    return [round(float(value), 3) for value in coords] if coords is not None else None


def _table_page(number, result_page):
    import pdfplumber

    tables, lines = [], []
    cell_count = 0
    with pdfplumber.open("input.pdf", pages=[number]) as pdf:
        page = pdf.pages[0]
        if len(page.chars) > MAX_PAGE_TEXT or len(page.edges) > 800:
            raise _PageError("document_table_limit", "PDF page exceeds the character or ruling complexity limit")
        for index, table in enumerate(page.find_tables(), 1):
            rows = table.rows
            cell_count += sum(len(row.cells) for row in rows)
            if cell_count > MAX_CELLS:
                raise _PageError("document_table_limit", f"PDF table grid exceeds {MAX_CELLS} cell positions")
            table_locator = f"page:{number}/table:{index}"
            extracted = []
            for row_num, (row, values) in enumerate(zip(rows, table.extract(), strict=True), 1):
                row_locator = f"{table_locator}/row:{row_num}"
                cells = [{"locator": f"{row_locator}/cell:{col}", "column": col, "text": value or "",
                          "bbox": _box(box)} for col, (value, box) in
                         enumerate(zip(values, row.cells, strict=True), 1)]
                texts = [cell["text"] for cell in cells]
                row_text = " | ".join(texts) if any(texts) else ""
                extracted.append({"locator": row_locator, "text": row_text, "cells": cells})
                if row_text:
                    lines.append(row_text)
            tables.append({"locator": table_locator, "bbox": _box(table.bbox), "rows": extracted})
        availability = "table_structure_only" if tables else "no_tables_detected"
        if lines:
            availability = "extracted_table_cells"
        result_page.update({"tables": tables, "text": "\n".join(lines), "width": float(page.width),
                            "height": float(page.height),
                            "bbox_frame": "pdfplumber PDF points [x0, top, x1, bottom] in the page_bbox frame",
                            "page_bbox": _box(page.bbox),
                            "text_availability": availability})
    return ("pdfplumber", ["Only ruled table grids and their embedded text on the requested page were extracted. "
                           "Borderless tables, merged cells and image-only values may be missed."])


def _ocr_page(number, result_page):
    from importlib.util import find_spec
    import pypdfium2 as pdfium

    if find_spec("ocrmypdf") is None or not shutil.which("tesseract"):
        raise _PageError("document_pdf_backend", INSTALL_HINT)
    # OCRmyPDF receives one bounded raster, never the multi-page input or an
    # existing hidden text layer. PDFium reads only the selected page for rendering.
    with pdfium.PdfDocument("input.pdf") as pdf:
        page = pdf[number - 1]
        width, height = page.get_size()
        scale = OCR_DPI / 72
        if not all(math.isfinite(v) and 0 < v <= 14_400 for v in (width, height)):
            raise _PageError("document_page_size", "PDF page dimensions exceed the render limit")
        scale = min(scale, MAX_RENDER_EDGE / max(width, height))
        bitmap = page.render(scale=scale, grayscale=True)
        raster_size = [bitmap.width, bitmap.height]
        bitmap.to_pil().save("page.png")
        bitmap.close()
        page.close()
    dpi = max(1, round(scale * 72))
    command = [sys.executable, "-I", "-m", "ocrmypdf", "--output-type", "none", "--sidecar", "page.txt",
               "--rasterizer", "pypdfium", "--image-dpi", str(dpi), "--jobs", "1", "-O", "0",
               "--tesseract-timeout", str(TIMEOUT_SECONDS + 5), "-l", "eng", "page.png", os.devnull]
    status = subprocess.run(command, check=False, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if status == 3:
        raise _PageError("document_pdf_backend", INSTALL_HINT)
    if status or not Path("page.txt").is_file():
        raise _PageError("document_pdf_ocr", f"OCRmyPDF failed to produce page text (exit {status})")
    sidecar = Path("page.txt")
    if sidecar.stat().st_size > MAX_PAGE_TEXT * 4:
        raise _PageError("document_text_size", "OCR text exceeds the page limit")
    value = sidecar.read_text(encoding="utf-8").strip()
    if value == "[OCR skipped on page(s) 1]":
        raise _PageError("document_pdf_ocr", "OCRmyPDF skipped the requested page")
    result_page.update({"text": value, "text_availability": "ocr_text" if value else "ocr_no_text",
                        "width": width, "height": height, "raster_pixels": raster_size,
                        "raster_dpi": dpi, "ocr_language": "eng"})
    return ("OCRmyPDF/Tesseract", ["English OCR of the requested page may misread numbers, symbols or reading order."])


def _extract(mode, number, page_limit, stream_limit):
    from pypdf import PdfReader

    reader = PdfReader("input.pdf", strict=True)
    if reader.is_encrypted:
        raise _PageError("document_pdf_encrypted", "Encrypted PDFs are not supported")
    page_count = len(reader.pages)
    if page_count > page_limit:
        raise _PageError("document_pages", "PDF exceeds the page limit")
    if number > page_count:
        raise _PageError("document_page", f"Requested page {number} exceeds the {page_count}-page PDF")
    contents = reader.pages[number - 1].get_contents()
    if contents is not None and len(contents.get_data()) > stream_limit:
        raise _PageError("document_page_size", "PDF page content exceeds the stream limit")
    page = {"page": number, "locator": f"page:{number}"}
    backend, limitations = _table_page(number, page) if mode == "tables" else _ocr_page(number, page)
    if len(json.dumps(page["text"], ensure_ascii=False).encode("utf-8")) > MAX_SCALAR_BYTES:
        raise _PageError("document_text_size", "Extracted text exceeds the single-page evidence readback limit")
    return {"document_type": "pdf", "mode": mode, "backend": backend, "pages": [page], "limitations": limitations}


if __name__ == "__main__":
    import resource

    # Set limits in the dedicated single-threaded worker; children inherit them.
    limits = [(resource.RLIMIT_FSIZE, 64 * 1024 * 1024), (resource.RLIMIT_CPU, TIMEOUT_SECONDS)]
    if sys.platform.startswith("linux"):
        limits.append((resource.RLIMIT_AS, MAX_WORKER_MEMORY))
    for kind, soft in limits:
        hard = resource.getrlimit(kind)[1]
        resource.setrlimit(kind, (soft if hard == resource.RLIM_INFINITY else min(soft, hard), hard))
    try:
        response = _extract(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    except ImportError:
        response = {"error": ["document_pdf_backend", INSTALL_HINT]}
    except _PageError as exc:
        response = {"error": exc.args}
    except Exception as exc:
        response = {"error": ["document_pdf", f"PDF page extraction failed: {type(exc).__name__}"]}
    Path("result.json").write_text(json.dumps(response, ensure_ascii=False, allow_nan=False,
                                            separators=(",", ":")), encoding="utf-8")
