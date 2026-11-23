"""Read-only user-file inspection; reading a file does not certify its contents."""

import csv
import io
import json
from pathlib import Path

from ..tools.catalog import canonical_json
from .safe_files import read_input


def inspect_geo_metadata(text):
    """Inspect GEO metadata rows without treating them as rectangular assay data.

    Repeated keys (e.g. characteristics) remain distinct. No sample grouping,
    biological interpretation, expression parsing or numerical validation occurs.
    """
    rows, body = [], False
    for row in csv.reader(io.StringIO(text), delimiter="\t", strict=True):
        if not row:
            continue
        if row[0] == "!series_matrix_table_begin":
            body = True
            break
        if not row[0].startswith(("!Series_", "!Sample_")) or len(row) < 2:
            raise ValueError("Malformed GEO metadata row")
        rows.append(row)
        if len(rows) > 1000:
            raise ValueError("GEO metadata exceeds 1000 rows")
    accessions = [r[1:] for r in rows if r[0] == "!Sample_geo_accession"]
    if len(accessions) != 1 or not accessions[0] or any(not s.strip() for s in accessions[0]):
        raise ValueError("GEO requires one nonempty sample accession row")
    ids = accessions[0]
    if len(ids) > 50000:
        raise ValueError("GEO metadata exceeds 50000 sample columns")
    occurrences, fields = {}, []
    for row in rows:
        if row[0].startswith("!Sample_"):
            if len(row) != len(ids) + 1:
                raise ValueError("GEO sample metadata width does not match accession row")
            occurrences[row[0]] = occurrences.get(row[0], 0) + 1
            fields.append({"name": row[0][:100], "occurrence": occurrences[row[0]],
                           "value_examples": [x[:100] for x in row[1:3]]})
    return {"format": "geo_series_matrix_metadata", "sample_columns": len(ids),
            "unique_sample_accessions": len(set(ids)),
            "duplicate_sample_accessions": len(ids) - len(set(ids)),
            "sample_fields": fields[:8], "sample_fields_omitted": max(0, len(fields) - 8),
            "metadata_rows": len(rows), "matrix_body_present": body,
            "expression_values_inspected": False,
            "preview_policy": "first8 sample fields, first2 values, 100 chars; repeated keys preserved",
            "limitations": ["Sample columns are not independent people. Only metadata rows inspected; no expression or biological validation."]}


def inspect_input(context, path):
    item = read_input(context, path, maximum_bytes=2 * 1024 * 1024)
    try:
        text = item.content.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("File inspection supports UTF-8 text/JSON/CSV/TSV, not arbitrary binary formats") from exc
    extension = Path(item.path).suffix.lower()
    first_line = text.lstrip().split("\n", 1)[0]
    if extension in {".txt", ".tsv"} and first_line.startswith(("!Series_", "!Sample_")):
        # Redact before bounded previews, so truncation cannot expose a secret prefix.
        inspection = inspect_geo_metadata(context.store.redact(text))
        media_type = "text/tab-separated-values"
    elif extension == ".json":
        value = json.loads(text)
        canonical_json(value)
        cleaned = context.store.redact(value)
        encoded = canonical_json(cleaned)
        inspection = {"format": "json", "root_type": type(value).__name__,
                      "preview": encoded[:2000], "preview_truncated": len(encoded) > 2000}
        if isinstance(value, dict):
            inspection.update(keys=[key[:200] for key in list(cleaned)[:40]], key_count=len(value),
                              labels_truncated=len(value) > 40 or any(len(key) > 200 for key in cleaned))
        if isinstance(value, list):
            inspection["items"] = len(value)
        media_type = "application/json"
    elif extension in {".csv", ".tsv"}:
        reader = csv.reader(io.StringIO(text), delimiter="\t" if extension == ".tsv" else ",")
        header = next(reader, [])
        if not header or len(header) > 256:
            raise ValueError("CSV/TSV requires a header with 1..256 columns")
        safe_header = context.store.redact(header)
        preview, rows = [], 0
        for row in reader:
            rows += 1
            if rows > 50000 or len(row) != len(header):
                raise ValueError("CSV/TSV exceeds 50000 rows or has inconsistent row width")
            if len(preview) < 5:
                # Redact full cells (and sensitive column names) before any
                # length cut can remove part of a known credential pattern.
                safe_row = [next(iter(context.store.redact({label: cell}).values())) for label, cell in zip(header, row)]
                preview.append([cell[:200] for cell in safe_row[:20]])
        inspection = {"format": extension[1:], "columns": [label[:200] for label in safe_header[:40]], "column_count": len(header),
                      "labels_truncated": len(header) > 40 or any(len(label) > 200 for label in safe_header),
                      "rows": rows, "preview": preview,
                      "preview_policy": "first5rows/20columns/200characters_per_cell; headers_first40"}
        media_type = "text/tab-separated-values" if extension == ".tsv" else "text/csv"
    else:
        inspection = {"format": "text", "characters": len(text), "lines": len(text.splitlines()),
                      "preview": context.store.redact(text)[:2000], "preview_truncated": len(text) > 2000}
        media_type = "text/plain"
    return item, context.store.redact(inspection), media_type
