"""Real bounded sequence/structure inspection, not prediction or efficacy scoring."""

from collections import Counter
import hashlib
import io
from threading import RLock
import warnings

import Bio
from Bio import SeqIO
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis


CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
SEQUENCE_LIMIT = 100000
# Python 3.12 warning contexts are process-global. Serialize this package's
# structure parsers; production calls additionally run in isolated job children.
# Direct embedding alongside unrelated warning-context users is not supported.
_STRUCTURE_WARNING_LOCK = RLock()
SEQUENCE_LIMITATIONS = [
    "Theoretical sequence-derived properties, not experimental measurements.",
    "Only 20 canonical amino acids; unmodified linear chain with free termini and average isotopic masses.",
    "No post-translational modifications, covalent assemblies or nonstandard residues modeled.",
    "No true structure, biological function, affinity, safety or efficacy inferred.",
]


def sequence_properties(sequence):
    if not isinstance(sequence, str) or len(sequence) > 2 * SEQUENCE_LIMIT or not sequence.isascii():
        raise ValueError("Sequence must be bounded ASCII amino-acid text")
    normalized = "".join(sequence.split()).upper()
    if not normalized or len(normalized) > SEQUENCE_LIMIT:
        raise ValueError("Sequence length must be 1..100000 residues")
    unknown = sorted(set(normalized) - set(CANONICAL))
    if unknown:
        raise ValueError("Ambiguous/nonstandard residues, stops and non-amino-acid symbols are unsupported: " + repr(unknown))
    analysis = ProteinAnalysis(normalized, monoisotopic=False)
    counts = Counter(normalized)
    return {"length": len(normalized), "composition": {aa: counts[aa] for aa in CANONICAL},
            "sequence_sha256": hashlib.sha256(normalized.encode("ascii")).hexdigest(),
            "molecular_weight_da": analysis.molecular_weight(),
            "theoretical_isoelectric_point": analysis.isoelectric_point(),
            "gravy": analysis.gravy(), "aromaticity": analysis.aromaticity(),
            "method": "Bio.SeqUtils.ProtParam.ProteinAnalysis",
            "package_versions": {"biopython": Bio.__version__},
            "limitations": list(SEQUENCE_LIMITATIONS)}


def _text(content, maximum):
    if not isinstance(content, bytes) or len(content) > maximum:
        raise ValueError("Input must be bounded bytes")
    try:
        return content.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("Input is not valid UTF-8 text") from exc


def parse_fasta(content):
    text = _text(content, 2 * 1024 * 1024)
    if not text.strip().startswith(">"):
        raise ValueError("FASTA must start with a record header")
    results, identifiers, total = [], set(), 0
    for record in SeqIO.parse(io.StringIO(text.lstrip()), "fasta"):
        if not record.id or record.id in identifiers or len(record.id) > 256 or len(record.description) > 2000:
            raise ValueError("FASTA identifiers must be unique and bounded")
        identifiers.add(record.id)
        properties = sequence_properties(str(record.seq))
        total += properties["length"]
        if len(results) >= 16 or total > 200000:
            raise ValueError("FASTA limit is 16 records and 200000 total residues")
        results.append({"id": record.id, "description": record.description, "properties": properties})
    if not results:
        raise ValueError("FASTA contains no records")
    return results


def inspect_structure(content, format):
    text = _text(content, 8 * 1024 * 1024)
    if format not in {"pdb", "mmcif"}:
        raise ValueError("Supported structure formats are pdb and mmcif")
    parser_warnings, warning_count = [], 0

    def capture_warning(message, category, filename, lineno, file=None, line=None):
        nonlocal warning_count
        warning_count += 1
        if len(parser_warnings) < 50:
            parser_warnings.append(str(message)[:500])

    try:
        with _STRUCTURE_WARNING_LOCK, warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = capture_warning
            parser = PDBParser(PERMISSIVE=True, QUIET=False) if format == "pdb" else MMCIFParser(QUIET=False)
            structure = parser.get_structure("input", io.StringIO(text))
            counts = {"models": sum(1 for _ in structure.get_models()),
                      "chains": sum(1 for _ in structure.get_chains()),
                      "residues": sum(1 for _ in structure.get_residues()),
                      "atoms": sum(1 for _ in structure.get_atoms())}
    except Exception as exc:
        raise ValueError(f"Structure parsing failed: {type(exc).__name__}: {str(exc)[:1000]}") from exc
    if not counts["atoms"]:
        raise ValueError("Structure contains no parsed atoms")
    return {**counts, "format": format, "input_sha256": hashlib.sha256(content).hexdigest(),
            "method": "Bio.PDB." + type(parser).__name__, "package_versions": {"biopython": Bio.__version__},
            "parser_warnings": parser_warnings, "parser_warning_count": warning_count,
            "warnings_truncated": warning_count > len(parser_warnings),
            "limitations": ["File inspection only, not structure prediction or experimental validation.",
                            "Input may be synthetic, modeled, incomplete or contain parser-discarded records.",
                            "Counts follow Bio.PDB model/chain/residue/selected-disordered-atom semantics; include hetero residues and water.",
                            "No function, affinity, stability, safety or efficacy inferred."]}
