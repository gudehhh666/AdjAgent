"""Remove one inert external article declaration from a local XML parse copy."""
import re


_PROLOG = re.compile(br'\A(?:\xef\xbb\xbf)?[ \t\r\n]*(?:<\?xml[ \t\r\n][^?]*\?>[ \t\r\n]*)?')
_EXTERNAL = re.compile(br'''<!DOCTYPE\s+article\s+(?:PUBLIC\s+(?:"[^"<>\[\]]*"|'[^'<>\[\]]*')\s+|SYSTEM\s+)(?:"[^"<>\[\]]*"|'[^'<>\[\]]*')\s*>''')
_DECLARATION = re.compile(br'<!\s*(?:DOCTYPE|ENTITY)\b', re.IGNORECASE)


def inert_jats_content(content):
    """Never load a DTD. Internal subsets, entities and misplaced DTDs are rejected.

    Downloaded source bytes remain untouched; callers use this copy for sniffing
    and parsing with entity expansion, external loading and network disabled.
    """
    # Byte-level declaration checks require an ASCII-compatible XML encoding.
    # Reject UTF-16/32 rather than letting null-interleaved declarations evade them.
    if b'\x00' in content:
        raise ValueError('JATS requires an ASCII-compatible XML byte encoding')
    if not _DECLARATION.search(content):
        return content
    start = _PROLOG.match(content).end()
    declaration = _EXTERNAL.match(content, start)
    if declaration is None:
        raise ValueError('JATS contains an internal, entity, or misplaced declaration')
    normalized = content[:start] + content[declaration.end():]
    if _DECLARATION.search(normalized):
        raise ValueError('JATS contains additional DTD or entity declarations')
    return normalized


def matched_article_ids(root, expected):
    """Match only front-matter identifiers; references can describe other papers."""
    matched, version = set(), None
    ids = root.xpath("./*[local-name()='front']/*[local-name()='article-meta']/*[local-name()='article-id']")
    for node in ids:
        key, value = node.get("pub-id-type"), (node.text or "").strip()
        if key == "pmcid-ver" and expected.get("pmcid"):
            match = re.fullmatch(re.escape(expected["pmcid"]) + r"\.([1-9][0-9]{0,5})", value)
            if not match or (version is not None and version != int(match[1])):
                raise ValueError("JATS version conflicts with the resolved article identity")
            version = int(match[1])
            matched.add(key)
        if key in {"pmc", "pmcid"}:
            key, value = "pmcid", "PMC" + value.upper().removeprefix("PMC")
        if key in {"doi", "pmid", "pmcid"} and expected.get(key):
            if str(expected[key]).casefold() != value.casefold():
                raise ValueError(f"JATS {key} conflicts with the resolved article identity")
            matched.add(key)
    return matched, version
