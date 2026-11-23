"""Version 3: continuous authored narrative followed by deduplicated audit data.

Labels and anchors are host-generated. Authored content uses the same inert-text
escaping as older renderers; this module never changes the report's evidence.
"""

from .rendering import (authored, _citation_state, _evidence_excerpt,
                        _identifier_lines, _labels)


def render_review_report(report):
    sources, source_numbers, claim_sources = _citation_state(report)
    snapshots = {item["claim"]["id"]: item for item in report["claim_snapshots"]}
    claim_numbers = {key: index for index, key in enumerate(snapshots, 1)}
    evidence_by_id = {}
    for snapshot in snapshots.values():
        for evidence in snapshot["evidence"]:
            evidence_by_id.setdefault(evidence["id"], evidence)
    evidence_numbers = {key: index for index, key in enumerate(evidence_by_id, 1)}

    def status(snapshot):
        assessment = snapshot.get("assessment")
        return assessment["status"] if assessment and snapshot["assessment_current"] else "unassessed"

    def references(identifiers):
        parts = []
        for identifier in identifiers:
            number = claim_numbers[identifier]
            snapshot = snapshots[identifier]
            parts.append(f"[C{number}](#c{number}) ({snapshot['claim']['kind']}; {status(snapshot)})")
        labels = _labels(identifiers, source_numbers, claim_sources)
        return " ".join(parts + ([labels] if labels else []))

    lines = ["# " + authored(report["title"]), "",
             "> 研究设计报告。结构验收不代表科学有效性；观察、提案及评估范围见论断附录。", ""]
    for section in report["sections"]:
        lines.extend(["## " + authored(section["heading"]), ""])
        paragraphs = section.get("paragraphs", [])
        if paragraphs:
            for paragraph in paragraphs:
                refs = references(paragraph["claim_ids"])
                lines.extend([authored(paragraph["text"]) + (" " + refs if refs else ""), ""])
            cited = {key for paragraph in paragraphs for key in paragraph["claim_ids"]}
            remaining = [key for key in section["claim_ids"] if key not in cited]
            if remaining:
                lines.extend(["本节补充论断：" + references(remaining), ""])
        else:
            for key in section["claim_ids"]:
                lines.extend([authored(snapshots[key]["claim"]["text"]) + " " + references([key]), ""])

    lines.extend(["## 局限汇总", ""])
    lines.extend("- " + authored(value) for value in report["limitations"])
    if not report["limitations"]:
        lines.append("- 未提供。")
    lines.extend(["", "## 论断与证据审计", "",
                  "以下编号是本报告内的映射；完整不可变记录与标识保留在 host 账本和 trace。", ""])
    for key, snapshot in snapshots.items():
        claim = snapshot["claim"]
        lines.extend([f"### C{claim_numbers[key]}", "",
                      f"Claim kind: {claim['kind']}; Assessment status: {status(snapshot)}", "",
                      authored(claim["text"]), ""])
        assessment = snapshot.get("assessment")
        if assessment:
            scope = "provisional semantic review" if assessment["provisional"] else "narrow literal-material verification only"
            lines.extend(["Assessment scope: " + scope, ""])
        if claim["kind"] != "observation":
            lines.extend(["此项为未确立的提案或被否定的论断，不是已证实结论。", ""])
        labels = [f"[E{evidence_numbers[e['id']]}](#e{evidence_numbers[e['id']]})" for e in snapshot["evidence"]]
        lines.extend(["证据：" + (" ".join(labels) if labels else "未绑定证据；不得当作有来源的事实。"), ""])
    lines.extend(["## 证据摘录", ""])
    for key, evidence in evidence_by_id.items():
        label, excerpt = _evidence_excerpt(evidence, source_numbers)
        lines.extend([f"### E{evidence_numbers[key]}", "", label, "", "> " + authored(excerpt), ""])
    lines.extend(["## Sources", ""])
    for number, source in enumerate(sources, 1):
        lines.extend(_identifier_lines(source, number, 2))
        lines.append("")
    if not sources:
        lines.append("No source-backed citations were bound to this report.")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")
