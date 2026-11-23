"""Explicit offline scripted decisions, NEVER a model capability benchmark.

The script consumes actual tool results through the same model/tool protocol.
Native protein/DEG calculations are real. Ferritin uses a labeled synthetic
user-file fixture and cannot validate the actual formulation/design question.
"""

import json

from .contracts import ModelReply, ToolCall
from .models import ProviderFailure


class OfflineScenario:
    secrets = ()
    metadata = {"provider_kind": "offline-scripted", "model": "engineering-scenario-v1"}

    def __init__(self, case, fixture_path):
        self.case, self.fixture_path, self.phase, self.counter = case, str(fixture_path), 0, 0
        self.requests, self.observations, self.proposals = [], [], []

    def _calls(self, calls):
        result = []
        for name, arguments in calls:
            self.counter += 1
            result.append(ToolCall(f"fixture_{self.counter}", name, arguments))
        return ModelReply(content="[OFFLINE SCRIPTED ENGINEERING CONTROL — NOT AN LLM DECISION]",
            tool_calls=result, finish_reason="tool_calls", usage={"input_tokens": 0, "output_tokens": 0})

    def complete(self, messages, tools, **limits):
        self.requests.append({"messages": messages, "tools": tools, **limits})
        last = []
        for message in reversed(messages):
            if message["role"] != "tool": break
            result = json.loads(message["content"])
            if result.get("status") != "ok":
                raise ProviderFailure("offline_fixture_tool_failed", "Actual tool failed; scripted fixture does not fabricate replacement evidence", cost_unknown=False)
            last.insert(0, result)
        phase, self.phase = self.phase, self.phase + 1
        kind = self.case["id"]
        if phase == 0:
            if kind == "protein_extension": return self._calls([("protein_sequence", {"sequence": "MKWVTFISLLFLFSSAYS"})])
            if kind == "historical_deg": return self._calls([("deg_analyze", {"top_n": 250, "n_permutations": 500})])
            return self._calls([("file_inspect", {"path": self.fixture_path})])
        if phase == 1:
            # Does not need hidden access to IDs in a truncated native result.
            return self._calls([("research_state", {"limit": 10})])
        if phase == 2:
            records = last[-1]["data"]["records"]
            source = next(item for item in records if item["entity"] == "source")
            if kind == "protein_extension": pointers = ["/length", "/composition", "/molecular_weight_da", "/theoretical_isoelectric_point", "/gravy", "/aromaticity"]
            elif kind == "historical_deg": pointers = [f"/tables/panel_centered_reversal/{i}" for i in range(5)] + ["/statistics", "/controls"]
            else: pointers = ["/scope"]
            return self._calls([("evidence_select", {"source_id": source["id"], "selector": {"pointer": pointer}}) for pointer in pointers])
        if phase == 3:
            return self._calls([("claim_observe", {"evidence_id": result["data"]["id"]}) for result in last])
        if phase == 4:
            self.observations = [result["data"]["id"] for result in last]
            headings = self.case["acceptance"].get("required_sections", ["范围与限制"])
            if kind == "historical_ferritin":
                texts = [f"工程回归占位：{heading}。本离线场景只检查章节与证据绑定机制；未检索真实文献，未提出或验证铁蛋白配方。不能作为该历史设计问题的科学答案。" for heading in headings]
            elif kind == "historical_deg":
                texts = ["这里仅报告真实数据计算得到的探索性候选顺序；不能由n=1、共享DMSO分母的转录关联确认作用机制、体内疫苗增效或安全性。机制仍需独立文献和可证伪实验，当前离线控制不是模型推理质量评估。"]
            else:
                texts = ["这些是实际计算的未修饰线性序列理论性质，不是实验测量；不能据此推断真实三维结构、生物功能、亲和力、安全性或效力。离线脚本不测试语言模型自主推理。"]
            return self._calls([("claim_record", {"text": text, "kind": "proposal", "evidence_ids": []}) for text in texts])
        if phase == 5:
            self.proposals = [result["data"]["id"] for result in last]
            if kind == "historical_ferritin":
                sections = [{"heading": heading, "claim_ids": [self.proposals[i]] + (self.observations if i == 0 else [])}
                    for i, heading in enumerate(self.case["acceptance"]["required_sections"])]
            elif kind == "historical_deg":
                sections = [{"heading": "候选排序（探索性）", "claim_ids": self.observations[:5]},
                    {"heading": "统计与对照", "claim_ids": self.observations[5:]},
                    {"heading": "机制解释边界", "claim_ids": self.proposals}]
            else:
                sections = [{"heading": "计算结果", "claim_ids": self.observations},
                    {"heading": "适用范围与限制", "claim_ids": self.proposals}]
            return self._calls([("report_create", {"title": "OFFLINE ENGINEERING REGRESSION · " + kind,
                "sections": sections, "limitations": [
                    "模型决策为明确标记的离线脚本，不是实时LLM评测；trace记录真实执行的工具与工件。",
                    "报告验收只验证结构、当前证据绑定和工件一致性，不代表科学答案、机制、安全性或体内效力获验证。",
                    "DEG/蛋白质计算如存在则为真实本地运算；铁蛋白场景仅使用显式合成用户文件，不是文献检索或科学设计答案。"]})])
        if phase == 6: return self._calls([("report_validate", {"report_id": last[-1]["data"]["id"]})])
        raise ProviderFailure("offline_fixture_not_accepted", "The scripted report did not pass the real acceptance gate", cost_unknown=False)

    def close(self): pass
