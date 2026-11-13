"""Explicit host workflow presets, projected through existing session guidance."""

import hashlib
import json
from pathlib import Path

from .profiles import load_trusted_guidance
from .research.acceptance import requirements
from .tools.catalog import canonical_json


WORKFLOW_NAMES = ("adjuvant-design-v1", "adjuvant-design-critic-v1")


def load_workflow(name):
    if not isinstance(name, str) or name not in WORKFLOW_NAMES:
        raise ValueError("Unknown host workflow")
    root = Path(__file__).resolve().parents[1] / "config/skills/adjuvant-design"
    instructions = load_trusted_guidance(root / "SKILL.md")
    acceptance = requirements(json.loads(load_trusted_guidance(root / "acceptance.json")))
    if name == "adjuvant-design-critic-v1":
        instructions += "\n\n" + load_trusted_guidance(root / "critic-workflow.md")
        acceptance["require_critic"] = True
        acceptance["require_concise_discovery"] = True
    return {"name": name, "version": 3 if name == "adjuvant-design-critic-v1" else 2, "instructions": instructions,
            "instructions_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
            "acceptance": acceptance,
            "acceptance_sha256": hashlib.sha256(canonical_json(acceptance).encode()).hexdigest()}
