"""Register reused legacy analysis/exec assets as first-class tools.

- code_runner.execute_python_code -> code-as-action for the data specialist
- adjuvant_scoring (deterministic stats library) -> imported by generated code
"""
from __future__ import annotations

from typing import Any

from core.config import env
from environment.registry import REGISTRY, ToolSpec
from environment.tools.exec.code_runner import execute_python_code, validate_python_code


def run_python(code: str, timeout: int = 60) -> dict[str, Any]:
    """Execute Python in a subprocess (code-as-action).

    SECURITY: this is NOT a sandbox — it runs arbitrary code with full FS/network
    access. Disabled unless ADJ_ALLOW_CODE_EXEC=1 is set in the environment.
    """
    if env("ADJ_ALLOW_CODE_EXEC", "0") != "1":
        return {"validation": "skipped",
                "output": "[run_python disabled] set ADJ_ALLOW_CODE_EXEC=1 to enable "
                          "code execution (NOT a sandbox; only enable for trusted runs)."}
    validation = validate_python_code(code)
    output = execute_python_code(code, timeout=timeout)
    return {"validation": validation, "output": output}


REGISTRY.register(
    ToolSpec(name="run_python",
             description="Execute Python code in a subprocess (code-as-action) for data "
                         "analysis; can import environment.tools.analysis.adjuvant_scoring "
                         "for deterministic stats. Returns stdout/stderr.",
             domain="exec",
             parameters={"code": "python source", "timeout": "seconds (int)"},
             returns="dict with validation and output"),
    run_python,
)
