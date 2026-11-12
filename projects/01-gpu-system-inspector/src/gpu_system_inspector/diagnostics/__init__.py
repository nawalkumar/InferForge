"""Evidence-based diagnostics over an inspection snapshot (architecture §4.8)."""

from gpu_system_inspector.diagnostics.engine import (
    UNEVALUATED_TITLE,
    DiagnosticInput,
    DiagnosticsEngine,
    DiagnosticsResult,
    RegisteredRule,
)
from gpu_system_inspector.diagnostics.rules import DEFAULT_RULES

__all__ = [
    "DEFAULT_RULES",
    "UNEVALUATED_TITLE",
    "DiagnosticInput",
    "DiagnosticsEngine",
    "DiagnosticsResult",
    "RegisteredRule",
]
