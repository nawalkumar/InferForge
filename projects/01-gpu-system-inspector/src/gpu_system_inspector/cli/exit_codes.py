"""Process exit codes (docs/cli_specification.md §6)."""

from enum import IntEnum


class ExitCode(IntEnum):
    SUCCESS = 0
    READINESS_NOT_MET = 1
    USAGE_ERROR = 2
    READINESS_UNKNOWN = 3
    OUTPUT_ERROR = 4
    INTERNAL_ERROR = 5
    INTERRUPTED = 130
