"""Print the report JSON Schema generated from the models (used to regenerate the shipped file)."""

import sys

from gpu_system_inspector.reporting.schema import render_report_schema

if __name__ == "__main__":
    sys.stdout.write(render_report_schema())
