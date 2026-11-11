"""The PyTorch probe child: `python -m gpu_system_inspector.adapters._torch_probe_main`.

Run by `SubprocessTorchProbe` (architecture §4.7, data flow §5, DEC-48). It imports torch
and prints one JSON document on stdout: the probe function's (`torch_probe_payload`), or
`import_ok: false` when the import fails. Anything unexpected propagates, so the child exits
non-zero and the parent reports `PROBE_EXIT_<n>`.
"""

from __future__ import annotations

import importlib
import json
import sys

from gpu_system_inspector.adapters.torch_probe_payload import import_failure, probe_payload


def main() -> int:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # installed but not importable (FX15)
        payload = import_failure(exc)
    else:
        payload = probe_payload(torch)
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
