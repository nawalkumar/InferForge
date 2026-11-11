"""The data foundation must stay free of hardware access (architecture §2, Stage 2 boundary)."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import gpu_system_inspector

PACKAGE_ROOT = Path(gpu_system_inspector.__file__).parent
ALLOWED_STDLIB = {
    "__future__",
    "collections",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "importlib",
    "json",
    "math",
    "re",
    "sys",
    "types",
    "typing",
}
FORBIDDEN_INTERNAL = ("gpu_system_inspector.cli", "gpu_system_inspector.adapters")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    "path",
    sorted((PACKAGE_ROOT / "models").glob("*.py"))
    + sorted((PACKAGE_ROOT / "reporting").glob("*.py")),
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_data_foundation_imports_only_pure_modules(path):
    for name in _imports(path):
        root = name.split(".")[0]
        if root == "gpu_system_inspector":
            assert not name.startswith(FORBIDDEN_INTERNAL), f"{path.name} imports {name}"
        else:
            assert root in ALLOWED_STDLIB, f"{path.name} imports {name}"


# Adapter implementations and the OS modules each may use. Everything else in adapters/ is
# a contract module and stays pure.
ADAPTER_IMPLEMENTATIONS = {
    "platform_os.py": {"os", "platform", "sys", "struct", "ctypes", "winreg"},
    "pci_sysfs.py": {"os"},
    # IMP-A05: the only module that may start processes (imported on first use).
    "commands.py": {"os", "subprocess", "shutil"},
    "nvidia_smi.py": {"os", "csv"},
    # IMP-A06: files through the `ToolkitFiles` seam; `nvcc` only through the runner.
    "toolkit.py": {"os", "stat"},
    # IMP-A07: torch only through importlib; the child runs through the shared runner.
    "torch_probe_payload.py": {"warnings"},
    # IMP-A01: `pynvml` is loaded through `importlib` by `open()`, never imported statically.
    "nvml.py": {"logging"},
}
NEVER_IMPORTED = {"pynvml", "torch", "subprocess", "socket", "shutil", "urllib", "http"}
INTERNAL = ("gpu_system_inspector.models", "gpu_system_inspector.adapters")


@pytest.mark.parametrize(
    "path", sorted((PACKAGE_ROOT / "adapters").glob("*.py")), ids=lambda p: f"adapters/{p.name}"
)
def test_adapters_import_only_models_stdlib_and_their_declared_os_modules(path):
    """Contracts: pure (architecture §2). Implementations: only their declared OS modules,
    never NVML, PyTorch, processes or the network."""
    declared = ADAPTER_IMPLEMENTATIONS.get(path.name, set())
    allowed = ALLOWED_STDLIB | {"time"} | declared
    for name in _imports(path):
        root = name.split(".")[0]
        if root == "gpu_system_inspector":
            assert name.startswith(INTERNAL), f"{path.name} imports {name}"
        else:
            forbidden = NEVER_IMPORTED - declared
            assert root in allowed and root not in forbidden, f"{path.name} imports {name}"


def test_only_the_command_runner_may_start_processes():
    for path in sorted((PACKAGE_ROOT / "adapters").glob("*.py")):
        roots = {name.split(".")[0] for name in _imports(path)}
        if path.name != "commands.py":
            assert not roots & {"subprocess", "shutil"}, path.name
        assert "pynvml" not in roots, path.name  # loaded lazily, never imported statically


@pytest.mark.parametrize(
    "path",
    sorted((PACKAGE_ROOT / "collectors").glob("*.py")),
    ids=lambda p: f"collectors/{p.name}",
)
def test_collectors_reach_the_host_only_through_adapters(path):
    for name in _imports(path):
        root = name.split(".")[0]
        if root == "gpu_system_inspector":
            assert name.startswith((*INTERNAL, "gpu_system_inspector.collectors")), (
                f"{path.name} imports {name}"
            )
        else:
            assert root in ALLOWED_STDLIB, f"{path.name} imports {name}"


CORE_INTERNAL = (*INTERNAL, "gpu_system_inspector.collectors", "gpu_system_inspector.reporting")


@pytest.mark.parametrize(
    "path",
    [*sorted((PACKAGE_ROOT / "core").glob("*.py")), PACKAGE_ROOT / "api.py"],
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_the_core_and_api_use_only_lower_layers(path):
    """`cli → core → {collectors, reporting} → adapters → models` (architecture §2)."""
    for name in _imports(path):
        root = name.split(".")[0]
        if root == "gpu_system_inspector":
            assert name.startswith((*CORE_INTERNAL, "gpu_system_inspector.core")), (
                f"{path.name} imports {name}"
            )
        else:
            assert root in ALLOWED_STDLIB, f"{path.name} imports {name}"


def test_importing_the_foundation_adapters_and_collectors_loads_no_hardware_modules():
    code = (
        "import sys\n"
        "import gpu_system_inspector.models, gpu_system_inspector.reporting\n"
        "import gpu_system_inspector.adapters, gpu_system_inspector.collectors\n"
        "import gpu_system_inspector.core, gpu_system_inspector.api\n"
        "print(sorted(m for m in ('pynvml', 'torch', 'subprocess', 'ctypes', 'socket')"
        " if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
