import subprocess
import sys
from importlib import metadata
from pathlib import Path

import gpu_system_inspector

DISTRIBUTION = "01-gpu-system-inspector"


def test_public_names_are_importable():
    for name in gpu_system_inspector.__all__:
        assert hasattr(gpu_system_inspector, name)


def test_version_matches_installed_distribution_metadata():
    assert gpu_system_inspector.__version__ == metadata.version(DISTRIBUTION)


def test_names_are_consistent_with_the_specification():
    assert gpu_system_inspector.DISPLAY_NAME == "GPU System Inspector"
    assert gpu_system_inspector.APPLICATION_NAME == "gpu-system-inspector"
    assert gpu_system_inspector.DISTRIBUTION_NAME == DISTRIBUTION
    assert gpu_system_inspector.REPORT_SCHEMA_NAME == "inferforge.gpu_system_inspector.report"
    assert gpu_system_inspector.REPORT_SCHEMA_VERSION == "1.0"


def test_package_ships_type_information():
    assert (Path(gpu_system_inspector.__file__).parent / "py.typed").is_file()


def test_importing_the_package_loads_no_gpu_libraries():
    code = (
        "import sys, gpu_system_inspector, gpu_system_inspector.cli.main\n"
        "print(sorted(m for m in ('pynvml', 'torch') if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
