from importlib import metadata

from gpu_system_inspector.cli.main import main

DISTRIBUTION = "01-gpu-system-inspector"


def _requirements() -> list[str]:
    return metadata.requires(DISTRIBUTION) or []


def test_requires_python_is_the_workspace_floor():
    assert metadata.metadata(DISTRIBUTION)["Requires-Python"] == ">=3.11"


def test_only_runtime_dependency_is_nvidia_ml_py():
    assert _requirements() == ["nvidia-ml-py>=12.535.108"]


def test_pytorch_is_neither_a_dependency_nor_an_extra():
    assert not any(req.lower().startswith("torch") for req in _requirements())
    assert (metadata.metadata(DISTRIBUTION).get_all("Provides-Extra") or []) == []


def test_console_script_points_at_the_cli_entry_point():
    (entry_point,) = metadata.entry_points(group="console_scripts", name="inferforge-gpu")
    assert entry_point.value == "gpu_system_inspector.cli.main:main"
    assert entry_point.load() is main


def test_distribution_is_marked_as_not_for_upload():
    classifiers = metadata.metadata(DISTRIBUTION).get_all("Classifier") or []
    assert "Private :: Do Not Upload" in classifiers
