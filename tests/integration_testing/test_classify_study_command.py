import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest

from dcm_classifier.classify_study import main

baseline_json_path: Path = (
    Path(__file__).parents[1]
    / "testing_data"
    / "integration_testing"
    / "classify_study_data"
    / "output.json"
)


def test_main_writes_json_catalog(get_data_dir, tmp_path):
    json_path = tmp_path / "output.json"

    main(["--session_directory", get_data_dir.as_posix(), "--json", str(json_path)])

    json_content = json.loads(json_path.read_text())
    baseline_json_content = json.loads(baseline_json_path.read_text())
    assert list(json_content.values()) == list(baseline_json_content.values())


def test_main_requires_session_directory(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2
    assert "--session_directory" in capsys.readouterr().err


def test_command_is_installed_with_the_package():
    try:
        installed = distribution("dcm_classifier")
    except PackageNotFoundError:
        pytest.skip("dcm_classifier is imported from source, not installed")
    commands = {
        entry_point.name: entry_point.value
        for entry_point in installed.entry_points
        if entry_point.group == "console_scripts"
    }
    assert commands.get("dcm-classify-study") == "dcm_classifier.classify_study:main"
