import importlib.util
import json
from pathlib import Path

import pytest

from dcm_classifier.dicom_config import inference_features
from dcm_classifier.image_type_inference import imagetype_to_integer_mapping

script_path: Path = Path(__file__).parents[2] / "scripts" / "data_further_processing.py"
spec = importlib.util.spec_from_file_location("data_further_processing", script_path)
data_further_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_further_processing)


def test_writes_json_and_nifti_per_sub_volume(get_data_dir, tmp_path):
    studies_directory = tmp_path / "studies"
    studies_directory.mkdir()
    (studies_directory / "tracew_study").symlink_to(get_data_dir / "5")
    output_directory = tmp_path / "output"

    data_further_processing.main(
        ["-d", str(studies_directory), "-o", str(output_directory)]
    )

    (json_file,) = output_directory.rglob("*.json")
    series_uid = json_file.stem.removeprefix("tracew_")
    assert json_file.name == f"tracew_{series_uid}.json"
    assert json_file.parent.name == f"5_{series_uid}"
    assert sorted(f.name for f in json_file.parent.glob("*.nii.gz")) == [
        f"tracew_{series_uid}_000.nii.gz",
        f"tracew_{series_uid}_001.nii.gz",
    ]

    sub_volumes = json.loads(json_file.read_text())
    assert sorted(sub_volumes) == ["0", "1"]
    for sub_volume in sub_volumes.values():
        assert set(sub_volume["TrainingData"]) == set(inference_features)
        assert set(sub_volume["OutputProbabilities"]) == set(
            imagetype_to_integer_mapping
        )
        assert sum(sub_volume["OutputProbabilities"].values()) == pytest.approx(1.0)
        best_class = max(
            sub_volume["OutputProbabilities"], key=sub_volume["OutputProbabilities"].get
        )
        assert sub_volume["ImageType"].removeprefix("LOW_PROBABILITY_") == best_class


def test_failing_study_does_not_stop_the_batch(
    get_data_dir, tmp_path, capsys, monkeypatch
):
    studies_directory = tmp_path / "studies"
    (studies_directory / "a_failing_study").mkdir(parents=True)
    (studies_directory / "b_tracew_study").symlink_to(get_data_dir / "5")
    output_directory = tmp_path / "output"
    prepare_study = data_further_processing.prepare_study

    def fail_on_first_study(study_directory, *args):
        if study_directory.name == "a_failing_study":
            raise RuntimeError("unreadable study")
        return prepare_study(study_directory, *args)

    monkeypatch.setattr(data_further_processing, "prepare_study", fail_on_first_study)

    with pytest.raises(SystemExit, match="1 studies, series, or sub-volumes"):
        data_further_processing.main(
            ["-d", str(studies_directory), "-o", str(output_directory)]
        )

    assert (
        "a_failing_study could not be prepared: unreadable study"
        in capsys.readouterr().out
    )
    assert len(list(output_directory.rglob("*.json"))) == 1


def test_second_copy_of_a_series_does_not_overwrite_the_first(
    get_data_dir, tmp_path, capsys
):
    studies_directory = tmp_path / "studies"
    studies_directory.mkdir()
    (studies_directory / "a_copy").symlink_to(get_data_dir / "5")
    (studies_directory / "b_copy").symlink_to(get_data_dir / "5")
    output_directory = tmp_path / "output"

    with pytest.raises(SystemExit, match="1 studies, series, or sub-volumes"):
        data_further_processing.main(
            ["-d", str(studies_directory), "-o", str(output_directory)]
        )

    assert "Series already prepared, skipped" in capsys.readouterr().out
    assert len(list(output_directory.rglob("*.json"))) == 1
