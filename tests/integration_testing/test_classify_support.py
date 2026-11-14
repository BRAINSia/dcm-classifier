import json
import shutil
from pathlib import Path

import pydicom
import pytest
from pydicom.uid import SecondaryCaptureImageStorage, generate_uid

from dcm_classifier.classify_support import (
    _classification_failure_reason,
    simple_classify_study,
)

current_file_path = Path(__file__).resolve()
testing_output_dir = current_file_path.parent.parent / "testing_data" / "outputs"


def test_classify_study_json(get_data_dir):
    testing_session_directory = get_data_dir.as_posix()

    if not testing_output_dir.exists():
        testing_output_dir.mkdir(parents=True)

    if not (testing_output_dir / "json").exists():
        (testing_output_dir / "json").mkdir(parents=True)

    simple_classify_study(
        session_directory=testing_session_directory,
        json_dumppath=(testing_output_dir / "json" / "output.json"),
    )

    # Check that the JSON file now exists and contains proper JSON.
    json_file = testing_output_dir / "json" / "output.json"
    assert json_file.exists(), "JSON dump file should have been created."

    with open(json_file) as f:
        json_content = json.load(f)

    # get baseline json file content
    with open(
        testing_output_dir.parent
        / "integration_testing/classify_study_data"
        / "output.json",
    ) as f:
        baseline_json_content = json.load(f)

    # Compare the JSON content with the baseline, ignoring the first line and comparing only values
    assert list(json_content.values()) == list(baseline_json_content.values()), (
        "JSON content values should match the baseline except for the first line."
    )

    # cleanup
    shutil.rmtree(testing_output_dir)


def test_classify_study_with_file_creation(get_data_dir):
    testing_session_directory = get_data_dir.as_posix()

    if not testing_output_dir.exists():
        testing_output_dir.mkdir(parents=True)

    if not (testing_output_dir / "json").exists():
        (testing_output_dir / "json").mkdir(parents=True)

    simple_classify_study(
        session_directory=testing_session_directory,
        json_dumppath=(testing_output_dir / "json" / "output.json"),
        output_dir=(testing_output_dir / "organized_dicom"),
        nifti_dir=(testing_output_dir / "nifti"),
    )

    # Check that the JSON file now exists and contains proper JSON. we will omit checking the content in this test
    json_file = testing_output_dir / "json" / "output.json"
    assert json_file.exists(), "JSON dump file should have been created."

    # Check that the output directory contains a subdirectory for the volume.
    sub_output_dir = testing_output_dir / "organized_dicom"

    # check the subdirectories in organized_dicom are created
    example_file_names = [
        "0001_000_gret2star",
        "0004_000_gret2star",
        "0005_000_tracew_b0",
        "0005_001_tracew_b1000",
        "0006_000_adc_b1000",
        "0007_000_flair",
        "0010_000_t1w",
        "0011_000_t2w",
    ]

    for file_name in example_file_names:
        assert (sub_output_dir / file_name).exists(), (
            f"Output directory should have a subdirectory for {file_name}."
        )

    # now we do the same but for the nifti directory
    nifti_dir = testing_output_dir / "nifti"
    assert nifti_dir.exists(), "nifti_dir should exist."

    # Expected file name format: "0001_001_MR.nii.gz" (series "1" becomes "0001", volume index "1" becomes "001")
    nifti_file_names = [
        "0001_000_gret2star.nii.gz",
        "0004_000_gret2star.nii.gz",
        "0005_000_tracew_b0.nii.gz",
        "0005_001_tracew_b1000.nii.gz",
        "0006_000_adc_b1000.nii.gz",
        "0007_000_flair.nii.gz",
        "0010_000_t1w.nii.gz",
        "0011_000_t2w.nii.gz",
    ]

    for file_name in nifti_file_names:
        assert (nifti_dir / file_name).exists(), (
            f"Expected nifti file {file_name} should have been created."
        )

    # cleanup
    shutil.rmtree(testing_output_dir)


def test_classify_study_with_no_readable_dicoms(get_data_dir, capsys):
    testing_session_directory = get_data_dir.parent / "empty_directory"

    if not testing_output_dir.exists():
        testing_output_dir.mkdir(parents=True)

    if not (testing_output_dir / "json").exists():
        (testing_output_dir / "json").mkdir(parents=True)

    simple_classify_study(
        session_directory=testing_session_directory,
    )

    # check that 'No series found in the study.' message is printed
    captured = capsys.readouterr()
    assert "No series found in the study." in captured.out, (
        "Should print 'No series found in the study.' message."
    )


def test_classify_study_json_keeps_invalid_volume(contrast_file_path, tmp_path):
    ds = pydicom.dcmread(next(contrast_file_path.rglob("*.dcm")))
    del ds.FlipAngle
    ds.save_as(tmp_path / "missing_flip_angle.dcm")

    session = simple_classify_study(session_directory=tmp_path)
    (volume,) = session[str(tmp_path)].values()

    assert volume["VolumeModality"] == "INVALID"
    assert (
        volume["ClassificationFailureReason"]
        == "Required DICOM fields are missing or invalid"
    )
    assert volume["StudyInstanceUID"] == str(ds.StudyInstanceUID)
    assert not any(key.startswith("GUESS_ONNX") for key in volume)


@pytest.mark.parametrize("mr_sop_class_only", [False, True])
def test_classify_study_mr_sop_class_only(
    contrast_file_path, tmp_path, mr_sop_class_only
):
    mr_file = next(contrast_file_path.rglob("*.dcm"))
    shutil.copy(mr_file, tmp_path / "mr.dcm")
    ds = pydicom.dcmread(mr_file)
    mr_series_number = int(ds.SeriesNumber)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 99
    ds.save_as(tmp_path / "secondary_capture.dcm")

    session = simple_classify_study(
        session_directory=tmp_path, mr_sop_class_only=mr_sop_class_only
    )
    series_numbers = {int(key.split("_")[0]) for key in session[str(tmp_path)]}

    expected = {mr_series_number} if mr_sop_class_only else {mr_series_number, 99}
    assert series_numbers == expected


@pytest.mark.parametrize(
    ("volume_dictionary", "reason"),
    [
        ({}, "Required DICOM fields are missing or invalid"),
        ({"EchoTime": 3.69}, "Inference features are missing or invalid"),
    ],
)
def test_classification_failure_reason(volume_dictionary, reason):
    class StubVolume:
        def get_volume_dictionary(self):
            return volume_dictionary

    assert _classification_failure_reason(StubVolume()) == reason
