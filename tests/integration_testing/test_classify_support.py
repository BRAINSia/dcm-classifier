import json
import shutil
from pathlib import Path


from dcm_classifier.classify_support import simple_classify_study

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
    assert list(json_content.values()) == list(
        baseline_json_content.values()
    ), "JSON content values should match the baseline except for the first line."

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
        assert (
            sub_output_dir / file_name
        ).exists(), f"Output directory should have a subdirectory for {file_name}."

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
        assert (
            nifti_dir / file_name
        ).exists(), f"Expected nifti file {file_name} should have been created."

    # cleanup
    shutil.rmtree(testing_output_dir)
