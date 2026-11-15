import logging
import random
import re
import shutil
from pathlib import Path

import pydicom
import pytest
from pydicom.uid import SecondaryCaptureImageStorage, generate_uid

from dcm_classifier.dicom_config import MISSING_VALUE, inference_features
from dcm_classifier.image_type_inference import (
    ImageTypeClassifierBase,
    imagetype_to_integer_mapping,
)
from dcm_classifier.study_processing import (
    ProcessOneDicomStudyToVolumesMappingBase,
)

relative_testing_data_path: Path = Path(__file__).parent.parent / "testing_data"


def test_study_with_search_series():
    study_path = relative_testing_data_path / "anonymized_data"
    study_to_volume_mapping_base = ProcessOneDicomStudyToVolumesMappingBase(
        study_path, {"test": 7}
    )
    assert study_to_volume_mapping_base.study_directory == study_path
    assert study_to_volume_mapping_base.search_series == {"test": 7}


def test_get_list_of_primary_volume_info(get_data_dir):
    # study_path = relative_testing_data_path.parent.parent.parent / "dcm_files"

    study_to_volume_mapping_base = ProcessOneDicomStudyToVolumesMappingBase(
        get_data_dir
    )
    volume_info_dictionaries = (
        study_to_volume_mapping_base.get_list_of_primary_volume_info()
    )
    print(len(volume_info_dictionaries))

    assert len(volume_info_dictionaries) == 16


def test_get_study_dictionary_and_set_inferer():
    study_path = (
        relative_testing_data_path / "anonymized_testing_data" / "anonymized_data"
    )

    default_classification_model_filename: Path = (
        Path(__file__).parents[2] / "models" / "rf_classifier.onnx"
    )

    study_to_volume_mapping_base = ProcessOneDicomStudyToVolumesMappingBase(study_path)
    study_dictionary = study_to_volume_mapping_base.get_study_dictionary()
    assert isinstance(study_dictionary, dict)
    assert len(study_dictionary) == 15
    study_to_volume_mapping_base.set_inferer(
        ImageTypeClassifierBase(
            classification_model_filename=default_classification_model_filename,
            classification_feature_list=inference_features,
            image_type_map=imagetype_to_integer_mapping,
            min_probability_threshold=0.4,
        )
    )
    assert (
        study_to_volume_mapping_base.inferer.classification_model_filename
        == default_classification_model_filename
    )
    assert (
        study_to_volume_mapping_base.inferer.classification_feature_list
        == inference_features
    )
    assert (
        study_to_volume_mapping_base.inferer.imagetype_to_int_map
        == imagetype_to_integer_mapping
    )
    assert study_to_volume_mapping_base.inferer.min_probability_threshold == 0.4


def test_un_pathlike_dir():
    with pytest.raises(TypeError, match="study_directory must be path-like; got 42"):
        ProcessOneDicomStudyToVolumesMappingBase(42)


# validate() is not implemented yet and only returns None
def test_validate(mock_volume_study):
    assert mock_volume_study.validate() is None


def test_split_orders_files_by_instance_number(get_data_dir, tmp_path):
    for dcm_file in (get_data_dir / "9").rglob("*.dcm"):
        instance_number = int(pydicom.dcmread(dcm_file).InstanceNumber)
        shutil.copy(dcm_file, tmp_path / f"slice.{instance_number}.dcm")
    files = [str(f) for f in tmp_path.glob("*.dcm")]
    random.Random(0).shuffle(files)

    study = ProcessOneDicomStudyToVolumesMappingBase(tmp_path)
    (volume,) = study._identify_and_split_sub_volumes_pydicom(files)

    instance_numbers = [
        int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber)
        for f in volume.one_volume_dcm_filenames
    ]
    assert instance_numbers == sorted(instance_numbers)


def test_mosaic_series_is_flagged_with_a_warning(get_data_dir, caplog):
    dwi_directory = get_data_dir.parent / "anonymized_dwi_series"

    study = ProcessOneDicomStudyToVolumesMappingBase(dwi_directory)

    (mosaic_warning,) = (
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "mosaic" in r.getMessage()
    )
    assert f"SeriesNumber 2 in {dwi_directory}" in mosaic_warning.getMessage()

    series = study.series_dictionary[2]
    assert series.get_is_mosaic()
    assert all(volume.get_is_mosaic() for volume in series.get_volume_list())


def test_non_mosaic_series_is_not_flagged(get_data_dir, caplog):
    study = ProcessOneDicomStudyToVolumesMappingBase(get_data_dir / "5")

    (series,) = study.series_dictionary.values()
    assert not series.get_is_mosaic()
    assert "mosaic" not in caplog.text


invalid_fields_directory: Path = (
    relative_testing_data_path
    / "anonymized_testing_data"
    / "invalid_data"
    / "invalid_fields"
)


@pytest.mark.parametrize("file_name", ["noSeriesNum.dcm", "invalidSeriesNum.dcm"])
def test_unusable_series_number_does_not_stop_the_study(
    get_data_dir, tmp_path, file_name, caplog
):
    shutil.copy(invalid_fields_directory / file_name, tmp_path / file_name)
    valid_file = next((get_data_dir / "9").rglob("*.dcm"))
    shutil.copy(valid_file, tmp_path / "valid.dcm")

    study = ProcessOneDicomStudyToVolumesMappingBase(
        tmp_path, inferer=ImageTypeClassifierBase()
    )
    assert "Required DICOM fields" in caplog.text
    assert re.search(f"SeriesNumber is missing.*{file_name}", caplog.text)
    study.run_inference()
    assert f"Missing features for SeriesNumber {MISSING_VALUE}" in caplog.text

    assert set(study.series_dictionary) == {9, MISSING_VALUE}
    (volume,) = study.series_dictionary[MISSING_VALUE].get_volume_list()
    assert volume.get_volume_modality() == "INVALID"
    assert study.series_dictionary[9].get_series_modality() != "INVALID"


@pytest.mark.parametrize("file_name", ["noSeriesNum.dcm", "invalidSeriesNum.dcm"])
def test_unusable_series_number_raises_when_errors_are_requested(tmp_path, file_name):
    shutil.copy(invalid_fields_directory / file_name, tmp_path / file_name)

    with pytest.raises(ValueError, match=f"SeriesNumber is missing.*{file_name}"):
        ProcessOneDicomStudyToVolumesMappingBase(tmp_path, raise_error_on_failure=True)


def _copy_unnumbered(tmp_path, file_name, series_uid):
    ds = pydicom.dcmread(invalid_fields_directory / file_name)
    ds.SeriesInstanceUID = series_uid
    ds.save_as(tmp_path / file_name)


def test_unnumbered_series_are_kept_apart(tmp_path, caplog):
    _copy_unnumbered(tmp_path, "noSeriesNum.dcm", generate_uid())
    _copy_unnumbered(tmp_path, "invalidSeriesNum.dcm", generate_uid())

    study = ProcessOneDicomStudyToVolumesMappingBase(tmp_path)

    assert set(study.series_dictionary) == {MISSING_VALUE, MISSING_VALUE - 1}
    assert f"filed under series {MISSING_VALUE}\n" in caplog.text
    assert f"filed under series {MISSING_VALUE - 1}\n" in caplog.text
    for series in study.series_dictionary.values():
        assert len(series.get_volume_list()) == 1


def test_unnumbered_volumes_of_one_series_stay_together(tmp_path, caplog):
    series_uid = generate_uid()
    _copy_unnumbered(tmp_path, "noSeriesNum.dcm", series_uid)
    _copy_unnumbered(tmp_path, "invalidSeriesNum.dcm", series_uid)

    study = ProcessOneDicomStudyToVolumesMappingBase(tmp_path)

    assert set(study.series_dictionary) == {MISSING_VALUE}
    assert caplog.text.count(f"filed under series {MISSING_VALUE}\n") == 2


def test_split_does_not_depend_on_bvalue_tags(get_data_dir, tmp_path):
    """Anonymization can remove the b-value tags (issue #68)."""
    tracew_directory = get_data_dir / "5"
    siemens_bvalue_tags = [(0x0019, 0x100C), (0x0029, 0x1010)]
    for dcm_file in tracew_directory.rglob("*.dcm"):
        ds = pydicom.dcmread(dcm_file)
        for tag in siemens_bvalue_tags:
            if tag in ds:
                del ds[tag]
        # The Siemens sequence name also encodes the b-value, "ep_b1000t".
        ds.SequenceName = "ep_b"
        ds.save_as(tmp_path / dcm_file.name)

    def volumes_as_file_name_sets(study_directory: Path) -> tuple[set, set]:
        study = ProcessOneDicomStudyToVolumesMappingBase(study_directory)
        (series,) = study.series_dictionary.values()
        volumes = series.get_volume_list()
        file_name_sets = {
            frozenset(Path(f).name for f in volume.get_one_volume_dcm_filenames())
            for volume in volumes
        }
        return file_name_sets, {volume.get_volume_bvalue() for volume in volumes}

    original_volumes, original_bvalues = volumes_as_file_name_sets(tracew_directory)
    stripped_volumes, stripped_bvalues = volumes_as_file_name_sets(tmp_path)

    assert original_bvalues == {0, 1000}
    assert stripped_bvalues == {MISSING_VALUE}
    assert len(original_volumes) == 2
    assert stripped_volumes == original_volumes


@pytest.mark.parametrize("mr_sop_class_only", [False, True])
def test_mr_sop_class_only_skips_non_mr_series(
    contrast_file_path, tmp_path, mr_sop_class_only
):
    mr_file = next(contrast_file_path.rglob("*.dcm"))
    shutil.copy(mr_file, tmp_path / "mr.dcm")
    ds = pydicom.dcmread(mr_file)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 99
    ds.save_as(tmp_path / "secondary_capture.dcm")

    study = ProcessOneDicomStudyToVolumesMappingBase(
        tmp_path, mr_sop_class_only=mr_sop_class_only
    )
    mr_series_number = int(pydicom.dcmread(mr_file).SeriesNumber)
    assert mr_series_number in study.series_dictionary
    assert (99 in study.series_dictionary) is not mr_sop_class_only
