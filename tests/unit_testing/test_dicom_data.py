import pydicom
from pydicom.errors import InvalidDicomError
import pytest
from dcm_classifier.dicom_volume import (
    DicomSingleVolumeInfoBase,
)
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from pathlib import Path


current_file_path = Path(__file__).parent.resolve()
inference_model_path = list(
    Path(__file__).parent.parent.parent.rglob("models/rf_classifier.onnx")
)[0]


def test_adc_dcm_series_modality(mock_volumes, default_image_type_classifier_base):
    inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path)
    dicom_files_dir: Path = current_file_path.parent.parent.parent / "testDcm" / "dcm_files"
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=dicom_files_dir, inferer=inferer
    )
    study.run_inference()

    for series_number, series in study.series_dictionary.items():
        assert series.get_modality() == "adc"


def test_ax_dcm_series_acq_plane(mock_series_study):
    inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path)
    dicom_files_dir: Path = current_file_path.parent.parent.parent / "testDcm" / "dcm_files"
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=dicom_files_dir, inferer=inferer
    )
    study.run_inference()
    for series_number, series in study.series_dictionary.items():
        assert series.get_acquisition_plane() == "ax"

    # mock series assertions
    for series_number, series in mock_series_study.series_dictionary.items():
        if 6 <= series_number <= 9:
            assert series.get_acquisition_plane() == "ax"


def test_sag_dcm_series_acq_plane(mock_series_study):
    for series_number, series in mock_series_study.series_dictionary.items():
        if series_number == 2 or series_number == 10 or series_number == 13:
            assert series.get_acquisition_plane() == "sag"


def test_cor_dcm_series_acq_plane(mock_series_study):
    for series_number, series in mock_series_study.series_dictionary.items():
        if series_number == 3 or series_number == 15:
            assert series.get_acquisition_plane() == "cor"


def test_t1_dcm_series_modality(mock_series_study):
    for series_number, series in mock_series_study.series_dictionary.items():
        if 10 <= series_number <= 15 and series_number != 11:
            assert series.get_modality() == "t1w"


def test_flair_dcm_series_modality(mock_series_study):
    for series_number, series in mock_series_study.series_dictionary.items():
        if series_number == 7:
            assert series.get_modality() == "flair"


def test_t2_dcm_series_modality(mock_series_study):
    for series_number, series in mock_series_study.series_dictionary.items():
        if series_number == 11:
            assert series.get_modality() == "t2w"


def test_no_valid_dicoms():
    with pytest.raises(FileNotFoundError) as ex:
        inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path)
        dicom_files_dir: Path = current_file_path.parent / "testing_data" / "dummy_directory"
        study = ProcessOneDicomStudyToVolumesMappingBase(
            study_directory=dicom_files_dir, inferer=inferer
        )
        study.run_inference()
    assert "No DICOMs in: " in str(ex.value)


@pytest.mark.skip(reason="Volume help needed")
def test__dcm_volume_modality(mock_volume_study, mock_volumes):
    # for series_num, series in mock_volume_study.get_study_dictionary().items():
    #     if 12 <= series_num <= 15:
    #         assert series.get_modality() == "t1w"
    # print(mock_volume_study.series_dictionary[7].get_volume_dictionary())
    # print(DicomSingleVolumeInfoBase(mock_volume_study.series_dictionary[7]).get_modality())
    dwi_vol = DicomSingleVolumeInfoBase(mock_volumes[1])
    print(dwi_vol.get_modality())
    # print(mock_volume_study.series_dictionary[6].get_modality())
    # single_series = DicomSingleSeries(mock_volume_study.series_dictionary[7])
    # assert DicomSingleVolumeInfoBase(mock_volumes[0]).get_modality() == "adc"


@pytest.mark.skip(reason="Volume help needed")
def test_adc_dcm_volume_acq_plane(mock_volumes, default_image_type_classifier_base):
    inferer = default_image_type_classifier_base
    dicom_files_dir: Path = current_file_path.parent.parent.parent / "testDcm" / "dcm_files"
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=dicom_files_dir, inferer=inferer
    )
    study.run_inference()
    for volume in study.get_list_of_primary_volume_info():
        print(volume)
        assert volume == "ax"
