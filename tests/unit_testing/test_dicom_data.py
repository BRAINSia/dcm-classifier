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

    # study_all_adc = ProcessOneDicomStudyToVolumesMappingBase(study_directory=current_file_path.parent / "testing_data" / "adc_volumes" / "volume_0", inferer=inferer)
    # study_all_adc.run_inference()
    #
    # for series_number, series in study.series_dictionary.items():
    #     assert series.get_modality() == "adc"

def test_ax_dcm_series_acq_plane():
    inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path)
    dicom_files_dir: Path = current_file_path.parent.parent.parent / "testDcm" / "dcm_files"
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=dicom_files_dir, inferer=inferer
    )
    study.run_inference()
    for series_number, series in study.series_dictionary.items():
        assert series.get_acquisition_plane() == "ax"


def test_no_valid_dicoms():
    with pytest.raises(FileNotFoundError) as ex:
        inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path)
        dicom_files_dir: Path = current_file_path.parent / "dummy_directory"
        study = ProcessOneDicomStudyToVolumesMappingBase(
            study_directory=dicom_files_dir, inferer=inferer
        )
        study.run_inference()
    assert "No DICOMs in: " in str(ex.value)


@pytest.mark.skip(reason="Volume help needed")
def test_adc_dcm_volume_modality(mock_volumes, default_image_type_classifier_base):
    inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path, mode="volume")
    dicom_files_dir: Path = current_file_path.parent.parent.parent / "testDcm/dcm_files"
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=dicom_files_dir, inferer=inferer
    )
    study.run_inference()
    for volume in study.get_list_of_primary_volume_info():
        assert volume["Modality"] == "adc"


@pytest.mark.skip(reason="Volume help needed")
def test_adc_dcm_volume_acq_plane(mock_volumes, default_image_type_classifier_base):
    inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path, mode="volume")
    dicom_files_dir: Path = current_file_path.parent.parent.parent / "testDcm/dcm_files"
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=dicom_files_dir, inferer=inferer
    )
    study.run_inference()
    for volume in study.get_list_of_primary_volume_info():
        print(volume)
        assert volume == "ax"
