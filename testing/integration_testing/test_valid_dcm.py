from pathlib import Path

import pydicom
import pytest
from dcm_classifier.dicom_volume import DicomSingleVolumeInfoBase
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from pydicom.errors import InvalidDicomError

current_file_path = Path(__file__).parent.resolve()
inference_model_path = list(Path(__file__).parent.parent.parent.rglob("models/rf_classifier.onnx"))[0]


# @pytest.mark.parametrize("file", [])
def test_dcm_validity():
    inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path)
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=current_file_path.parent / "dcm_files", inferer=inferer
    )
    # study.run_inference()

    volumes = study.get_list_of_primary_volume_info()
    dict = study.series_dictionary
    print(dict.get(1).get_modality())
    print(volumes)
    assert len(volumes) == 1


@pytest.mark.parametrize("file, modality", [("new_MRBRAIN.DCM", "MR"), ("a_valid_file.dcm", "MR"), ("valid_CT_file.dcm", "CT")])
def test_mr_dcm(file, modality):
    # inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path)
    # study = ProcessOneDicomStudyToVolumesMappingBase(
    #     study_directory=current_file_path.parent.parent.parent / "dcm_files", inferer=inferer
    # )
    # study.run_inference()

    # print(study.get_list_of_primary_volume_info())
    # print(study.series_dictionary.get(3).volume_info_list)
    print(DicomSingleVolumeInfoBase([current_file_path.parent / "dcm_files" / file]).get_modality())
    assert DicomSingleVolumeInfoBase(
        [current_file_path.parent / "dcm_files" / file]).get_modality() == modality


def test_CT_dcm():
    pass
