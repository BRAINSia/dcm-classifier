import pydicom
from pydicom.errors import InvalidDicomError
import pytest
# from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from pathlib import Path


current_file_path = Path(__file__).parent.resolve()
inference_model_path = list(Path(__file__).parent.parent.parent.rglob("models/rf_classifier.onnx"))[0]
print(inference_model_path)
def test_path():
    print(inference_model_path)
    assert inference_model_path.exists()

def test_dcm_modality():
    # inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path)
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory="~/cavriley/programs/testDcm/dcm_files", inferer=None
        )
    # study.run_inference()
    assert study.get_list_of_primary_volume_info()[0]["modality"] == "MR"

    volumes = ProcessOneDicomStudyToVolumesMappingBase("testing/dcm_files").get_list_of_primary_volume_info()
    print(volumes)
    assert len(volumes) == 1
    assert volumes[0].get("Modality") == "MR"


# def test_ct_dcm_single_vol(get_ct_dcm):
#     inferer = ImageTypeClassifierBase(classification_model_filename="testing/dcm_files/a_valid_file.dcm")
#     study = ProcessOneDicomStudyToVolumesMappingBase(
#         study_directory="testing/dcm_files/a_valid_file.dcm", inferer=inferer
#         )
#     study.run_inference()
#
#     assert study.get_modality == "CT"
