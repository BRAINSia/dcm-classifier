from pathlib import Path

import pydicom
import pytest
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from pydicom.errors import InvalidDicomError

current_file_path = Path(__file__).parent.resolve()
inference_model_path = list(Path(__file__).parent.parent.parent.rglob("models/rf_classifier.onnx"))[0]


def test_dcm_validity():
    inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path)
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory="testing/dcm_files", inferer=inferer
        )
    study.run_inference()

    volumes = study.get_list_of_primary_volume_info()

    assert len(volumes) == 1


    # with pytest.raises(InvalidDicomError) as ex:
    #
    # assert "File is missing DICOM File Meta Information" in str(ex.value)
