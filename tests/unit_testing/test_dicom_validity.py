from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError
import pytest

from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.dicom_volume import (
    DicomSingleVolumeInfoBase,
)

relative_testing_path = Path(__file__).parent.parent.resolve()


@pytest.mark.skip(reason="Not implemented yet")
def test_dcm_validity():
    # inferer = ImageTypeClassifierBase(classification_model_filename="testing/dcm_files/1.2.840.113619.2.55.3.2833516411.326.1590519953.1.1.dcm")
    # # study = ProcessOneDicomStudyToVolumesMappingBase(
    # #     study_directory="testing/dcm_files/DICOM", inferer=inferer
    # #     )
    # study = ProcessOneDicomStudyToVolumesMappingBase("testing/dcm_files")
    # study.set_inferer(inferer)
    # study.run_inference()
    #
    # volumes = ProcessOneDicomStudyToVolumesMappingBase("testing/dcm_files").get_list_of_primary_volume_info()
    print(relative_testing_path)
    assert relative_testing_path.exists()
    volume = DicomSingleVolumeInfoBase(
        [relative_testing_path / "dcm_files" / "valid_file.dcm"]
    )
    print(volume)
    # assert len(volumes) == 1
    # assert len(study.get_list_of_primary_volume_info()) == 1


# def test_invalid_dcm(get_invalid_dcm):
#     with pytest.raises(InvalidDicomError) as ex:
#         volumes[1] == pydicom.dataset.FileDataset
#     assert "File is missing DICOM File Meta Information" in str(ex.value)
