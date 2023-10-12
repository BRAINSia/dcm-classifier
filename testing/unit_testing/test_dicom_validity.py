import pydicom
from pydicom.errors import InvalidDicomError
import pytest
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase

volumes = ProcessOneDicomStudyToVolumesMappingBase("testing/dcm_files").get_list_of_primary_volume_info()
def test_valid_dcm(get_valid_dcm):

    assert volumes[0] == pydicom.dataset.FileDataset

def test_invalid_dcm(get_invalid_dcm):
    with pytest.raises(InvalidDicomError) as ex:
        volumes[1] == pydicom.dataset.FileDataset
    assert "File is missing DICOM File Meta Information" in str(ex.value)
