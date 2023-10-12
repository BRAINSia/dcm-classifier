import pydicom
from pydicom.errors import InvalidDicomError
import pytest

def test_valid_dcm(get_valid_dcm):
    assert type(pydicom.dcmread(get_valid_dcm)) == pydicom.dataset.FileDataset

def test_invalid_dcm(get_invalid_dcm):
    with pytest.raises(InvalidDicomError) as ex:
        pydicom.dcmread(get_invalid_dcm)
    assert "File is missing DICOM File Meta Information" in str(ex.value)
