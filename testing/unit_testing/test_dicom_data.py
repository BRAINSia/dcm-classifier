import pydicom
from pydicom.errors import InvalidDicomError
import pytest

def test_mr_dcm(get_mr_dcm):
    assert type(pydicom.dcmread(get_mr_dcm)) == pydicom.dataset.FileDataset
