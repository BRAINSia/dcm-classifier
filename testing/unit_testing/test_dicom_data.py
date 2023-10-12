import pydicom
from pydicom.errors import InvalidDicomError
import pytest

def test_mr_dcm(get_mr_dcm):
    ds = pydicom.dcmread(get_mr_dcm)
    assert ds.Modality == "MR"

def test_ct_dcm(get_ct_dcm):
    ds = pydicom.dcmread(get_ct_dcm)
    assert ds.Modality == "CT"
