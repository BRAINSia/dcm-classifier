import pydicom
from classify_study
from pydicom.errors import InvalidDicomError
import pytest

def test_mr_dcm_single_vol(get_mr_dcm):
    volume = DicomSingleVolumeInfoBase(get_mr_dcm)
    assert volume.get_modality == "MR"

def test_ct_dcm_single_vol(get_ct_dcm):
    volume = DicomSingleVolumeInfoBase(get_ct_dcm)
    assert volume.get_modality == "CT"
