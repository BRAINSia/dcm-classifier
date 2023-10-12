import pytest
import datetime
import os

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID

suffix = '.dcm'


# Populate required values for file meta information
file_meta = FileMetaDataset()
file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
file_meta.ImplementationClassUID = UID("1.2.3.4")

@pytest.fixture
def get_valid_dcm():
    realDcm = "valid_file" + suffix

    valid_ds = FileDataset(realDcm, {},
                           file_meta=file_meta, preamble=b"\0" * 128)

    valid_ds.PatientName = "Test^Firstname"
    valid_ds.PatientID = "123456"

    # Set the transfer syntax
    valid_ds.is_little_endian = True
    valid_ds.is_implicit_VR = True

    # Set creation date/time
    dt = datetime.datetime.now()
    valid_ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    valid_ds.ContentTime = timeStr

    # Write as a different transfer syntax XXX shouldn't need this but pydicom
    # 0.9.5 bug not recognizing transfer syntax
    valid_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRBigEndian
    valid_ds.is_little_endian = False
    valid_ds.is_implicit_VR = False

    realDcm = "testing/dcm_files/" + realDcm
    valid_ds.save_as(realDcm)

    return realDcm

@pytest.fixture
def get_invalid_dcm():

    invalidDcm = "invalid_file" + suffix

    invalid_ds: pydicom.Dataset = FileDataset(invalidDcm, {},
                           file_meta=file_meta)

    # Add the data elements -- not trying to set all required here. Check DICOM
    # standard
    invalid_ds.PatientName = "Test^Firstname"
    invalid_ds.PatientID = "123456"

    # Set the transfer syntax
    invalid_ds.is_little_endian = True
    invalid_ds.is_implicit_VR = True

    # Set creation date/time
    dt = datetime.datetime.now()
    invalid_ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    invalid_ds.ContentTime = timeStr

    # print("Writing invalid file", invalidDcm)
    # invalid_ds.save_as(invalidDcm)
    # print("File saved.")

    # Write as a different transfer syntax XXX shouldn't need this but pydicom
    # 0.9.5 bug not recognizing transfer syntax
    invalid_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRBigEndian
    invalid_ds.is_little_endian = False
    invalid_ds.is_implicit_VR = False

    # print("Writing test file as Big Endian Explicit VR", invalidDcm)
    # invalid_ds.save_as(invalidDcm)

    invalidDcm = "testing/dcm_files/" + invalidDcm
    invalid_ds.save_as(invalidDcm)

    return invalidDcm

@pytest.fixture
def get_MR_dcm():
    mrDcm = "mr_file" + suffix

    ds: pydicom.Dataset = FileDataset(mrDcm, {},
                           file_meta=file_meta, preamble=b"\0" * 128)

    ds.PatientName = "Test^Firstname"
    ds.PatientID = "123456"

    ds.Modality = "MR"
    ds.SeriesDescription = "T1"
    ds.SeriesNumber = "1"

    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr

    mrDcm = "testing/dcm_files/" + mrDcm
    ds.save_as(mrDcm)

    return mrDcm

@pytest.fixture
def get_CT_dcm():
    ctDcm = "ct_file" + suffix

    ds: pydicom.Dataset = FileDataset(ctDcm, {},
                           file_meta=file_meta, preamble=b"\0" * 128)

    ds.PatientName = "Test^Firstname"
    ds.PatientID = "123456"

    ds.Modality = "CT"
    ds.SeriesDescription = "T1"
    ds.SeriesNumber = "1"

    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr

    ctDcm = "testing/dcm_files/" + ctDcm
    ds.save_as(ctDcm)

    return ctDcm


