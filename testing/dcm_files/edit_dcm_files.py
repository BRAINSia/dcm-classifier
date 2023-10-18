import datetime
import os

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset, FileDataset
from pydicom.uid import UID
from pydicom.filewriter import dcmwrite

file_meta = FileMetaDataset()
file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
file_meta.ImplementationClassUID = UID("1.2.3.4")
file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
file_meta.FileMetaInformationGroupLength = 0
file_meta.FileMetaInformationVersion = b"\0\1"
file_meta.ImplementationVersionName = "PYDICOM"
file_meta.ImplementationClassUID = "1.3.46.670589.17"


ds = FileDataset("valid_file.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)

ds.PatientName = "Test^Firstname"
ds.PatientID = "123456"
ds.Modality = "MR"
ds.StudyDate = "20190101"
ds.StudyTime = "123456"
ds.StudyID = "123456789"
ds.SeriesNumber = 1234
ds.AccessionNumber = "123456789"
ds.PatientSex = "F"
ds.PatientOrientation = "L"
ds.PatientBirthDate = "20000101"
ds.StudyDescription = "Test Study 1"
ds.SeriesDescription = "Test Series 1"
ds.BodyPartExamined = "BRAIN"
ds.is_little_endian = True
ds.is_implicit_VR = True
ds.ClinicalTrialSponsorName = "Test Sponsor"
ds.ClinicalTrialProtocolID = "Test Protocol ID"
ds.StudyInstanceUID = "1.3.46.670589.17"
ds.SeriesInstanceUID = "1.3.46.670589.18"
ds.FrameOfReferenceUID = "1.3.46.670589.19"
ds.ImagePositionPatient = [0, 0, 0]
ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
ds.PixelSpacing = [1, 1]
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"
ds.Rows = 512
ds.Columns = 512
ds.BitsAllocated = 16
ds.BitsStored = 16
ds.HighBit = 15
ds.PixelRepresentation = 0
ds.ImageType = ["ORIGINAL", "PRIMARY", "OTHER"]
ds.ScanningSequence = "GR"
ds.SequenceVariant = "NONE"
ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
ds.SOPInstanceUID = "1.3.6.1.4.1.9328.50.51.2674852332200054861"

ds.PixelBandwidth = 100
ds.SAR = 100

print(ds)
dcmwrite("valid_file.dcm", ds)  # use `dcmwrite()` if creating file from scratch, `save_as()` doesn't include header info
print("*" * 80)

