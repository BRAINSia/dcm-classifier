from pathlib import Path
import pandas as pd
import pydicom

current_file: Path = Path(__file__)

root_dir: Path = current_file.parent.parent

test_data: Path = root_dir / "tests" / "testing_data" / "anonymized_testing_data" / "anonymized_data" / "1" / "DICOM"

# first_file: Path = list(test_data.rglob("*.dcm"))[0]
#
# ds: pydicom.dataset.FileDataset = pydicom.dcmread(first_file, stop_before_pixels=True)
#
# fields: list[str] = [str(elem.name) for elem in ds]
#
# print(fields)

fields = ['Specific Character Set',
          'Image Type',
          'Instance Creation Date',
          'Instance Creation Time',
          'SOP Class UID',
          'SOP Instance UID',
          'Study Date',
          'Series Date',
          'Acquisition Date',
          'Content Date',
          'Study Time',
          'Series Time',
          'Acquisition Time',
          'Content Time',
          'Accession Number',
          'Modality',
          'Modalities in Study',
          'Manufacturer',
          "Referring Physician's Name",
          'Procedure Code Sequence',
          "Manufacturer's Model Name",
          "Patient's Name",
          'Patient ID',
          "Patient's Birth Date",
          "Patient's Sex",
          'Patient Identity Removed',
          'De-identification Method',
          'De-identification Method Code Sequence',
          'Body Part Examined',
          'Scanning Sequence',
          'Sequence Variant',
          'Scan Options',
          'MR Acquisition Type',
          'Sequence Name',
          'Angio Flag',
          'Slice Thickness',
          'Repetition Time',
          'Echo Time',
          'Number of Averages',
          'Imaging Frequency',
          'Imaged Nucleus',
          'Echo Number(s)',
          'Magnetic Field Strength',
          'Number of Phase Encoding Steps',
          'Echo Train Length',
          'Percent Sampling',
          'Percent Phase Field of View',
          'Pixel Bandwidth',
          'Software Versions',
          'Transmit Coil Name',
          'Acquisition Matrix',
          'In-plane Phase Encoding Direction',
          'Flip Angle',
          'Variable Flip Angle Flag',
          'SAR',
          'dB/dt',
          'Patient Position',
          'Study Instance UID',
          'Series Instance UID',
          'Study ID',
          'Series Number',
          'Acquisition Number',
          'Instance Number',
          'Image Position (Patient)',
          'Image Orientation (Patient)',
          'Frame of Reference UID',
          'Position Reference Indicator',
          'Slice Location',
          'Number of Study Related Instances',
          'Samples per Pixel',
          'Photometric Interpretation',
          'Rows',
          'Columns',
          'Pixel Spacing',
          'Bits Allocated',
          'Bits Stored',
          'High Bit',
          'Pixel Representation',
          'Smallest Image Pixel Value',
          'Largest Image Pixel Value',
          'Longitudinal Temporal Information Modified',
          'Window Center',
          'Window Width',
          'Window Center & Width Explanation',
          'Private Creator',
          '[CSA Image Header Info]',
          'Performed Procedure Step Start Date',
          'Confidentiality Code']

print(len(fields))
