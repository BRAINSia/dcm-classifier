from dcm_classifier.dicom_validator import DicomValidatorBase
from dcm_classifier.dicom_volume import DicomSingleVolumeInfoBase
from pathlib import Path
import pytest


@pytest.mark.skip(reason="Not implemented yet")
def test_volume_validation(mock_volumes):
    validator = DicomValidatorBase(DicomSingleVolumeInfoBase(mock_volumes[0]))
    report = validator.generate_validation_report_str()
    print(report)
    # assert validator._validation_failure_reports == [
    #     "Series 700 has a sentinel b-value of -12345.0"
    # ]
