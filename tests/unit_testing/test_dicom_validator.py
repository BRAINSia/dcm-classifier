from dcm_classifier.dicom_validator import DicomValidatorBase
from dcm_classifier.dicom_volume import DicomSingleVolumeInfoBase
from pathlib import Path
import pytest


@pytest.mark.skip(reason="Not implemented yet")
def test_volume_validation(mock_volume_study, mock_volumes):
    # print(type(mock_volume_study))  # <class 'dcm_classifier.study_processing.ProcessOneDicomStudyToVolumesMappingBase'>
    validator = DicomValidatorBase(DicomSingleVolumeInfoBase(mock_volumes[0]))
    report = validator.generate_validation_report_str(verbose_reporting=True)
    print(report)
    assert len(report) > 0
    # assert validator._validation_failure_reports == [
    #     "Series 700 has a sentinel b-value of -12345.0"
    # ]


# Always going to pass, `validate()` returns True
def test_validate(mock_volumes):
    validator = DicomValidatorBase(DicomSingleVolumeInfoBase(mock_volumes[0]))
    assert validator.validate() is not False
