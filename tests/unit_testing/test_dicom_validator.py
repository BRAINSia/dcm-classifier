from dcm_classifier.dicom_validator import DicomValidatorBase
from dcm_classifier.dicom_volume import DicomSingleVolumeInfoBase
from pathlib import Path


def test_volume_validation(mock_volume_study, mock_volumes):
    validator = DicomValidatorBase(DicomSingleVolumeInfoBase(mock_volumes[0]))
    validator.append_to_validation_failure_reports("testing")
    validator.append_to_validation_failure_reports("This is a TEST")

    report = validator.generate_validation_report_str(verbose_reporting=True)
    assert report is not None
    assert "Failure Messages:" in report
    assert "testing" in report and "This is a TEST" in report


# Always going to pass, `validate()` returns True
def test_validate(mock_volumes):
    validator = DicomValidatorBase(DicomSingleVolumeInfoBase(mock_volumes[0]))
    assert validator.validate() is not False


def test_write_validation_report(mock_volumes):
    validator = DicomValidatorBase(DicomSingleVolumeInfoBase(mock_volumes[0]))
    validator.append_to_validation_failure_reports("testing")
    validator.append_to_validation_failure_reports("This is a TEST")

    test_report_path: Path = (
        Path(__file__).parent.parent / "testing_data" / "test_report.txt"
    )
    validator.write_validation_report(test_report_path)
    assert test_report_path.exists()
    assert test_report_path.is_file()

    with open(test_report_path) as f:
        msg = f.read()
        assert "testing" in msg and "This is a TEST" in msg

    test_report_path.unlink(missing_ok=True)


def test_write_validation_report_append(mock_volumes, capsys):
    validator = DicomValidatorBase(DicomSingleVolumeInfoBase(mock_volumes[0]))
    validator.append_to_validation_failure_reports("testing")
    validator.write_validation_report(None)
    captured = capsys.readouterr()
    print(captured.out)
    assert "testing" in captured.out and "Identified bvalue: -12345" in captured.out
