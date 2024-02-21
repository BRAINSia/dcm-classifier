from pathlib import Path

import pandas as pd
from dcm_classifier.dicom_volume import (
    DicomSingleVolumeInfoBase,
)
from collections import OrderedDict
import pytest
from dcm_classifier.utility_functions import FImageType

current_file_path = Path(__file__).parent


def test_get_series_uid(mock_volumes):
    series_uid = DicomSingleVolumeInfoBase(mock_volumes[0]).get_series_uid()
    assert series_uid == "1.2.276.0.7230010.3.1.3.168456204.6074.1606326635.433425"


def test_get_series_number(mock_volumes):
    series_number = DicomSingleVolumeInfoBase(mock_volumes[0]).get_series_number()
    assert series_number == 700


def test_get_study_uid(mock_volumes):
    study_uid = DicomSingleVolumeInfoBase(mock_volumes[0]).get_study_uid()
    assert study_uid == "1.2.276.0.7230010.3.1.2.168456204.6074.1606326635.433364"


def test_get_b_value(mock_volumes):
    # happy path for a volune that does have a b-value
    b_value = DicomSingleVolumeInfoBase(mock_volumes[1]).get_volume_bvalue()
    assert b_value == 100

    # sad path for a volume that doesn't have a b-value
    b_value = DicomSingleVolumeInfoBase(mock_volumes[0]).get_volume_bvalue()
    assert b_value == -12345


def test_primary_volume_info(mock_volumes):
    volume_info = DicomSingleVolumeInfoBase(mock_volumes[0]).get_primary_volume_info(0)
    assert isinstance(volume_info, OrderedDict)


def test_get_series_pixel_spacing(mock_volumes):
    pixel_spacing = DicomSingleVolumeInfoBase(
        mock_volumes[0]
    ).get_series_pixel_spacing()
    assert isinstance(pixel_spacing, str)
    assert pixel_spacing == "[0.9375, 0.9375]"


# sad path, validate is False due to sentinel b-value seen in test above
# def test_validate(mock_volumes):
#     mock_volume = DicomSingleVolumeInfoBase(mock_volumes[0])
#     volume_validate = mock_volume.validate()
#     assert volume_validate is False
#     assert distance_between_slices != "-12345.0"
#     assert mock_volume.get_volume_bvalue() == -12345


def test_image_diagnostics(mock_volumes):
    volume_info = DicomSingleVolumeInfoBase(mock_volumes[0])
    _ = volume_info.get_image_diagnostics()


def test_get_series_size(mock_volumes):
    series_size = DicomSingleVolumeInfoBase(mock_volumes[0]).get_series_size()
    assert isinstance(series_size, str)
    assert series_size == "[256, 256, 22]"


# sad path: testing inconsistent slice error is being found in the validation report string
# def test_generate_validation_report_str(mock_volumes):
#     # volume_info = DicomSingleVolumeInfoBase(mock_volumes[1])
#     # volume_info.validate()
#     # validation_report_str = volume_info.generate_validation_report_str()
#     # assert "ERROR: Inconsistent slice thickness found" in validation_report_str
#     # assert "Identified image type: DWI" in validation_report_str
#     # assert "Identified bvalue: 100" in validation_report_str
#     # assert "Identified SeriesNumber: 702" in validation_report_str


# @pytest.mark.skip(reason="Not implemented yet")
def test_get_invalid_vol_itk_image(mock_volumes):
    # TODO - implement this test
    with pytest.raises(FileNotFoundError) as ex:
        _ = DicomSingleVolumeInfoBase(mock_volumes[0]).get_itk_image()
    assert "No DICOMs in: " in str(ex.value)
    # assert image is not None
    # assert isinstance(image, FImageType)


def test_get_itk_image(get_data_dir):
    dicom_file_dir = get_data_dir / "1" / "DICOM"
    assert dicom_file_dir.exists()
    vol = list()
    for file in dicom_file_dir.iterdir():
        vol.append(file)

    image = DicomSingleVolumeInfoBase(vol).get_itk_image()
    assert image is not None
    assert isinstance(image, FImageType)


@pytest.mark.skip(reason="Not implemented yet")
def test_get_one_volume_dcm_filenames():
    pass


def test_set_modality(mock_volumes):
    volume_info = DicomSingleVolumeInfoBase(mock_volumes[0])
    assert volume_info.get_volume_modality() == "INVALID"
    volume_info.set_volume_modality("t1w")
    assert volume_info.get_volume_modality() == "t1w"


def test_get_modality(mock_volumes):
    volume_info = DicomSingleVolumeInfoBase(mock_volumes[0])
    assert volume_info.get_volume_modality() == "INVALID"
    volume_info.set_volume_modality("fa")
    assert volume_info.get_volume_modality() == "fa"


@pytest.mark.skip(reason="Not implemented yet")
def test_set_modality_probabilities(mock_volumes):
    probabilities = pd.DataFrame(
        data={"case1": 0.4, "case2": 0.3, "case3": 0.75}, index=["t1w", "flair", "t2w"]
    )
    vol = DicomSingleVolumeInfoBase(mock_volumes[0])
    vol.set_modality_probabilities(probabilities)
    assert vol.get_modality_probabilities().aggregate == probabilities


def test_no_files_provided():
    with pytest.raises(ValueError) as ex:
        _ = DicomSingleVolumeInfoBase([])
    assert "No file names provided list" in str(ex.value)


def test_invalid_volume_modality(mock_volumes):
    assert DicomSingleVolumeInfoBase(mock_volumes[0]).get_volume_modality() == "INVALID"


@pytest.mark.skip(reason="Not implemented yet")
def test_get_modality_probabilities():
    pass


# Test asserts false because the mock volume has a flair modality but that is an MR modality...
def test_is_MR_modality(mock_volume_study):
    for series_num, series in mock_volume_study.get_study_dictionary().items():
        for volume in series.get_volume_list():
            assert volume.is_MR_modality() is False
