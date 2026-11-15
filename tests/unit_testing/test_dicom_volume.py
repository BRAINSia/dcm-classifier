from collections import OrderedDict
from pathlib import Path

import pytest
from deprecation import DeprecatedWarning

from dcm_classifier.dicom_config import MISSING_VALUE
from dcm_classifier.dicom_volume import (
    DicomSingleVolumeInfoBase,
)
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from dcm_classifier.utility_functions import FImageType

current_file_path = Path(__file__).parent
inference_model_path = next(
    iter(Path(__file__).parent.parent.parent.rglob("models/rf_classifier.onnx"))
)

# The mock volumes have no SliceThickness, which the library reports.
pytestmark = pytest.mark.filterwarnings(
    "ignore:(?s).*Required DICOM fields.*SliceThickness:UserWarning"
)

inferer = ImageTypeClassifierBase(classification_model_filename=inference_model_path)


def test_get_series_uid(mock_volumes):
    with pytest.warns(
        DeprecatedWarning, match=r"removed in 1\.0\.0.*get_dicom_field_by_name"
    ):
        series_uid = DicomSingleVolumeInfoBase(mock_volumes[0]).get_series_uid()
    assert series_uid == "1.2.276.0.7230010.3.1.3.168456204.6074.1606326635.433425"


def test_get_series_number(mock_volumes):
    with pytest.warns(DeprecatedWarning, match="Use generic `get_dicom_field_by_name"):
        series_number = DicomSingleVolumeInfoBase(mock_volumes[0]).get_series_number()
    assert series_number == 700


def test_get_study_uid(mock_volumes):
    with pytest.warns(DeprecatedWarning, match="Use generic `get_dicom_field_by_name"):
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
    with pytest.warns(DeprecatedWarning, match="Use generic `get_dicom_field_by_name"):
        pixel_spacing = DicomSingleVolumeInfoBase(
            mock_volumes[0]
        ).get_series_pixel_spacing()
    assert isinstance(pixel_spacing, str)
    assert pixel_spacing == "[0.9375, 0.9375]"


def test_image_diagnostics(mock_volumes):
    volume_info = DicomSingleVolumeInfoBase(mock_volumes[0])
    _ = volume_info.get_image_diagnostics()


def test_get_series_size(mock_volumes):
    series_size = DicomSingleVolumeInfoBase(mock_volumes[0]).get_series_size()
    assert isinstance(series_size, str)
    assert series_size == "[256, 256, 22]"


def test_get_invalid_vol_itk_image(mock_volumes):
    with pytest.raises(FileNotFoundError) as ex:
        _ = DicomSingleVolumeInfoBase(mock_volumes[0]).get_itk_image()
    assert "No DICOMs in: " in str(ex.value)


def test_get_itk_image(get_data_dir):
    dicom_file_dir = get_data_dir / "1" / "DICOM"
    assert dicom_file_dir.exists()
    vol = list(dicom_file_dir.iterdir())

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


def test_no_files_provided():
    with pytest.raises(ValueError, match="No file names provided list"):
        _ = DicomSingleVolumeInfoBase([])


def test_invalid_volume_modality(mock_volumes):
    assert DicomSingleVolumeInfoBase(mock_volumes[0]).get_volume_modality() == "INVALID"


# Test asserts false because the mock volume has a flair modality but that is an MR modality...
def test_is_MR_modality(mock_volume_study):
    for series in mock_volume_study.get_study_dictionary().values():
        for volume in series.get_volume_list():
            assert volume.is_MR_modality() is False


def test_setting_invalid_modality(mock_tracew_series):
    for series in mock_tracew_series:
        for volume in series.get_volume_list():
            with pytest.raises(ValueError, match="Can only set_modality with a str"):
                volume.set_volume_modality(4)


def test_get_series_modality_from_subvol(mock_tracew_series):
    for series in mock_tracew_series:
        for volume in series.get_volume_list():
            assert volume.get_series_modality() == "tracew"


def test_get_parent_series(mock_tracew_series):
    for series in mock_tracew_series:
        for volume in series.get_volume_list():
            assert volume.get_parent_series() == series


def test_setting_invalid_modality_probabilities(mock_tracew_series):
    for series in mock_tracew_series:
        for volume in series.get_volume_list():
            with pytest.raises(
                ValueError,
                match=r"Can only set_modality_probabilities with a DataFrame",
            ):
                volume.set_modality_probabilities(4)


@pytest.mark.parametrize("file_name", ["invalidSeriesNum.dcm", "noSeriesNum.dcm"])
def test_unusable_series_number_reads_as_missing(get_data_dir, file_name):
    dicom_path = get_data_dir.parent / "invalid_data" / "invalid_fields" / file_name
    with pytest.warns(UserWarning, match="SeriesNumber"):
        volume = DicomSingleVolumeInfoBase([dicom_path.as_posix()])

    with pytest.warns(DeprecationWarning, match="get_dicom_field_by_name"):
        series_number = volume.get_series_number()
    assert series_number == MISSING_VALUE


def test_get_volume_index(mock_tracew_series):
    for series in mock_tracew_series:
        for volume in series.get_volume_list():
            volume.set_volume_index(1)

            assert volume.get_volume_index() == 1


def test_dcm_vol_has_contrast(contrast_file_path):
    assert contrast_file_path.exists()

    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=contrast_file_path, inferer=inferer
    )
    study.run_inference()

    for series in study.series_dictionary.values():
        assert series.get_volume_list()[0].get_has_contrast() is True

        assert series.get_volume_list()[0].get_contrast_agent() != "None"


def test_dcm_vol_no_contrast(no_contrast_file_path):
    assert no_contrast_file_path.exists()

    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=no_contrast_file_path, inferer=inferer
    )
    study.run_inference()

    for series in study.series_dictionary.values():
        assert series.get_volume_list()[0].get_has_contrast() is False

        assert series.get_volume_list()[0].get_contrast_agent() == "None"


def test_dataset_value_set_to_none(get_data_dir):
    file_path = get_data_dir.parent / "empty_data_fields"
    assert file_path.exists()

    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=file_path, inferer=inferer
    )
    study.run_inference()

    series = study.get_study_dictionary().get(1)

    assert series.get_has_contrast() is False

    assert series.get_volume_list()[0].get_volume_modality() == "gret2star"
    assert (
        series.get_volume_list()[0].get_volume_dictionary()["ContrastBolusAgent"]
        == "None"
    )
    assert (
        series.get_volume_list()[0].get_volume_dictionary()["InversionTime"] == "-12345"
    )
    assert series.get_volume_list()[0].get_volume_dictionary()["SAR"] == "-12345"


def test_t1w_dcm_volume_modality(mock_volume_study):
    for series_num, series in mock_volume_study.get_study_dictionary().items():
        for volume in series.get_volume_list():
            if 12 <= series_num <= 15:
                assert volume.get_volume_modality() == "t1w"


def test_ax_dcm_volume_acq_plane(mock_volume_study):
    for series_num, series in mock_volume_study.get_study_dictionary().items():
        for volume in series.get_volume_list():
            if 5 <= series_num <= 12 and series_num != 10:
                assert volume.get_acquisition_plane() == "ax"


def test_sag_dcm_volume_acq_plane(mock_volume_study):
    for series_num, series in mock_volume_study.get_study_dictionary().items():
        for volume in series.get_volume_list():
            if series_num in {10, 13}:
                assert volume.get_acquisition_plane() == "sag"


def test_cor_dcm_volume_acq_plane(mock_volume_study):
    for series_number, series in mock_volume_study.get_study_dictionary().items():
        for volume in series.get_volume_list():
            if series_number == 15:
                assert volume.get_acquisition_plane() == "cor"


def test_scanning_sequence_in_flair(mock_flair_series):
    for series in mock_flair_series:
        for volume in series.get_volume_list():
            assert volume.get_volume_dictionary()["ScanningSequence_IR"] == 1
            assert volume.get_volume_dictionary()["ScanningSequence_SE"] == 1
