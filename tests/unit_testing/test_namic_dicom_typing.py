import pydicom
import pytest
from dcm_classifier.dicom_volume import DicomSingleVolumeInfoBase
from dcm_classifier.example_image_processing import slugify, rglob_for_singular_result
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.namic_dicom_typing import vprint, get_coded_dictionary_elements, check_for_diffusion_gradient, \
    convert_array_to_min_max, sanitize_dicom_dataset
from dcm_classifier.dicom_config import required_DICOM_fields, optional_DICOM_fields
from pathlib import Path

from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase

relative_testing_data_path: Path = Path(__file__).parent.parent / "testing_data"
current_file_path: Path = Path(__file__).parent
inference_model_path = list(current_file_path.parent.parent.rglob("models/rf_classifier.onnx"))[0]


def test_rglob_for_singular_result():
    single_file_dir = (
        relative_testing_data_path / "dummy_directory" / "dir_with_one_file"
    )
    single_file = rglob_for_singular_result(single_file_dir, "*.file", "f")
    assert single_file is not None
    # TODO: Need to verify that failure occurs when more than 1 file is found
    # double_file_dir = relative_testing_data_path / "dummy_directory" / "dir_with_two_files"
    # double_file = rglob_for_singular_result(double_file_dir, "*.file", "f")


def test_slugify():
    # TODO: Test should have upper, lower, and unicode characters included in all input test strings
    # TODO: Something like:
    # smiley_unicode = u'\U0001f604'.encode('unicode-escape'
    # f"""start_UPPER_lower_!_@_#_"_'_:_unicode_{smiley_unicode}_end"""
    # NOTE: The same inpute string can be used for all test cases,
    #       there is no need to have multiple input strings
    test_strings = [
        "dwi series number 006 index 01",
        "DWI-series-number-006-index-01",
        "dwi-series-nùmber-006-index-01",
    ]
    results = []

    results.append(slugify(test_strings[0]))
    results.append(slugify(test_strings[1], allow_uppercase=True))
    results.append(slugify(test_strings[2], allow_unicode=True))
    #
    # TODO: results.append(slugify(test_strings[2], allow_uppercase=False))
    # TODO: results.append(slugify(test_strings[4], allow_unicode=False))
    #
    # TODO: results.append(slugify(test_strings[5], allow_uppercase=True, allow_unicode=True))
    # TODO: results.append(slugify(test_strings[6], allow_uppercase=False, allow_unicode=True))
    # TODO: results.append(slugify(test_strings[7], allow_uppercase=True, allow_unicode=False))
    # TODO: results.append(slugify(test_strings[8], allow_uppercase=False, allow_unicode=False))

    assert results == [
        "dwi-series-number-006-index-01",
        "DWI-series-number-006-index-01",
        "dwi-series-nùmber-006-index-01",
    ]


@pytest.mark.skip(reason="Not implemented yet")
def test_cmd_exists():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_compare_rgb_slices():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_get_bvalue():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_get_min_max():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_get_coded_dictionary_elements():
    pass



@pytest.mark.skip(reason="Not implemented yet")
def test_exp_image():
    pass


def test_vprint():
    assert vprint("test") is None


def test_convert_array_to_min_max():
    array = [1, 2, 3, 4, 5]
    assert convert_array_to_min_max("Test Array", array) == [("TestArrayMin", 1), ("TestArrayMax", 5)]


@pytest.mark.skip(reason="Not implemented yet")
def test_convert_array_to_index_value():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_multiple_series_UID(mock_volumes):
    with pytest.raises(ValueError) as ex:
        DicomSingleVolumeInfoBase(mock_volumes[2])
    assert "Missing required echo time value" in str(ex.value)
    print(DicomSingleVolumeInfoBase(mock_volumes[2])._make_one_study_info_mapping_from_filelist())

@pytest.mark.skip(reason="Not implemented yet")
def test_no_derived_image(mock_volumes):
    assert check_for_diffusion_gradient(mock_volumes[2]) is False

@pytest.mark.skip(reason="Not implemented yet")
def test_missing_echo_time(mock_volumes):
    pass



dicom_file_dir: Path = current_file_path.parent.parent.parent / "testDcm" / "dcm_files"

def test_ADC_in_image_type():
    assert dicom_file_dir.exists()
    vol = list()
    i = 0
    for file in dicom_file_dir.iterdir():
        if i == 1:
            vol.append(file)
        i += 1

    assert check_for_diffusion_gradient(vol) is False


def test_invalid_fields():
    assert dicom_file_dir.exists()
    vol = list()
    for file in dicom_file_dir.iterdir():
        if "qyzp12" not in file.as_posix():
            vol.append(file)

    # Tests dicom file with an empty series number field
    f = pydicom.dcmread(vol[0])
    ds_dict = dict()
    with pytest.raises(TypeError) as ex:
        ds_dict = sanitize_dicom_dataset(f, required_DICOM_fields, optional_DICOM_fields)[0]
    assert "not 'NoneType'" in str(ex.value)

    # assert ds_dict["SeriesNumber"] == "INVALID_VALUE"

    # Tests dicom file with an empty echo time field
    f = pydicom.dcmread(vol[1])
    with pytest.raises(TypeError) as ex:
        ds_dict = sanitize_dicom_dataset(f, required_DICOM_fields, optional_DICOM_fields)[0]
    assert "not 'NoneType'" in str(ex.value)

    # assert ds_dict["EchoTime"] == "INVALID_VALUE"

    # Tests dicom file with an empty pixel bandwidth field
    f = pydicom.dcmread(vol[2])
    with pytest.raises(TypeError) as ex:
        ds_dict = sanitize_dicom_dataset(f, required_DICOM_fields, optional_DICOM_fields)[0]
    assert "not 'NoneType'" in str(ex.value)

    # assert ds_dict["PixelBandwidth"] == "INVALID_VALUE"


