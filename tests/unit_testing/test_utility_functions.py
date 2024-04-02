import numpy as np
import pydicom
import pytest
from dcm_classifier.example_image_processing import slugify, rglob_for_singular_result
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from dcm_classifier.utility_functions import (
    vprint,
    get_diffusion_gradient_direction,
    convert_array_to_min_max,
    sanitize_dicom_dataset,
    itk_read_from_dicomfn_list,
    is_integer,
    ensure_magnitude_of_1,
    convert_array_to_index_value,
    get_coded_dictionary_elements,
)
from dcm_classifier.dicom_config import required_DICOM_fields, optional_DICOM_fields
from pathlib import Path

relative_testing_data_path: Path = Path(__file__).parent.parent / "testing_data"
current_file_path: Path = Path(__file__).parent
inference_model_path = list(
    current_file_path.parent.parent.rglob("models/rf_classifier.onnx")
)[0]


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


def test_vprint(capsys):
    assert vprint("test", True) is None

    captured = capsys.readouterr()
    assert captured.out == "test\n"


def test_convert_array_to_min_max():
    array = [1, 2, 3, 4, 5]
    assert convert_array_to_min_max("Test Array", array) == [
        ("TestArrayMin", 1),
        ("TestArrayMax", 5),
    ]


@pytest.mark.skip(reason="Not implemented yet")
def test_convert_array_to_index_value():
    pass


dicom_file_dir: Path = (
    current_file_path.parent
    / "testing_data"
    / "anonymized_testing_data"
    / "invalid_data"
    / "invalid_fields"
)

mult_series_dir: Path = (
    current_file_path.parent
    / "testing_data"
    / "anonymized_testing_data"
    / "invalid_data"
    / "mult_series_uid"
)


def test_multiple_series_UID():
    assert mult_series_dir.exists()
    vol = list()
    for file in mult_series_dir.iterdir():
        vol.append(file)

    with pytest.raises(AssertionError) as ex:
        itk_read_from_dicomfn_list(vol)
    assert "Too many series in DICOMs in:" in str(ex.value)


def test_diff_grad_dir():
    assert dicom_file_dir.exists()
    file = list(dicom_file_dir.rglob("*.dcm"))[0]
    ds = pydicom.dcmread(file, stop_before_pixels=True)

    assert get_diffusion_gradient_direction(ds) is None


def test_unknown_in_image_type():
    assert dicom_file_dir.exists()
    vol = list()
    for file in dicom_file_dir.iterdir():
        if "noImgType" in file.stem:
            vol.append(file)

    f = pydicom.dcmread(vol[0])
    ds_dict = sanitize_dicom_dataset(f, required_DICOM_fields, optional_DICOM_fields)[0]
    assert ds_dict["ImageType"] == "UnknownImageType"


def test_no_series_number():
    assert dicom_file_dir.exists()
    vol = list()
    for file in dicom_file_dir.iterdir():
        if "noSeriesNum" in file.stem:
            vol.append(file)

    f = pydicom.dcmread(vol[0])
    ds_dict = sanitize_dicom_dataset(f, required_DICOM_fields, optional_DICOM_fields)[0]

    assert ds_dict["SeriesNumber"] == "INVALID_VALUE"


def test_no_echo_time():
    assert dicom_file_dir.exists()
    vol = list()
    for file in dicom_file_dir.iterdir():
        if "noEchoTime" in file.stem:
            vol.append(file)

    f = pydicom.dcmread(vol[0])
    ds_dict = sanitize_dicom_dataset(f, required_DICOM_fields, optional_DICOM_fields)[0]

    assert ds_dict["EchoTime"] == -12345


def test_no_pixel_bandwidth():
    assert dicom_file_dir.exists()
    vol = list()
    for file in dicom_file_dir.iterdir():
        if "noPixelBW" in file.stem:
            vol.append(file)

    f = pydicom.dcmread(vol[0])
    ds_dict = sanitize_dicom_dataset(f, required_DICOM_fields, optional_DICOM_fields)[0]

    assert ds_dict["PixelBandwidth"] == "INVALID_VALUE"


def test_invalid_fields():
    file_dir = dicom_file_dir.parent

    assert file_dir.exists()
    vol = list()
    for file in file_dir.iterdir():
        if "no_valid" in file.stem:
            vol.append(file)

    f = pydicom.dcmread(vol[0])
    ds_dict = sanitize_dicom_dataset(f, required_DICOM_fields, optional_DICOM_fields)[0]

    # Convert all values to strings
    ds_dict = {str(field): str(value) for field, value in ds_dict.items()}

    all_fields = required_DICOM_fields + optional_DICOM_fields
    all_fields = [
        field
        for field in all_fields
        if field
        not in [
            "Diffusionb-value",
            "Diffusionb-valueMax",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "FileName",
            "list_of_ordered_volume_files",
        ]
    ]

    # assert the fields that are in the dataset are set to one of these invalid values
    invalid_fields = ["INVALID_VALUE", "-12345", "Unknown", "-12345.0", "None", ""]
    for field in all_fields:
        print(field, ds_dict[field])
        if "unknown" in ds_dict[field].lower():
            assert True
        else:
            assert ds_dict[field] in invalid_fields


def test_run_inference_validate_features(get_data_dir):
    dwi_directory = get_data_dir.parent / "anonymized_dwi_series"
    assert dwi_directory.exists()

    # run study on anonymized data
    inferer = ImageTypeClassifierBase(
        classification_model_filename=inference_model_path
    )
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=dwi_directory, inferer=inferer
    )

    # remove the fields that are validated in the inference

    for series_num, series in study.series_dictionary.items():
        for vol in series.get_volume_list():
            del vol.__getattribute__("volume_info_dict")["ImageType"]
            del vol.__getattribute__("volume_info_dict")["SeriesNumber"]
            del vol.__getattribute__("volume_info_dict")["EchoTime"]
            del vol.__getattribute__("volume_info_dict")["PixelBandwidth"]

    study.run_inference()

    for series_num, series in study.series_dictionary.items():
        # print(
        #     ImageTypeClassifierBase.infer_modality(
        #         series.get_volume_list()[0].get_volume_dictionary()
        #     )
        # )
        for vol in series.get_volume_list():
            assert vol.get_volume_modality() == "INVALID"


# def test_invalid_fields():
#     assert dicom_file_dir.exists()
#     vol = list()
#     for file in dicom_file_dir.iterdir():
#         if "invalid" in file.stem:
#             vol.append(file)
#
#     # Tests dicom file with an empty series number field
#     f = pydicom.dcmread(vol[0])
#     ds_dict = None
#     with pytest.raises(TypeError) as ex:
#         ds_dict = sanitize_dicom_dataset(f, required_DICOM_fields, optional_DICOM_fields)[0]
#     assert "not 'NoneType'" in str(ex.value)
#
#     # Tests dicom file with an empty echo time field
#     f = pydicom.dcmread(vol[1])
#     with pytest.raises(TypeError) as ex:
#         ds_dict = sanitize_dicom_dataset(f, required_DICOM_fields, optional_DICOM_fields)[0]
#     assert "not 'NoneType'" in str(ex.value)
#
#     # Tests dicom file with an empty pixel bandwidth field
#     f = pydicom.dcmread(vol[2])
#     with pytest.raises(TypeError) as ex:
#         ds_dict = sanitize_dicom_dataset(f, required_DICOM_fields, optional_DICOM_fields)[0]
#     assert "not 'NoneType'" in str(ex.value)


def test_unit_vector():
    vec: np.ndarray = np.array([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)])

    assert ensure_magnitude_of_1(vec) is True

    vec = np.array([1, 1, 1])
    assert ensure_magnitude_of_1(vec) is False


def test_get_gradient_direction(get_data_dir):
    # testing no gradient, not dwi series
    no_grad_dir = get_data_dir / "1" / "DICOM"
    no_grad = list(no_grad_dir.rglob("*.dcm"))[0]
    ds = pydicom.dcmread(no_grad)
    assert get_diffusion_gradient_direction(ds) is None

    # testing gradient direction
    grad_dir = get_data_dir.parent / "anonymized_dwi_series" / "DICOM"
    grad = list(grad_dir.rglob("*.dcm"))[0]
    ds_w_grad = pydicom.dcmread(grad)

    # ensure each element in the diffusion gradient direction is equal to the value in the DICOM header
    for elem in zip(
        get_diffusion_gradient_direction(ds_w_grad), ds_w_grad[(0x0019, 0x100E)].value
    ):
        assert elem[0] == elem[1]


def test_is_integer():
    assert is_integer("1") is True
    assert is_integer("test") is False
    assert is_integer("1.0") is False


def test_conv_arr_to_index_val():
    arr = "yes//this//is//a//test"
    with pytest.raises(ValueError) as ex:
        convert_array_to_index_value("test", arr)
    assert "could not convert string to float" in str(ex.value)


def test_get_invalid_coded_dictionary():
    test_dict = {"example_value": "INVALID_VALUE"}

    assert get_coded_dictionary_elements(test_dict) == {}


def test_get_EADC_image_type():
    test_dict = {"ImageType": ["ORIGINAL", "PRIMARY", "EADC", "NONE"]}

    assert get_coded_dictionary_elements(test_dict)["ImageType_EADC"] == 1
