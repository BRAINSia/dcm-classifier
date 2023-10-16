import itk
import pytest
from dcm_classifier.write_volume_from_dicom import (
    write_volume_from_series_directory,
)
from dcm_classifier.namic_dicom_typing import (
    two_point_compute_adc,
    compute_adc_from_multi_b_values,
    compare_3d_float_images,
    rglob_for_singular_result,
    itk_get_center_slice,
    slugify,
)

from pathlib import Path

relative_testing_data_path: Path = Path(__file__).parent.parent / "testing_data"


def test_two_point_compute_adc():
    different_b_value_volume_paths = [
        "sub-TEST0040_ses-01_run-01/DWI-seriesnum_007-index_00",
        "sub-TEST0040_ses-01_run-01/DWI-seriesnum_007-index_01",
    ]
    output_paths = [
        relative_testing_data_path / "dwi_volumes/test_volume0.nii.gz",
        relative_testing_data_path / "dwi_volumes/test_volume1.nii.gz",
    ]

    b_values = [0.0, 400.0]

    b_weighted_images = []

    for i in range(len(different_b_value_volume_paths)):
        assert not output_paths[i].exists()
        volume_path = relative_testing_data_path / different_b_value_volume_paths[i]
        write_volume_from_series_directory(str(volume_path), output_paths[i])
        assert output_paths[i].exists()
        volume_short_pixel_type = itk.imread(output_paths[i])
        volume_float_pixel_type = volume_short_pixel_type.astype(itk.F)
        b_weighted_images.append(volume_float_pixel_type)

    adc_image = two_point_compute_adc(b_weighted_images, b_values)
    assert type(adc_image) == itk.Image[itk.F, 3]

    for output_path in output_paths:
        output_path.unlink(missing_ok=True)


def test_compute_adc_from_mulit_b_values():
    different_b_value_volume_paths = [
        "sub-TEST0040_ses-01_run-01/DWI-seriesnum_007-index_00",
        "sub-TEST0040_ses-01_run-01/DWI-seriesnum_007-index_01",
        "sub-TEST0040_ses-01_run-01/DWI-seriesnum_007-index_02",
        "sub-TEST0040_ses-01_run-01/DWI-seriesnum_007-index_03",
    ]

    output_paths = [
        relative_testing_data_path / "dwi_volumes/test_volume0.nii.gz",
        relative_testing_data_path / "dwi_volumes/test_volume1.nii.gz",
        relative_testing_data_path / "dwi_volumes/test_volume2.nii.gz",
        relative_testing_data_path / "dwi_volumes/test_volume3.nii.gz",
    ]

    b_values = [0.0, 400.0, 50.0, 800.0]
    b_weighted_images = []

    for i in range(len(different_b_value_volume_paths)):
        assert not output_paths[i].exists()
        volume_path = relative_testing_data_path / different_b_value_volume_paths[i]
        write_volume_from_series_directory(str(volume_path), output_paths[i])
        assert output_paths[i].exists()
        volume_short_pixel_type = itk.imread(output_paths[i])
        volume_float_pixel_type = volume_short_pixel_type.astype(itk.F)
        b_weighted_images.append(volume_float_pixel_type)

    adc_image = compute_adc_from_multi_b_values(b_weighted_images, b_values)
    assert type(adc_image) == itk.Image[itk.F, 3]

    for output_path in output_paths:
        output_path.unlink(missing_ok=True)


def test_compare_3d_float_images():
    volume_path = "sub-TEST0040_ses-01_run-01/DWI-seriesnum_007-index_00"

    output_path = relative_testing_data_path / "dwi_volumes/test_volume0.nii.gz"

    assert not output_path.exists()
    full_volume_path = relative_testing_data_path / volume_path

    write_volume_from_series_directory(str(full_volume_path), output_path)
    assert output_path.exists()

    volume_short_pixel_type = itk.imread(output_path)
    volume_float_pixel_type = volume_short_pixel_type.astype(itk.F)
    itk.imwrite(volume_float_pixel_type, output_path)

    image_comparison = compare_3d_float_images(output_path, output_path, 0, 0, True)
    assert image_comparison[0] == 0
    assert image_comparison[2] == True

    output_path.unlink(missing_ok=True)


def test_compare_3d_float_images_sad_path():
    different_b_value_volume_paths = [
        "sub-TEST0040_ses-01_run-01/DWI-seriesnum_007-index_00",
        "sub-TEST0046_ses-01_run-01/DWI-seriesnum_006-index_00",
    ]

    output_paths = [
        relative_testing_data_path / "dwi_volumes/test_volume0.nii.gz",
        relative_testing_data_path / "dwi_volumes/test_volume1.nii.gz",
    ]

    for i in range(len(different_b_value_volume_paths)):
        assert not output_paths[i].exists()
        volume_path = relative_testing_data_path / different_b_value_volume_paths[i]
        write_volume_from_series_directory(str(volume_path), output_paths[i])
        assert output_paths[i].exists()
        volume_short_pixel_type = itk.imread(output_paths[i])
        volume_float_pixel_type = volume_short_pixel_type.astype(itk.F)
        itk.imwrite(volume_float_pixel_type, output_paths[i])

    image_comparison = compare_3d_float_images(
        output_paths[0], output_paths[1], 0, 0, False
    )
    assert image_comparison[0] != 0
    assert image_comparison[2] == False

    for output_path in output_paths:
        output_path.unlink(missing_ok=True)


def test_rglob_for_singular_result():
    single_image_dir = relative_testing_data_path / "sub-TEST0001/ses-01/anat"
    single_image = rglob_for_singular_result(single_image_dir, "*.nii.gz", "f")
    assert single_image is not None


def test_get_center_slice():
    output_path = relative_testing_data_path / "dwi_volumes/test_volume0.nii.gz"

    assert not output_path.exists()
    volume_path = "sub-TEST0040_ses-01_run-01/DWI-seriesnum_007-index_00"
    full_volume_path = relative_testing_data_path / volume_path

    write_volume_from_series_directory(str(full_volume_path), output_path)
    assert output_path.exists()

    volume_short_pixel_type = itk.imread(output_path)
    volume_float_pixel_type = volume_short_pixel_type.astype(itk.F)

    imageCalculatorFilter = itk.MinimumMaximumImageCalculator[itk.Image[itk.F, 3]].New()
    imageCalculatorFilter.SetImage(volume_float_pixel_type)
    imageCalculatorFilter.Compute()
    min = imageCalculatorFilter.GetMinimum()
    max = imageCalculatorFilter.GetMaximum()

    center_slice = itk_get_center_slice(volume_float_pixel_type, min, max)

    assert type(center_slice) == itk.Image[itk.UC, 3]
    output_path.unlink(missing_ok=True)


def test_slugify():
    test_strings = [
        "dwi series number 006 index 01",
        "DWI-series-number-006-index-01",
        "dwi-series-nùmber-006-index-01",
    ]
    results = []

    results.append(slugify(test_strings[0]))
    results.append(slugify(test_strings[1], allow_uppercase=True))
    results.append(slugify(test_strings[2], allow_unicode=True))

    assert results == [
        "dwi-series-number-006-index-01",
        "DWI-series-number-006-index-01",
        "dwi-series-nùmber-006-index-01",
    ]

@pytest.Mark.skip(reason="Not implemented yet")
def test_cmd_exists():
    pass

@pytest.Mark.skip(reason="Not implemented yet")
def test_compare_rgb_slices():
    pass

@pytest.Mark.skip(reason="Not implemented yet")
def test_get_bvalue():
    pass

@pytest.Mark.skip(reason="Not implemented yet")
def test_get_min_max():
    pass

@pytest.Mark.skip(reason="Not implemented yet")
def test_get_coded_dictionary_elements():
    pass

@pytest.Mark.skip(reason="Not implemented yet")
def test_exp_image():
    pass

@pytest.Mark.skip(reason="Not implemented yet")
def test_vprint():
    pass

@pytest.Mark.skip(reason="Not implemented yet")
def test_convert_array_to_min_max():
    pass

@pytest.Mark.skip(reason="Not implemented yet")
def test_convert_array_to_index_value():
    pass
