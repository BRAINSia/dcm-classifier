# =========================================================================
#
#    Copyright NumFOCUS
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#           https://www.apache.org/licenses/LICENSE-2.0.txt
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#  =========================================================================

import sys
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import collections
import pydicom
from copy import deepcopy
import itk
import re
import unicodedata
import warnings

from dcm_classifier.dicom_config import required_DICOM_fields


FImageType = itk.Image[itk.F, 3]
UCImageType = itk.Image[itk.UC, 3]


def two_point_compute_adc(
    list_of_raw_bimages: List[FImageType], list_of_bvalues: List[float]
) -> FImageType:
    """
    Compute ADC from the min and max b-value images (assumes list_of_images is sorted by ascending b-value).

    Args:
        list_of_raw_bimages (List[FImageType]): A list of b-weighted images.
        list_of_bvalues (List[float]): A list of b-values.

    Returns:
        FImageType: The computed itk ADC image with pixel type itk.F (float).
    """
    # ADC = - (ln(SI_b1) - ln(SI_b2) ) / (b_1 - b_2)
    # ADC = - (ln(SI_b1/SI_b2))/ (b_1 - b_2)

    small_offset_for_ln_computation: float = 1e-6
    SI_b1 = add_const_to_itk_images(
        list_of_raw_bimages[0], small_offset_for_ln_computation
    )
    SI_b2 = add_const_to_itk_images(
        list_of_raw_bimages[-1], small_offset_for_ln_computation
    )
    ratio_image = div_itk_images(SI_b1, SI_b2)
    ln_im = log_image(ratio_image)
    scale_factor: float = (
        -1.0 * uniform_adc_scale_factor() / (list_of_bvalues[0] - list_of_bvalues[-1])
    )
    ADC = multiply_itk_images(ln_im, scale_factor)
    return itk_clamp_image_filter(ADC, 0)


def compute_adc_from_multi_b_values(
    list_of_raw_bimages: List[FImageType], list_of_bvalues: List[float]
) -> FImageType:
    """
    Compute ADC from multiple b-value images.

    https://www.ajronline.org/doi/full/10.2214/AJR.15.15945?mobileUi=0

    Args:
        list_of_raw_bimages (List[FImageType]): A list of b-weighted images.
        list_of_bvalues (List[float]): A list of b-values.

    Returns:
        FImageType: The computed itk ADC image with pixel type itk.F (float).
    """

    list_of_log_images: List[FImageType] = list()
    small_offset_for_ln_computation: float = 1e-6
    for curr_raw_image in list_of_raw_bimages:
        non_zero_positive_shift: FImageType = add_const_to_itk_images(
            curr_raw_image, small_offset_for_ln_computation
        )
        log_im: FImageType = log_image(non_zero_positive_shift)
        list_of_log_images.append(log_im)
    del list_of_raw_bimages
    N: float = float(len(list_of_log_images))
    # Numerator computations
    scaled_by_bvalue: List[FImageType] = scaled_by_bvalue_images(
        list_of_log_images, list_of_bvalues
    )
    sum_of_scaled_images: FImageType = add_list_of_images(scaled_by_bvalue)
    n_time_sum_scaled_images: FImageType = multiply_itk_images(sum_of_scaled_images, N)

    sum_of_images: FImageType = add_list_of_images(list_of_log_images)
    scaled_sums: List[FImageType] = list()
    for current_bvalue in list_of_bvalues:
        scaled_sum = multiply_itk_images(sum_of_images, current_bvalue)
        scaled_sums.append(scaled_sum)
    scaled_sums_aggregate: FImageType = add_list_of_images(scaled_sums)
    numerator: FImageType = sub_itk_images(
        n_time_sum_scaled_images, scaled_sums_aggregate
    )
    # Denominator_computations
    sum_square_bvalue: float = 0.0
    sum_bvalue: float = 0.0
    for current_bvalue in list_of_bvalues:
        sum_bvalue += current_bvalue
        sum_square_bvalue += current_bvalue * current_bvalue
    denominator: float = N * sum_square_bvalue - sum_bvalue * sum_bvalue

    ADC: FImageType = multiply_itk_images(
        numerator, -1.0 * uniform_adc_scale_factor() / denominator
    )
    return itk_clamp_image_filter(ADC, 0)


def itk_read_from_dicomfn_list(
    single_volume_dcm_files_list: List[Union[str, Path]]
) -> FImageType:
    """
    Read DICOM files from a list and return an ITK image.

    Args:
        single_volume_dcm_files_list (List[Union[str, Path]]): A list of DICOM file paths.

    Returns:
        FImageType: The ITK image created from the DICOM files with pixel type itk.F (float).
    """
    import tempfile
    import shutil
    import os

    dir_path = Path(tempfile.mkdtemp(suffix="XXX"))
    shutil.rmtree(dir_path, ignore_errors=True)
    dir_path.mkdir(exist_ok=True, parents=True)

    # TODO: Files are sorted already,so don't resort the here
    for dcm_file in single_volume_dcm_files_list:
        dcm_file_path: Path = Path(dcm_file)
        new_dcm_file = dir_path / dcm_file_path.name
        os.symlink(dcm_file_path, new_dcm_file)

    del single_volume_dcm_files_list
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    # namesGenerator.AddSeriesRestriction("0008|0021")
    # namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(dir_path.as_posix())
    seriesUID_list = namesGenerator.GetSeriesUIDs()
    if len(seriesUID_list) < 1:
        print(f"No DICOMs in: {dir_path.as_posix()}")
        sys.exit(1)  # TODO, Throw exception
    if len(seriesUID_list) > 1:
        print(f"Too many series in DICOMs in: {dir_path.as_posix()}")
        sys.exit(2)  # TODO, Throw exception

    seriesIdentifier = seriesUID_list[0]
    ordered_filenames = namesGenerator.GetFileNames(seriesIdentifier)

    isr = itk.ImageSeriesReader[FImageType].New()
    # Typical clinical image slice spacing is > 1mm so a difference of 0.01 mm in slice spacing can be ignored.
    # to suppress warnings like 'Non uniform sampling or missing slices detected,  maximum nonuniformity:0.000480769'
    isr.SetSpacingWarningRelThreshold(0.01)
    isr.SetFileNames(ordered_filenames)
    isr.Update()
    itk_image = isr.GetOutput()
    shutil.rmtree(dir_path)
    return itk_image


def read_dwi_series_itk(dicom_directory: Path) -> (float, FImageType):
    """
    Given a directory of DICOM images of coherent DICOMs, read the volume and b-value.

    Args:
        dicom_directory (Path): The directory containing DICOM image files.

    Returns:
        Tuple[float, FImageType]: A tuple containing the extracted b-value and the ITK image with pixel type itk.F (float).
    """
    all_files: List[str] = [x.as_posix() for x in dicom_directory.glob("*.dcm")]
    bvalue_image = itk_read_from_dicomfn_list(all_files)

    dataset = pydicom.read_file(all_files[0], stop_before_pixels=True)
    dicom_extracted_bvalue = get_bvalue(dataset, round_to_nearst_10=True)
    return dicom_extracted_bvalue, bvalue_image


def cmd_exists(cmd):
    """
    Check if a command exists in the system's PATH.

    Args:
        cmd (str): The command to check for existence.

    Returns:
        bool: True if the command exists, False otherwise.
    """
    import shutil

    return shutil.which(cmd) is not None


def is_number(s: Any) -> bool:
    """
    Check if a string is a number.
    https://stackoverflow.com/q/354038

    Args:
        s (Any): The string to check.

    Returns:
        bool: True if the string is a number, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_integer(s: Any) -> bool:
    """
    Check if a string is a number.
    https://stackoverflow.com/q/354038

    Args:
        s (Any): The string to check.

    Returns:
        bool: True if the string is a number, False otherwise.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def rglob_for_singular_result(
    base_dir: Path,
    pattern: str,
    require_result_type: Optional[str] = None,
) -> Optional[Path]:
    """
    Recursively search for files or directories matching a pattern in a base directory.

    Args:
        base_dir (Path): The base directory to start the search from.
        pattern (str): The pattern to match against.
        require_result_type (Optional[str]): If specified, the type of result to require ("f" for file, "d" for directory).

    Returns:
        Optional[Path]: The matching result if found, or None if no result or multiple results are found.
    """

    list_results: List[Path] = [x for x in base_dir.rglob(pattern)]
    if (len(list_results)) != 1:
        return None
    singular_result = list_results[0]
    if require_result_type == "f" and not singular_result.is_file():
        return None
    if require_result_type == "d" and not singular_result.is_dir():
        return None
    return singular_result


def rglob_for_singular_result_from_pattern_list(
    base_dir: Path,
    patterns: List[str],
    require_result_type: Optional[str] = None,
) -> Optional[Path]:
    """
    Recursively search for files or directories matching patterns from a list in a base directory.

    Args:
        base_dir (Path): The base directory to start the search from.
        patterns (List[str]): The list of patterns to match against.
        require_result_type (Optional[str]): If specified, the type of result to require ("f" for file, "d" for directory).

    Returns:
        Optional[Path]: The matching result if found, or None if no result or multiple results are found.
    """
    for pattern in patterns:
        candidate = rglob_for_singular_result(base_dir, pattern, require_result_type)
        if candidate is not None:
            return candidate
    return None


def compare_RGB_slices(refr_slice_fn: Path, test_slice_fn: Path) -> (int, dict):
    """
    Compare two RGB image slices and count the number of pixels with differences.

    Args:
        refr_slice_fn (Path): Filepath to the reference RGB image slice.
        test_slice_fn (Path): Filepath to the test RGB image slice.

    Returns:
        Tuple[int, Dict[str, FImageType]]: A tuple containing the number of differing pixels
        and a dictionary containing the reference, test, and difference images.
    """

    images_dict: Dict[FImageType] = dict()
    refr_slice = itk.imread(filename=refr_slice_fn, pixel_type=itk.F)
    test_slice = itk.imread(filename=test_slice_fn, pixel_type=itk.F)
    cif = itk.ComparisonImageFilter[FImageType, FImageType].New(
        TestInput=refr_slice,
        ValidInput=test_slice,
        DifferenceThreshold=0.001,
        ToleranceRadius=1,
    )
    cif.SetDirectionTolerance(0.01)
    cif.SetCoordinateTolerance(0.01)
    cif.Update()
    diff_slice = cif.GetOutput()

    num_pixels_in_error: int = cif.GetNumberOfPixelsWithDifferences()
    images_dict["refr"] = refr_slice
    images_dict["test"] = test_slice
    images_dict["diff"] = diff_slice

    return (num_pixels_in_error, images_dict)


def compare_3d_float_images(
    refr_fn: Path,
    test_fn: Path,
    difference_threshold: float,
    tolerance_radius: float,
    force_exact_directions: bool,
) -> (int, dict, bool):
    """
    Compare two 3D float images and count the number of pixels with differences.

    Args:
        refr_fn (Path): Filepath to the reference image.
        test_fn (Path): Filepath to the test image.
        difference_threshold (float): Threshold for pixel differences.
        tolerance_radius (float): Tolerance radius for comparison.
        force_exact_directions (bool): Whether to force exact directions in comparison.

    Returns:
        Tuple[int, Dict[str, FImageType], bool]: A tuple containing the number of differing pixels,
        a dictionary containing the reference, test, and difference images, and a boolean indicating
        if the images are in the same space.
    """

    images_dict: Dict[FImageType] = dict()
    images_dict["refr"] = itk.imread(filename=refr_fn, pixel_type=itk.F)
    images_dict["test"] = itk.imread(filename=test_fn, pixel_type=itk.F)

    cif = itk.ComparisonImageFilter[FImageType, FImageType].New()
    cif.SetTestInput(images_dict["refr"])
    cif.SetDifferenceThreshold(difference_threshold)
    cif.SetToleranceRadius(tolerance_radius)
    cif.SetVerifyInputInformation(force_exact_directions)
    num_pixels_in_error: int
    images_in_same_space: bool = True
    if not force_exact_directions:
        cif.SetDirectionTolerance(0.1)
        cif.SetCoordinateTolerance(0.001)
    try:
        cif.SetValidInput(images_dict["test"])
        cif.Update()
        images_dict["diff"] = cif.GetOutput()
        num_pixels_in_error = cif.GetNumberOfPixelsWithDifferences()
    except Exception as e:
        images_in_same_space = False
        # If images are not in the same space, do resampling
        rif = itk.ResampleImageFilter[FImageType, FImageType].New()
        rif.SetInput(images_dict["test"])
        rif.SetOutputParametersFromImage(images_dict["refr"])
        rif.Update()
        cif.SetValidInput(rif.GetOutput())
        cif.Update()
        images_dict["diff"] = cif.GetOutput()
        num_pixels_in_error = cif.GetNumberOfPixelsWithDifferences()

    return num_pixels_in_error, images_dict, images_in_same_space


def slugify(value, allow_unicode=False):
    """
    Convert a string to a slug format.

    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.


    Args:
        value: The string to be converted.
        allow_unicode (bool): If True, allows Unicode characters in the slug.

    Returns:
        str: The slugified string.
    """

    "https://stackoverflow.com/a/295466"
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    final_value = re.sub(r"[-\s]+", "-", value).strip("-_")
    return final_value


def get_bvalue(dicom_header_info, round_to_nearst_10=True) -> float:
    """
    Extract and compute the b-value from DICOM header information.

    How to compute b-values
    http://clinical-mri.com/wp-content/uploads/software_hardware_updates/Graessner.pdf

    How to compute b-values difference of non-zero values
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4610399/

    https://dicom.innolitics.com/ciods/enhanced-mr-image/enhanced-mr-image-multi-frame-functional-groups/52009229/00189117/00189087
    NOTE: Bvalue is conditionally required.  This script is to re-inject implied values based on manual inspection or other data sources

    `dicom_header_info = dicom.read_file(dicom_file_name, stop_before_pixels=True)`

    Args:
        dicom_header_info: A pydicom object containing DICOM header information.
        round_to_nearst_10 (bool): Whether to round the computed b-value to the nearest 10. (i.e. bvalues of 46,47,48,49,50,51,52,53,54,55 are reported as 50)

    Returns:
        float: The computed or extracted b-value.
    """

    # bvalue tags from private fields provided by https://www.nmr.mgh.harvard.edu/~greve/dicom-unpack
    # Prefer searching in order, in case multiple bvalue dicom elements are duplicated.
    private_tags_map = collections.OrderedDict(
        {
            "Standard": (0x0018, 0x9087),  # Preferred tag (24,36999)
            "GE": (0x0043, 0x1039),  # (67,4153)
            "Philips": (0x2001, 0x1003),
            "Siemens": (0x0019, 0x100C),
            "Siemens_historical": (0x0029, 0x1010),  # NOT SUPPORTED
            "Siemens_old": (0x0019, 0x000C),
            "UHI": (0x0065, 0x1009),
            # "Toshiba" : # Uses (0x0018, 0x9087) standard
        }
    )

    for k, v in private_tags_map.items():
        if v in dicom_header_info:
            # This decoding of bvalues follows the NAMIC conventions defined at
            # https://www.na-mic.org/wiki/NAMIC_Wiki:DTI:DICOM_for_DWI_and_DTI
            dicom_element = dicom_header_info[v]
            if v == private_tags_map["GE"]:
                large_number_modulo_for_GE = 100000
                # value = dicom_element.value[0] % large_number_modulo_for_GE
                # TODO: This is a hack to get around an error. Might be due to missing data in SickKids project
                try:
                    value = dicom_element.value[0] % large_number_modulo_for_GE
                except TypeError:
                    return -12345
            elif v == private_tags_map["Siemens_historical"]:
                continue
            else:
                value = dicom_element.value
            if dicom_element.VR == "OB":
                value = value.decode("utf-8")
            # print(f"Found BValue at {v} for {k}, {value} of type {dicom_element.VR}")
            try:
                result = float(value)
            except ValueError:
                print(f"Could not convert {value} to float")
                return -12345
            if round_to_nearst_10:
                result = round(result / 10.0) * 10
            return result
    return -12345


def get_min_max(inputImage: FImageType) -> (float, float):
    """
    Calculate and return the minimum and maximum pixel values of the given image.

    Args:
        inputImage (FImageType): The input image for which the minimum and maximum pixel values are to be calculated.

    Returns:
        Tuple[float, float]: A tuple containing two floats: the minimum and maximum pixel values of the input image.
    """

    min_max_image_filter = itk.MinimumMaximumImageCalculator[FImageType].New()
    min_max_image_filter.SetImage(inputImage)
    min_max_image_filter.Compute()
    pixel_min = min_max_image_filter.GetMinimum()
    pixel_max = min_max_image_filter.GetMaximum()
    return pixel_min, pixel_max


def itk_get_center_slice(
    inputImage: FImageType, pixel_min: float, pixel_max: float
) -> UCImageType:
    """
    Extract the center slice of an input image and perform intensity windowing.

    Args:
        inputImage (FImageType): The input itk image with pixel type itk.F (float).
        pixel_min (float): Minimum pixel value.
        pixel_max (float): Maximum pixel value.

    Returns:
        UCImageType: The center slice image with adjusted intensity values.
    """
    extractFilter = itk.ExtractImageFilter[FImageType, FImageType].New()
    extractFilter.SetInput(inputImage)
    extractFilter.SetDirectionCollapseToSubmatrix()

    slice_axis: int = 2
    inputRegion = inputImage.GetBufferedRegion()
    desiredRegion = inputRegion

    # set up the extraction region [one slice]
    size = inputRegion.GetSize()
    sliceNumber = size[slice_axis] // 2
    size[slice_axis] = 1  # we extract center slice along z direction
    desiredRegion.SetSize(size)

    start = inputRegion.GetIndex()
    start[slice_axis] = start[slice_axis] + sliceNumber

    desiredRegion.SetIndex(start)
    print(f"XXXX  {slice_axis} SIZE: {size}, {size}, {start}, {sliceNumber}")

    extractFilter.SetExtractionRegion(desiredRegion)
    extractFilter.Update()
    image_rescale = itk.IntensityWindowingImageFilter[FImageType, UCImageType].New()
    image_rescale.SetInput(extractFilter.GetOutput())

    image_rescale.SetWindowMinimum(pixel_min)
    image_rescale.SetWindowMaximum(pixel_max)

    image_rescale.SetOutputMinimum(0)
    image_rescale.SetOutputMaximum(255)

    image_rescale.Update()

    return image_rescale.GetOutput()


def log_image(inimage: FImageType) -> FImageType:
    """
    Compute the natural logarithm of pixel values in the input image.

    Args:
        inimage (FImageType): The input image.

    Returns:
        FImageType: The image with pixel values transformed using the natural logarithm.
    """
    lif = itk.LogImageFilter[FImageType, FImageType].New()
    lif.SetInput(inimage)
    lif.Update()
    return lif.GetOutput()


def exp_image(inimage: FImageType) -> FImageType:
    """
    Compute the exponential of pixel values in the input image.

    Args:
        inimage (FImageType): The input image.

    Returns:
        FImageType: The image with pixel values transformed using the exponential function.
    """
    exp_image_filter = itk.ExpImageFilter[FImageType, FImageType].New()
    exp_image_filter.SetInput(inimage)
    exp_image_filter.Update()
    return exp_image_filter.GetOutput()


def add_itk_images(im1: FImageType, im2: FImageType) -> FImageType:
    """
    Add two input images element-wise.

    Args:
        im1 (FImageType): The first input image.
        im2 (FImageType): The second input image.

    Returns:
        FImageType: The resulting image from adding the two input images element-wise.
    """
    sum_image_filter = itk.AddImageFilter[FImageType, FImageType, FImageType].New()
    sum_image_filter.SetInput1(im1)
    sum_image_filter.SetInput2(im2)
    sum_image_filter.Update()
    return sum_image_filter.GetOutput()


def add_const_to_itk_images(im1: FImageType, offset: float) -> FImageType:
    """
    Add a constant offset to all pixel values in the input image.

    Args:
        im1 (FImageType): The input image.
        offset (float): The constant offset to be added to each pixel value.

    Returns:
        FImageType: The image with the constant offset added to its pixel values.
    """
    sum_image_filter = itk.AddImageFilter[FImageType, FImageType, FImageType].New()
    sum_image_filter.SetInput(im1)
    sum_image_filter.SetConstant(offset)
    sum_image_filter.Update()
    return sum_image_filter.GetOutput()


def sub_itk_images(im1: FImageType, im2: FImageType) -> FImageType:
    """
    Subtract pixel values of the second input image from the first input image element-wise.

    Args:
        im1 (FImageType): The first input image.
        im2 (FImageType): The second input image.

    Returns:
        FImageType: The resulting image from subtracting the pixel values of the second image from the first image.
    """

    sub_image_filter = itk.SubtractImageFilter[FImageType, FImageType, FImageType].New()
    sub_image_filter.SetInput1(im1)
    sub_image_filter.SetInput2(im2)
    sub_image_filter.Update()
    return sub_image_filter.GetOutput()


def div_itk_images(im1: FImageType, im2: FImageType) -> FImageType:
    """
    Divide pixel values of the first input image by the corresponding pixel values of the second input image element-wise.

    Args:
        im1 (FImageType): The first input image.
        im2 (FImageType): The second input image.

    Returns:
        FImageType: The resulting image from dividing the pixel values of the first image by the second image.
    """

    div_image_filter = itk.DivideImageFilter[FImageType, FImageType, FImageType].New()
    div_image_filter.SetInput1(im1)
    div_image_filter.SetInput2(im2)
    div_image_filter.Update()
    return div_image_filter.GetOutput()


def multiply_itk_images(im1: FImageType, scale: float) -> FImageType:
    """
    Multiply all pixel values in the input image by a constant scale.

    Args:
        im1 (FImageType): The input image.
        scale (float): The constant scale factor to multiply each pixel value.

    Returns:
        FImageType: The image with pixel values multiplied by the specified scale factor.
    """

    # TODO: Add inplace computations for speed
    mult_image_filter = itk.MultiplyImageFilter[
        FImageType, FImageType, FImageType
    ].New()
    mult_image_filter.SetInput(im1)
    mult_image_filter.SetConstant(scale)
    mult_image_filter.Update()
    return mult_image_filter.GetOutput()


def add_list_of_images(list_of_images: List[FImageType]) -> FImageType:
    """
    Sum the pixel values of a list of images element-wise.

    Args:
        list_of_images (List[FImageType]): A list of input images to be summed.

    Returns:
        FImageType: The resulting image after adding all the input images element-wise.
    """
    if len(list_of_images) == 1:
        return list_of_images[0]
    accumulator: FImageType = add_itk_images(list_of_images[0], list_of_images[1])
    for next_image in list_of_images[2:]:
        accumulator = add_itk_images(accumulator, next_image)
    return accumulator


def itk_clamp_image_filter(
    input_image: FImageType, lower_clamp: float, upper_clamp: float = 10**38
) -> FImageType:
    """
    Clamp pixel values of an input image within a specified range.

    Args:
        input_image (FImageType): The input image.
        lower_clamp (float): The lower bound for clamping pixel values.
        upper_clamp (float, optional): The upper bound for clamping pixel values. Default is 10^38.

    Returns:
        FImageType: The image with pixel values clamped within the specified range.
    """
    cif = itk.ClampImageFilter[FImageType, FImageType].New()
    cif.SetInput(input_image)
    cif.SetBounds(lower_clamp, upper_clamp)
    cif.Update()
    clipped_image: FImageType = cif.GetOutput()
    return clipped_image


def scaled_by_bvalue_images(
    list_of_images: List[FImageType], list_of_bvalues: List[float]
) -> List[FImageType]:
    """
    Multiply a list of images by their corresponding b-values element-wise.

    Args:
        list_of_images (List[FImageType]): A list of input images.
        list_of_bvalues (List[float]): A list of b-values corresponding to the input images.

    Returns:
        List[FImageType]: A list of images with pixel values scaled by their respective b-values.
    """

    scaled_by_bvalue_list: List[FImageType] = list()
    for index, image in enumerate(list_of_images):
        scaled_im: FImageType = multiply_itk_images(image, list_of_bvalues[index])
        scaled_by_bvalue_list.append(scaled_im)
    return scaled_by_bvalue_list


def uniform_adc_scale_factor() -> float:
    """
    Return the uniform scaling factor for b-value images.

    Returns:
        float: The scale factor, which is 10^6.
    """

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3998685/
    # For GE and Siemens platforms tested, stored PVs of derived ADC
    # maps appear to be uniformly scaled by 106. That is, a true diffusion
    # coefficient of 1 x 10-3 mm2/s is stored and read as “1000.”
    return 10**6


def vprint(msg: str, verbose=False):
    """
    Conditionally print a message if the 'verbose' flag is set.

    Args:
        msg (str): The message to print.
        verbose (bool, optional): Whether to print the message. Default is False.
    """
    if verbose:
        print(msg)


def sanitize_dicom_dataset(
    ro_dataset: pydicom.Dataset,
    required_info_list: List[str] = required_DICOM_fields,
) -> tuple[dict, bool]:
    """
    Validates the DICOM fields in the DICOM header to ensure all required fields are present.

    Raises an exception if any required fields are missing.

    """
    dataset_dictionary: Dict[str, Any] = dict()
    dataset = deepcopy(ro_dataset)  # DO NOT MODIFY THE INPUT DATASET!
    dicom_filename: Path = dataset.filename
    dataset_dictionary["FileName"]: str = dicom_filename
    dataset = pydicom.Dataset(dataset)  # DO NOT MODIFY THE INPUT DATASET!
    dataset.remove_private_tags()
    values = dataset.values()
    INVALID_VALUE = "INVALID_VALUE"
    for v in values:
        if isinstance(v, pydicom.dataelem.RawDataElement):
            e = pydicom.dataelem.DataElement_from_raw(v)
        else:
            e = v

        # process the name to match naming in required_DICOM_fields
        name = str(e.name).replace(" ", "").replace("(", "").replace(")", "")
        if name not in required_info_list:
            # No need to process columns that are not required
            continue
        else:
            value = e.value
            dataset_dictionary[name] = value

    # check if all fields in the required_DICOM_fields are present in dataset dictionary.
    # If fields are not present or they are formatted incorrectly, add them with INVALID_VALUE
    missing_fields = []
    for field in required_info_list:
        if field not in dataset_dictionary.keys():
            dataset_dictionary[field] = INVALID_VALUE
            missing_fields.append(field)
        elif field == "EchoTime":
            if not is_number(dataset_dictionary[field]):
                dataset_dictionary[field] = INVALID_VALUE
                missing_fields.append(field)
                vprint(f"Missing required echo time value {dicom_filename}")
        elif field == "SeriesNumber":
            if not is_integer(dataset_dictionary[field]):
                dataset_dictionary[field] = INVALID_VALUE
                missing_fields.append(field)
                vprint(f"Missing required echo time value {dicom_filename}")
        elif field == "SAR":
            if not is_number(dataset_dictionary[field]):
                dataset_dictionary[field] = INVALID_VALUE
                missing_fields.append(field)
                vprint(f"Missing required SAR value {dicom_filename}")
        elif field == "PixelBandwidth":
            if not is_number(dataset_dictionary[field]):
                dataset_dictionary[field] = INVALID_VALUE
                missing_fields.append(field)
                vprint(f"Missing required PixelBandwidth value {dicom_filename}")
        else:
            # check that the field is not empty or None
            if (
                dataset_dictionary[field] is None
                or str(dataset_dictionary[field]) == ""
            ):
                dataset_dictionary[field] = INVALID_VALUE
                missing_fields.append(field)
                vprint(f"Missing required field {dicom_filename}")

    # Warn the user if there are INVALID_VALUE fields
    if len(missing_fields) > 0:
        warnings.warn(
            f"\nWARNING: Required DICOM fields: {missing_fields} in {dicom_filename} are missing or have invalid values.\n"
        )
        return dataset_dictionary, False

    return dataset_dictionary, True


def get_coded_dictionary_elements(
    dicom_sanitized_dataset: dict,
) -> Dict[str, Any]:
    """
    Extract specific information from a DICOM fields dataset and create a coded dictionary with extracted features.

    Args:
        dicom_sanitized_dataset (dict): Sanitized dictionary containing required DICOM header information.

    Returns:
        Dict[str, Any]: A dictionary containing extracted information in a coded format.
    """
    dataset_dictionary: Dict[str, Any] = deepcopy(dicom_sanitized_dataset)
    for name, value in dicom_sanitized_dataset.items():
        if name == "PixelSpacing":
            tuple_list = convert_array_to_min_max(name, value)
            for vv in tuple_list:
                dataset_dictionary[vv[0]] = str(vv[1])
        elif name == "ImageType":
            lower_value_str: str = str(value).lower()
            dataset_dictionary["ImageType"] = str(value)
            if "'DERIVED'".lower() in lower_value_str:
                dataset_dictionary["IsDerivedImageType"] = 1
            else:
                dataset_dictionary["IsDerivedImageType"] = 0

            if ("'ADC'".lower() in lower_value_str) or (
                ("'SECONDARY'".lower() in lower_value_str)
                and ("'PROCESSED'".lower() in lower_value_str)
            ):
                dataset_dictionary["ImageTypeADC"] = 1
            else:
                dataset_dictionary["ImageTypeADC"] = 0
            if ("'FA'".lower() in lower_value_str) or (
                ("'SECONDARY'".lower() in lower_value_str)
                and ("'PROCESSED'".lower() in lower_value_str)
            ):
                dataset_dictionary["ImageTypeFA"] = 1
            else:
                dataset_dictionary["ImageTypeFA"] = 0
            if "'TRACEW'".lower() in lower_value_str:
                dataset_dictionary["ImageTypeTrace"] = 1
            else:
                dataset_dictionary["ImageTypeTrace"] = 0
        elif name == "ImageOrientationPatient":
            tuple_list = convert_array_to_index_value(name, value)
            for vv in tuple_list:
                dataset_dictionary[vv[0]] = str(vv[1])
        elif name == "Manufacturer":
            """
            GE MEDICAL SYSTEMS
            Philips Healthcare
            Philips Medical Systems
            SIEMENS
            """
            lower_manufactureer_string: str = str(value).lower()

            if "ge" in lower_manufactureer_string:
                manufacturer_code = 1
            elif "philips" in lower_manufactureer_string:
                manufacturer_code = 2
            elif "siemens" in lower_manufactureer_string:
                manufacturer_code = 3
            else:
                manufacturer_code = 0
            dataset_dictionary["ManufacturerCode"] = manufacturer_code
        else:
            dataset_dictionary[name] = str(value)
    return dataset_dictionary


def convert_array_to_min_max(name, value_list) -> list:
    """
    Compute the minimum and maximum values of a DICOM array field.

    Args:
        name: Original DICOM field name.
        value_list: Original DICOM list values.

    Returns:
        list: A list of (name, value) pairs representing the minimum and maximum values.
    """

    name = name.replace(" ", "")
    number_list = [float(x) for x in value_list]
    list_min = min(number_list)
    list_max = max(number_list)
    return [(name + "Min", list_min), (name + "Max", list_max)]


def convert_array_to_index_value(name, value_list) -> list:
    """
    Takes a DICOM array and expands it to an indexed list of values.

    Args:
        name: Original DICOM field name.
        value_list: Original DICOM list values.

    Returns:
        A list of (name, value) pairs, where each value in the original list is indexed with a unique name.
    """

    name = name.replace(" ", "").replace("(", "").replace(")", "")
    # Note Absolute value as only the magnitude can have importance
    number_list = [abs(float(x)) for x in value_list]
    named_list = list()
    for index in range(0, len(number_list)):
        named_list.append((name + "_" + str(index), number_list[index]))
    return named_list
