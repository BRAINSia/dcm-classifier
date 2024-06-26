import re
import unicodedata
import itk
from pathlib import Path

import numpy as np
import pydicom
from .utility_functions import get_bvalue, itk_read_from_dicomfn_list
from .dicom_series import DicomSingleSeries


FImageType = itk.Image[itk.F, 3]
TracewImageType = itk.Image[itk.F, 4]
UCImageType = itk.Image[itk.UC, 3]


def compute_tracew_adc_from_diffusion(
    series: DicomSingleSeries, tracew_bval: int = 1000
) -> (FImageType, FImageType):
    """
    Compute the trace-weighted image from a diffusion series.

    :param series: The diffusion series.
    :type series: DicomSingleSeries
    :param tracew_bval: The preferred b-value for the trace-weighted image, defaults to 1000
    :type tracew_bval: int, optional

    :return: The computed trace-weighted image with pixel type itk.F (float).
    :rtype: FImageType
    """
    volume_list = series.get_volume_list()
    bval_volumes_dict = {}
    for volume in volume_list:
        bval = volume.get_volume_bvalue()
        dcm_files = volume.get_one_volume_dcm_filenames()
        itk_image = itk_read_from_dicomfn_list(dcm_files)
        if bval in bval_volumes_dict.keys():
            bval_volumes_dict[bval].append(itk_image)
        else:
            bval_volumes_dict[bval] = [itk_image]

    bval_list = sorted(list(bval_volumes_dict.keys()))
    # average the volumes with the same b-value
    bval_avg_vol_dict = {}
    for bval in bval_list:
        itk_im_list = bval_volumes_dict[bval]
        arr_list = [itk.GetArrayFromImage(itk_im) for itk_im in itk_im_list]
        avg_arr = np.average(np.array(arr_list), axis=0)
        # TODO: If this is a mozaic image as well, it might have mutliple slices that also need to be averaged
        # Seems to be multiple volumes per volume????
        # TODO: We should ideally check for mosaic and convert it to volume inside the itk_read_from_dicomfn_list function
        # TODO: Additionally, order of averaging (withing channels of subvolume and subvolumes) might matter
        if avg_arr.shape[0] < 10:
            avg_arr = np.average(avg_arr, axis=0, keepdims=True)
            print(f"XXXXX avg_avg_arr shape: {avg_arr.shape}")
        avg_vol = itk.GetImageFromArray(avg_arr)
        avg_vol.CopyInformation(itk_im_list[0])
        bval_avg_vol_dict[bval] = avg_vol

    # find bval closest to tracew_bval
    closest_bval = min(bval_list, key=lambda x: abs(x - tracew_bval))
    tracew = bval_avg_vol_dict[closest_bval]

    adc = compute_adc_from_multi_b_values(
        list(bval_avg_vol_dict.values()), list(bval_avg_vol_dict.keys())
    )
    return tracew, adc


def two_point_compute_adc(
    list_of_raw_bimages: list[FImageType], list_of_bvalues: list[float]
) -> FImageType:
    """
    Compute ADC from the min and max b-value images (assumes list_of_images is sorted by ascending b-value).

    :param list_of_raw_bimages: A list of b-weighted images.
    :type list_of_raw_bimages: list[FImageType]

    :param list_of_bvalues: A list of b-values.
    :type list_of_bvalues: list[float]

    :return: The computed itk ADC image with pixel type itk.F (float).
    :rtype: FImageType
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
    list_of_raw_bimages: list[FImageType], list_of_bvalues: list[float]
) -> FImageType:
    """
    Compute ADC from multiple b-value images.

    https://www.ajronline.org/doi/full/10.2214/AJR.15.15945?mobileUi=0

    :param list_of_raw_bimages: A list of b-weighted images.
    :type list_of_raw_bimages: list[FImageType]

    :param list_of_bvalues: A list of b-values.
    :type list_of_bvalues: list[float]

    :return: The computed itk ADC image with pixel type itk.F (float).
    :rtype: FImageType
    """

    list_of_log_images: list[FImageType] = list()
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
    scaled_by_bvalue: list[FImageType] = scaled_by_bvalue_images(
        list_of_log_images, list_of_bvalues
    )
    sum_of_scaled_images: FImageType = add_list_of_images(scaled_by_bvalue)
    n_time_sum_scaled_images: FImageType = multiply_itk_images(sum_of_scaled_images, N)

    sum_of_images: FImageType = add_list_of_images(list_of_log_images)
    scaled_sums: list[FImageType] = list()
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


def read_dwi_series_itk(dicom_directory: Path) -> (float, FImageType):
    """
    Given a directory of DICOM images of coherent DICOMs, read the volume and b-value.

    :param dicom_directory: The directory containing DICOM image files.
    :type dicom_directory: Path

    :return: A tuple containing the extracted b-value and the ITK image with pixel type itk.F (float).
    :rtype: Tuple[float, FImageType]
    """
    all_files: list[str] = [x.as_posix() for x in dicom_directory.glob("*.dcm")]
    bvalue_image = itk_read_from_dicomfn_list(all_files)

    dataset = pydicom.dcmread(all_files[0], stop_before_pixels=True)
    dicom_extracted_bvalue = get_bvalue(dataset, round_to_nearst_10=True)
    return dicom_extracted_bvalue, bvalue_image


def rglob_for_singular_result(
    base_dir: Path,
    pattern: str,
    require_result_type: str | None = None,
    recursive_search: bool = True,
) -> Path | None:
    """
    Recursively search for files or directories matching a pattern in a base directory.

    :param base_dir: The base directory to start the search from.
    :type base_dir: Path
    :param pattern: The pattern to match against.
    :type pattern: str
    :param require_result_type: If specified, the type of result to require ("f" for file, "d" for directory).
    :type require_result_type: str, optional

    :return: The matching result if found, or None if no result or multiple results are found.
    :rtype: Optional[Path]
    """

    glob_obj = base_dir.rglob(pattern) if recursive_search else base_dir.glob(pattern)
    list_results: list[Path] = list(glob_obj)
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
    patterns: list[str],
    require_result_type: str | None = None,
    recursive_search: bool = True,
) -> Path | None:
    """
    Recursively search for files or directories matching patterns from a list in a base directory.

    :param base_dir: The base directory to start the search from.
    :type base_dir: Path
    :param patterns: The list of patterns to match against.
    :type patterns: List[str]
    :param require_result_type: If specified, the type of result to require ("f" for file, "d" for directory).
    :type require_result_type: str, optional

    :return: The matching result if found, or None if no result or multiple results are found.
    :rtype: Optional[Path]

    """
    for pattern in patterns:
        candidate = rglob_for_singular_result(
            base_dir, pattern, require_result_type, recursive_search
        )
        if candidate is not None:
            return candidate
    return None


def cmd_exists(cmd):
    """
    Check if a command exists in the system's PATH.

    :param cmd: The command to check for existence.
    :type cmd: str

    :return: True if the command exists, False otherwise.
    :rtype: bool
    """
    import shutil

    return shutil.which(cmd) is not None


def compare_RGB_slices(refr_slice_fn: Path, test_slice_fn: Path) -> (int, dict):
    """
    Compare two RGB image slices and count the number of pixels with differences.

    :param refr_slice_fn: The reference RGB image slice.
    :type refr_slice_fn: Path
    :param test_slice_fn: The test RGB image slice.
    :type test_slice_fn: Path
    :return: A tuple containing the number of differing pixels and a dictionary containing the reference, test, and difference images.
    :rtype: Tuple[int, Dict[str, FImageType]]

    """

    images_dict: dict[FImageType] = dict()
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

    :param refr_fn: The reference image filename.
    :type refr_fn: Path
    :param test_fn: The test image filename.
    :type test_fn: Path
    :param difference_threshold: The threshold for pixel differences.
    :type difference_threshold: float
    :param tolerance_radius: The tolerance radius for comparison.
    :type tolerance_radius: float
    :param force_exact_directions: Whether to force exact directions in comparison.
    :type force_exact_directions: bool
    :return: A tuple containing the number of differing pixels, a dictionary containing the reference, test, and difference images, and a boolean indicating if the images are in the same space.
    :rtype: Tuple[int, Dict[str, FImageType], bool]

    """

    images_dict: dict[FImageType] = dict()
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
    except Exception:
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


def slugify(
    value: str, allow_uppercase: bool = False, allow_unicode: bool = False
) -> str:
    """
    Convert a string to a slug format.

    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.


    :param value: The string to be converted.
    :type value: str
    :param allow_uppercase: Whether to allow uppercase characters.
    :type allow_uppercase: bool
    :param allow_unicode: Whether to allow Unicode characters.
    :type allow_unicode: bool

    :return: The slugified string.
    :rtype: str
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
    if not allow_uppercase:
        value = value.lower()
    value = re.sub(r"[^\w\s-]", "", value)
    final_value = re.sub(r"[-\s]+", "-", value).strip("-_")
    return final_value


def get_min_max(inputImage: FImageType) -> (float, float):
    """
    Calculate and return the minimum and maximum pixel values of the given image.

    :param inputImage: The input image for which the minimum and maximum pixel values are to be calculated.
    :type inputImage: FImageType

    :return: A tuple containing two floats: the minimum and maximum pixel values of the input image.
    :rtype: Tuple[float, float]
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

    :param inputImage: The input image.
    :type inputImage: FImageType
    :param pixel_min: The minimum pixel value.
    :type pixel_min: float
    :param pixel_max: The maximum pixel value.
    :type pixel_max: float
    :return: The center slice image with adjusted intensity values.
    :rtype: UCImageType

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

    :param inimage: The input image.
    :type inimage: FImageType

    :return: The image with pixel values transformed using the natural logarithm.
    :rtype: FImageType

    """
    lif = itk.LogImageFilter[FImageType, FImageType].New()
    lif.SetInput(inimage)
    lif.Update()
    return lif.GetOutput()


def exp_image(inimage: FImageType) -> FImageType:
    """
    Compute the exponential of pixel values in the input image.

    :param inimage: The input image.
    :type inimage: FImageType
    :return: The image with pixel values transformed using the exponential function.
    :rtype: FImageType

    """
    exp_image_filter = itk.ExpImageFilter[FImageType, FImageType].New()
    exp_image_filter.SetInput(inimage)
    exp_image_filter.Update()
    return exp_image_filter.GetOutput()


def add_itk_images(im1: FImageType, im2: FImageType) -> FImageType:
    """
    Add two input images element-wise.

    :param im1: The first input image.
    :type im1: FImageType
    :param im2: The second input image.
    :type im2: FImageType

    :return: The image with pixel values added element-wise.
    :rtype: FImageType
    """
    sum_image_filter = itk.AddImageFilter[FImageType, FImageType, FImageType].New()
    sum_image_filter.SetInput1(im1)
    sum_image_filter.SetInput2(im2)
    sum_image_filter.Update()
    return sum_image_filter.GetOutput()


def add_const_to_itk_images(im1: FImageType, offset: float) -> FImageType:
    """
    Add a constant offset to all pixel values in the input image.

    :param im1: The input image.
    :type im1: FImageType
    :param offset: The constant offset to be added to each pixel value.
    :type offset: float

    :return: The image with the constant offset added to its pixel values.
    :rtype: FImageType
    """
    sum_image_filter = itk.AddImageFilter[FImageType, FImageType, FImageType].New()
    sum_image_filter.SetInput(im1)
    sum_image_filter.SetConstant(offset)
    sum_image_filter.Update()
    return sum_image_filter.GetOutput()


def sub_itk_images(im1: FImageType, im2: FImageType) -> FImageType:
    """
    Subtract pixel values of the second input image from the first input image element-wise.

    :param im1: The first input image.
    :type im1: FImageType
    :param im2: The second input image.
    :type im2: FImageType

    :return: The image with pixel values subtracted element-wise.
    :rtype: FImageType
    """

    sub_image_filter = itk.SubtractImageFilter[FImageType, FImageType, FImageType].New()
    sub_image_filter.SetInput1(im1)
    sub_image_filter.SetInput2(im2)
    sub_image_filter.Update()
    return sub_image_filter.GetOutput()


def div_itk_images(im1: FImageType, im2: FImageType) -> FImageType:
    """
    Divide pixel values of the first input image by the corresponding pixel values of the second input image element-wise.

    :param im1: The first input image.
    :type im1: FImageType
    :param im2: The second input image.
    :type im2: FImageType

    :return: The image with pixel values divided element-wise.
    :rtype: FImageType
    """

    div_image_filter = itk.DivideImageFilter[FImageType, FImageType, FImageType].New()
    div_image_filter.SetInput1(im1)
    div_image_filter.SetInput2(im2)
    div_image_filter.Update()
    return div_image_filter.GetOutput()


def multiply_itk_images(im1: FImageType, scale: float) -> FImageType:
    """
    Multiply all pixel values in the input image by a constant scale.

    :param im1: The input image.
    :type im1: FImageType
    :param scale: The constant scale factor to multiply each pixel value.
    :type scale: float

    :return: The image with pixel values multiplied by the specified scale factor.
    :rtype: FImageType
    """

    # TODO: Add inplace computations for speed
    mult_image_filter = itk.MultiplyImageFilter[
        FImageType, FImageType, FImageType
    ].New()
    mult_image_filter.SetInput(im1)
    mult_image_filter.SetConstant(scale)
    mult_image_filter.Update()
    return mult_image_filter.GetOutput()


def add_list_of_images(list_of_images: list[FImageType]) -> FImageType:
    """
    Sum the pixel values of a list of images element-wise.

    :param list_of_images: The list of input images to be summed.
    :type list_of_images: List[FImageType]

    :return: The image with pixel values summed element-wise.
    :rtype: FImageType
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

    :param input_image: The input image.
    :type input_image: FImageType
    :param lower_clamp: The lower bound for clamping pixel values.
    :type lower_clamp: float
    :param upper_clamp: The upper bound for clamping pixel values, defaults to 10**38
    :type upper_clamp: float, optional

    :return: The image with pixel values clamped within the specified range.
    :rtype: FImageType
    """
    cif = itk.ClampImageFilter[FImageType, FImageType].New()
    cif.SetInput(input_image)
    cif.SetBounds(lower_clamp, upper_clamp)
    cif.Update()
    clipped_image: FImageType = cif.GetOutput()
    return clipped_image


def scaled_by_bvalue_images(
    list_of_images: list[FImageType], list_of_bvalues: list[float]
) -> list[FImageType]:
    """
    Multiply a list of images by their corresponding b-values element-wise.

    :param list_of_images: The list of input images.
    :type list_of_images: List[FImageType]
    :param list_of_bvalues: The list of b-values corresponding to the input images.
    :type list_of_bvalues: List[float]

    :return: The list of images with pixel values scaled by their respective b-values.
    :rtype: List[FImageType]

    """

    scaled_by_bvalue_list: list[FImageType] = list()
    for index, image in enumerate(list_of_images):
        scaled_im: FImageType = multiply_itk_images(image, list_of_bvalues[index])
        scaled_by_bvalue_list.append(scaled_im)
    return scaled_by_bvalue_list


def uniform_adc_scale_factor() -> float:
    """
    Return the uniform scaling factor for b-value images.

    :return: The uniform scaling factor for b-value images.
    :rtype: float
    """

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3998685/
    # For GE and Siemens platforms tested, stored PVs of derived ADC
    # maps appear to be uniformly scaled by 106. That is, a true diffusion
    # coefficient of 1 x 10-3 mm2/s is stored and read as “1000.”
    return 10**6
