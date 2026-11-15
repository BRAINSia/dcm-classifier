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

"""Image arithmetic for diffusion MRI, including ADC from several b-values."""

import itk

from .utility_functions import FImageType

__all__ = [
    "add_const_to_itk_images",
    "add_itk_images",
    "add_list_of_images",
    "compute_adc_from_multi_b_values",
    "itk_clamp_image_filter",
    "log_image",
    "multiply_itk_images",
    "scaled_by_bvalue_images",
    "sub_itk_images",
    "uniform_adc_scale_factor",
]


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

    list_of_log_images: list[FImageType] = []
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
    scaled_sums: list[FImageType] = []
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

    scaled_by_bvalue_list: list[FImageType] = []
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
