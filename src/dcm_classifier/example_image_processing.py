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

"""Deprecated alias of :mod:`dcm_classifier.image_processing`, to be removed in 2.0.0."""

import warnings

from .image_processing import (
    add_const_to_itk_images,
    add_itk_images,
    add_list_of_images,
    compute_adc_from_multi_b_values,
    itk_clamp_image_filter,
    log_image,
    multiply_itk_images,
    scaled_by_bvalue_images,
    sub_itk_images,
    uniform_adc_scale_factor,
)

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

warnings.warn(
    "dcm_classifier.example_image_processing is deprecated and will be removed in "
    "2.0.0; import from dcm_classifier.image_processing instead",
    DeprecationWarning,
    stacklevel=2,
)
