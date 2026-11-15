import importlib
import math
import sys

import itk
import numpy as np
import pytest

from dcm_classifier import image_processing
from dcm_classifier.image_processing import (
    compute_adc_from_multi_b_values,
    uniform_adc_scale_factor,
)

DIFFUSIVITY = 1.0e-3  # mm^2/s, typical of brain tissue
B_VALUES = [0.0, 500.0, 1000.0]


def _constant_image(value: float) -> itk.Image:
    array = np.full((3, 4, 5), value, dtype=np.float32)
    return itk.image_from_array(array)


def test_adc_of_monoexponential_decay():
    s0 = 1000.0
    images = [_constant_image(s0 * math.exp(-b * DIFFUSIVITY)) for b in B_VALUES]

    adc = itk.array_from_image(compute_adc_from_multi_b_values(images, B_VALUES))

    expected = DIFFUSIVITY * uniform_adc_scale_factor()
    np.testing.assert_allclose(adc, expected, rtol=1e-3)


def test_adc_is_clamped_at_zero():
    rising = [_constant_image(100.0 * math.exp(b * DIFFUSIVITY)) for b in B_VALUES]

    adc = itk.array_from_image(compute_adc_from_multi_b_values(rising, B_VALUES))

    assert adc.min() == 0.0


def test_example_module_is_a_deprecated_alias():
    sys.modules.pop("dcm_classifier.example_image_processing", None)
    with pytest.warns(DeprecationWarning, match=r"dcm_classifier\.image_processing"):
        example = importlib.import_module("dcm_classifier.example_image_processing")

    assert (
        example.compute_adc_from_multi_b_values
        is image_processing.compute_adc_from_multi_b_values
    )
