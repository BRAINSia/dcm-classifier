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

from .image_processing import *
from .image_processing import __all__  # noqa: F401

warnings.warn(
    "dcm_classifier.example_image_processing is deprecated and will be removed in "
    "2.0.0; import from dcm_classifier.image_processing instead",
    DeprecationWarning,
    stacklevel=2,
)
