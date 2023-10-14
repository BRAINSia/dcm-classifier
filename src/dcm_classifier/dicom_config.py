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


"""
List of DICOM header columns that are required for this tool. In the future, the list can be expanded
or modified as needed.

Those fields were selected experimentally and were deemed useful for the modality and acquisition plane
classification task, or needed for data organization. Requiring only those fields allows for some flexibility in the
DICOM header and reduces the dimensionality of the data, thus improving processing efficiency.

Attributes:
    required_DICOM_fields (tuple): A tuple containing the names of
    DICOM header columns to be dropped from the dataset.

Example:
    Example usage to drop these columns from a DataFrame:
    df.drop(columns=required_DICOM_fields, inplace=True)
"""

required_DICOM_fields = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesNumber",
    "PixelSpacing",
    "ImageType",
    "ImageOrientationPatient",
    "Manufacturer",
    "PixelBandwidth",
    "SAR",
    "EchoTime",
    "RepetitionTime",
    "FlipAngle",
)
