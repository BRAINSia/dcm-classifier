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

required_DICOM_fields: list[str] = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesNumber",
    "ImageOrientationPatient",
    "PixelBandwidth",
    "EchoTime",
    "RepetitionTime",
    "FlipAngle",
    "PixelSpacing",
    "SliceThickness",
]

optional_DICOM_fields: list[str] = [
    "SAR",  # Not present in all derived image types
    "ImageType",  # Not required
    "Manufacturer",  # Not required
    "Contrast/BolusAgent",  # "(0x0018, 0x0010)"
    "Diffusionb-value",  # Not required
    "Diffusionb-valueMax",  # Not required
    "Echo Number(s)",
    "Echo Train Length",
    "Scanning Sequence",
    "Sequence Variant",
    "In-plane Phase Encoding Direction",
    "dB/dt",
    "Imaging Frequency",
    "MR Acquisition Type",
    "Number of Averages",
    "Inversion Time",
    "Variable Flip Angle Flag",
]

inference_features: list[str] = [
    "Image Type_ORIGINAL",
    "Image Type_M",
    "Image Type_SE",
    "Image Type_ADC",
    "Image Type_UNSPECIFIED",
    "Manufacturer_siemens",
    "Manufacturer_ge",
    "Manufacturer_philips",
    "Manufacturer_toshiba",
    "Diffusionb-value",
    "Diffusionb-valueMax",
    "Echo Number(s)",
    "EchoTime",
    "Echo Train Length",
    "FlipAngle",
    "PixelBandwidth",
    "RepetitionTime",
    "SAR",
    "Scanning Sequence_SE",
    "Scanning Sequence_RM",
    "Sequence Variant_SK",
    "In-plane Phase Encoding Direction_COL",
    "In-plane Phase Encoding Direction_ROW",
    "dB/dt",
    "Imaging Frequency",
    "MR Acquisition Type_2D",
    "Number of Averages",
    "Image Type_DERIVED",
    "Inversion Time",
    "Sequence Variant_SP",
    "Sequence Variant_OSP",
    "Image Type_NORM",
    "Image Type_DIS2D",
    "Image Type_DIFFUSION",
    "Scanning Sequence_GR",
    "Scanning Sequence_EP",
    "MR Acquisition Type_3D",
    "Variable Flip Angle Flag_N",
    "Image Type_OTHER",
    "Sequence Variant_NONE",
    "Image Type_NONE",
    "Image Type_ND",
    "Image Type_2",
    "Image Type_EADC",
    "Scanning Sequence_IR",
    "Sequence Variant_SS",
    "Sequence Variant_MP",
    "Image Type_FFE",
    "Image Type_P",
    "Image Type_MOSAIC",
    "Image Type_FA",
    "Variable Flip Angle Flag_Y",
]
