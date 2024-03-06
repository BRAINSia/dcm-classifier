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
    "Diffusionb-valueBool",
    "Diffusionb-valueMax",
    "Echo Number(s)",
    "EchoTime",
    "Echo Train Length",
    "FlipAngle",
    "Image Type_ADC",
    "Image Type_DERIVED",
    "Image Type_DIFFUSION",
    "Image Type_EADC",
    "Image Type_FA",
    "Image Type_MOSAIC",
    "Image Type_ORIGINAL",
    "Image Type_PRIMARY",
    "Image Type_SECONDARY",
    "Image Type_TRACEW",
    "Imaging Frequency",
    "In-plane Phase Encoding Direction_COL",
    "In-plane Phase Encoding Direction_ROW",
    "Inversion Time",
    "MR Acquisition Type_2D",
    "MR Acquisition Type_3D",
    "Manufacturer_ge",
    "Manufacturer_philips",
    "Manufacturer_siemens",
    "Manufacturer_toshiba",
    "Number of Averages",
    "PixelBandwidth",
    "RepetitionTime",
    "SAR",
    "Scanning Sequence_EP",
    "Scanning Sequence_GR",
    "Scanning Sequence_IR",
    "Scanning Sequence_RM",
    "Scanning Sequence_SE",
    "Sequence Variant_MP",
    "Sequence Variant_NONE",
    "Sequence Variant_OSP",
    "Sequence Variant_SK",
    "Sequence Variant_SP",
    "Sequence Variant_SS",
    "Variable Flip Angle Flag_N",
    "Variable Flip Angle Flag_Y",
    "dB/dt",
    "HasDiffusionGradientOrientation",
]
