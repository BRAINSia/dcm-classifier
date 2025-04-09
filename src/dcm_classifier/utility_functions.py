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
import json
import shutil
from collections import OrderedDict as ordered_dict

from pathlib import Path
from typing import Any
import collections

import pandas as pd
import pydicom
from copy import deepcopy
import itk
import warnings
import tempfile
import numpy as np
import numbers

import tabulate
from numpy import ndarray
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from .dicom_config import (
    inference_features as features,
    required_DICOM_fields,
    optional_DICOM_fields,
)
from datetime import datetime

from .image_type_inference import ImageTypeClassifierBase
from .study_processing import ProcessOneDicomStudyToVolumesMappingBase

FImageType = itk.Image[itk.F, 3]
UCImageType = itk.Image[itk.UC, 3]


def itk_read_from_dicomfn_list(
    single_volume_dcm_files_list: list[str | Path],
) -> FImageType:
    """
    Read DICOM files from a list and return an ITK image.

    :param single_volume_dcm_files_list: A list of DICOM file paths.
    :type single_volume_dcm_files_list: list[str | Path]

    :return: The ITK image created from the DICOM files with pixel type itk.F (float).
    :rtype: FImageType

    :raises FileNotFoundError: If no DICOM files are found in the input list.
    """
    with tempfile.TemporaryDirectory(
        prefix="all_dcm_for_volume_", suffix="_TMP"
    ) as tmpdir_symlink:
        tmp_symlink_dir_path: Path = Path(tmpdir_symlink)
        for dcm_file_path in [
            Path(dcm_file) for dcm_file in single_volume_dcm_files_list
        ]:
            new_dcm_file: Path = tmp_symlink_dir_path / dcm_file_path.name
            new_dcm_file.symlink_to(dcm_file_path)
        del single_volume_dcm_files_list

        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        # namesGenerator.AddSeriesRestriction("0008|0021")
        # namesGenerator.SetGlobalWarningDisplay(False)
        namesGenerator.SetDirectory(tmp_symlink_dir_path.as_posix())
        seriesUID_list = namesGenerator.GetSeriesUIDs()
        if len(seriesUID_list) < 1:
            raise FileNotFoundError(
                f"No DICOMs in: {tmp_symlink_dir_path.as_posix()} (itk_read_from_dicomfn_list)"
            )
        if len(seriesUID_list) > 1:
            msg: str = (
                f"Too many series in DICOMs in: {tmp_symlink_dir_path.as_posix()}"
            )
            raise AssertionError(msg)

        seriesIdentifier = seriesUID_list[0]
        ordered_filenames = namesGenerator.GetFileNames(seriesIdentifier)

        isr = itk.ImageSeriesReader[FImageType].New()
        # Typical clinical image slice spacing is > 1mm so a difference of 0.01 mm in slice spacing can be ignored.
        # to suppress warnings like 'Non uniform sampling or missing slices detected,  maximum nonuniformity:0.000480769'
        isr.SetSpacingWarningRelThreshold(0.01)
        isr.SetFileNames(ordered_filenames)
        isr.Update()
        itk_image = isr.GetOutput()
    return itk_image


def is_number(s: Any) -> bool:
    """
    Check if a string is a number.
    https://stackoverflow.com/q/354038

    :param s: The string to check.
    :type s: Any

    :return: True if the string is a number, False otherwise.
    :rtype: bool
    """
    try:
        float(s)
        return True
    except Exception:
        return False


def is_integer(s: Any) -> bool:
    """
    Check if a string is a number.
    https://stackoverflow.com/q/354038

    :param s: The string to check.
    :type s: Any

    :return: True if the string is an integer, False otherwise.
    :rtype: bool

    """
    try:
        int(s)
        return True
    except Exception:
        return False


def get_bvalue(dicom_header_info: Dataset, round_to_nearst_10: bool = True) -> float:
    """
    Extract and compute the b-value from DICOM header information.

    How to compute b-values

    http://clinical-mri.com/wp-content/uploads/software_hardware_updates/Graessner.pdf

    How to compute b-values difference of non-zero values

    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4610399/


    https://dicom.innolitics.com/ciods/enhanced-mr-image/enhanced-mr-image-multi-frame-functional-groups/52009229/00189117/00189087

    .. note::
        Bvalue is conditionally required.  This script is to re-inject implied values based on manual inspection or other data sources

    `dicom_header_info = dicom.dcmread(dicom_file_name, stop_before_pixels=True)`

    :param dicom_header_info: pydicom.Dataset object containing header information.
    :type dicom_header_info: pydicom.Dataset

    :param round_to_nearst_10: Whether to round the computed b-value to the nearest 10. (i.e. bvalues of 46,47,48,49,50,51,52,53,54,55 are reported as 50)
    :type round_to_nearst_10: bool, optional

    :return: The computed or extracted b-value.
    :rtype: float

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
    # TODO: https://pydicom.github.io/pydicom/dev/auto_examples/metadata_processing/plot_add_dict_entries.html
    # Add these private tags to the DICOM dictionary for better processing

    for k, v in private_tags_map.items():
        if v in dicom_header_info:
            # ensure safe extraction of the dicom element
            if isinstance(dicom_header_info[v], pydicom.dataelem.RawDataElement):
                dicom_element = pydicom.dataelem.DataElement_from_raw(
                    dicom_header_info[v]
                )
            else:
                dicom_element = dicom_header_info[v]

            # This decoding of bvalues follows the NAMIC conventions defined at
            # https://www.na-mic.org/wiki/NAMIC_Wiki:DTI:DICOM_for_DWI_and_DTI
            # TODO: add testing for this
            if v == private_tags_map["GE"]:
                large_number_modulo_for_GE = 100000
                # TODO: This is a hack to get around an error. Might be due to missing data in SickKids project
                try:
                    value = dicom_element.value[0] % large_number_modulo_for_GE
                except TypeError:
                    return -12345
            elif v == private_tags_map["Siemens_historical"]:
                # This is not supported yet
                continue
            else:
                value = dicom_element.value
            # TODO: add testing for this
            if dicom_element.VR == "OB":
                if isinstance(value, bytes):
                    value = value.decode("utf-8", "backslashreplace")
                elif isinstance(value, numbers.Number):
                    pass
                else:
                    print(
                        f"UNKNOWN CONVERSION OF VR={dicom_element.VR}: {type(dicom_element.value)} len={len(dicom_element.value)} ==> {value}"
                    )
                    return -12345
            # print(f"Found BValue at {v} for {k}, {value} of type {dicom_element.VR}")
            if value is None:
                # Field exists without value, USE DEFAULT
                return -12345
            try:
                result = float(value)
                if result > 5000:
                    return -12345
            except ValueError:
                print(
                    f"UNKNOWN CONVERSION OF VR={dicom_element.VR}: {type(dicom_element.value)} len={len(dicom_element.value)} ==> {dicom_element.value} to float"
                )
                return -12345
            except Exception as e:
                print(f"UNKNOWN IDENTIFICATION OF BVALUE: {e}")
                return -12345
            if round_to_nearst_10:
                result = round(result / 10.0) * 10
            return result
    return -12345


def ensure_magnitude_of_1(vector: np.ndarray) -> bool:
    """
    Ensure that the magnitude of a vector is 1.

    :param vector: numpy array containing the vector.
    :type vector: np.ndarray

    :return: True if the magnitude of the vector is 1, False otherwise.
    :rtype: bool
    """
    magnitude = np.linalg.norm(vector)
    if np.absolute(1 - magnitude) > 1e-2:
        return False
    return True


def get_diffusion_gradient_direction(
    dicom_header_info: pydicom.Dataset,
) -> np.ndarray | None:
    """
    Extract the diffusion gradient direction from DICOM header information.

    :param dicom_header_info: pydicom.Dataset object containing header information.
    :type dicom_header_info: pydicom.Dataset

    :return: numpy array containing the diffusion gradient direction.
    :rtype: np.ndarray | None
    """
    private_tags_map = collections.OrderedDict(
        {
            "Standard": (0x0018, 0x9076),
            "GE": (0x0019, 0x10E0),
            "Philips": (0x2001, 0x1004),
            "Siemens": (0x0019, 0x100E),
            "Siemens_historical": (0x0019, 0x000E),  # NOT SUPPORTED
            "Siemens_old": (0x0019, 0x000E),
            "UHI": (0x0065, 0x1037),
            # "Toshiba" : # Uses (0x0018, 0x9087) standard
        }
    )
    gradient_direction = None
    for k, v in private_tags_map.items():
        if v in dicom_header_info:
            # ensure safe extraction of the dicom element
            if isinstance(dicom_header_info[v], pydicom.dataelem.RawDataElement):
                gradient_direction_element = pydicom.dataelem.DataElement_from_raw(
                    dicom_header_info[v]
                )
            else:
                gradient_direction_element = dicom_header_info[v]
            if v == private_tags_map["Siemens"]:
                gradient_direction_raw = gradient_direction_element.value
                if (
                    gradient_direction_element.VR == "OB"
                    and len(gradient_direction_raw) == 24
                ):
                    gradient_direction = np.frombuffer(
                        gradient_direction_raw, dtype="double"
                    )
                else:
                    gradient_direction = np.array(gradient_direction_raw)

                break
            else:
                try:
                    gradient_direction = np.array(gradient_direction_element.value)
                    break
                except TypeError:
                    gradient_direction = None

    # ensure that the gradient direction is a 3D vector
    if gradient_direction is not None:
        gradient_direction_size = gradient_direction.size
        if gradient_direction_size != 3 or not ensure_magnitude_of_1(
            gradient_direction
        ):
            gradient_direction = None

    return gradient_direction


def infer_diffusion_from_gradient(filenames: list[Path]) -> str:
    """
    NAMIC Notes on DWI private fields:
    https://www.na-mic.org/wiki/NAMIC_Wiki:DTI:DICOM_for_DWI_and_DTI

    :param filenames: list of DICOM file names corresponding to a single volume.
    :type filenames: list[Path]

    :return: "dwig" if the volume contains non-zero diffusion gradient directions, "tracew" otherwise.
    :rtype: str

    """
    # check for derived data with constant diffusion gradient direction
    # this could happen when the header information is created based on one of the DWI files
    ds1 = pydicom.dcmread(filenames[0].as_posix(), stop_before_pixels=True)
    if (0x0008, 0x0008) not in ds1:
        return "MISSING_IMAGE_TYPE"

    gradient_direction_list = []
    for file in filenames:
        ds = pydicom.dcmread(file.as_posix(), stop_before_pixels=True)
        gradient_direction = get_diffusion_gradient_direction(ds)
        if gradient_direction is not None:
            gradient_direction_list.append(gradient_direction)

    unique_gradient_directions = np.unique(gradient_direction_list, axis=0)
    if len(unique_gradient_directions) >= 3:
        return "dwig"

    return "tracew"


def vprint(msg: str, verbose: bool = False) -> None:
    """
    Conditionally print a message if the 'verbose' flag is set.

    :param msg: The message to print.
    :type msg: str

    :param verbose: Whether to print the message. Default is False.
    :type verbose: bool, optional

    """
    if verbose:
        print(msg)


def sanitize_dicom_dataset(
    ro_dataset: pydicom.Dataset,
    required_info_list: list[str],
    optional_info_list: list[str],
) -> tuple[dict, bool]:
    """
    Validates the DICOM fields in the DICOM header to ensure all required fields are present.

    :param ro_dataset: The input DICOM dataset.
    :type ro_dataset: pydicom.Dataset
    :param required_info_list: A list of required DICOM fields.
    :type required_info_list: list[str]
    :param optional_info_list: A list of optional DICOM fields.
    :type optional_info_list: list[str]

    :return: A dictionary containing the sanitized DICOM fields and a boolean indicating whether the validation was successful.
    :rtype: tuple[dict, bool]

    :raises ValueError: Raises an exception if any required fields are missing.

    """
    dataset_dictionary: dict[str, Any] = dict()
    dataset = deepcopy(ro_dataset)  # DO NOT MODIFY THE INPUT DATASET!
    dicom_filename: Path = dataset.filename
    dataset_dictionary["FileName"]: str = dicom_filename
    dataset = pydicom.Dataset(dataset)  # DO NOT MODIFY THE INPUT DATASET!
    dataset.remove_private_tags()
    INVALID_VALUE = "INVALID_VALUE"

    # check if all fields in the required_info_list are present in dataset dictionary.
    # If fields are not present, or they are formatted incorrectly, add them with INVALID_VALUE
    missing_fields = []
    for field in required_info_list:
        if field not in dataset:
            # RepetitionTime and EchoTime might not be present in ADC images.
            # Therefore, if they are missing, set them to -12345
            if field == "RepetitionTime" or field == "EchoTime":
                dataset_dictionary[field] = -12345
            else:
                dataset_dictionary[field] = INVALID_VALUE
                missing_fields.append(field)
        elif field == "EchoTime" or field == "RepetitionTime":
            if dataset[field].value is None or str(dataset[field].value) == "":
                # ADC sequences may not have EchoTime
                dataset_dictionary[field] = -12345
            elif not is_number(dataset[field].value):
                dataset_dictionary[field] = INVALID_VALUE
                missing_fields.append(field)
                vprint(f"Missing required {field} value {dicom_filename}")
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "SeriesNumber":
            if not is_integer(dataset[field].value):
                dataset_dictionary[field] = INVALID_VALUE
                missing_fields.append(field)
                vprint(f"Missing required {field} value {dicom_filename}")
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "PixelBandwidth":
            if not is_number(dataset[field].value):
                dataset_dictionary[field] = INVALID_VALUE
                missing_fields.append(field)
                vprint(f"Missing required {field} value {dicom_filename}")
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "SliceThickness":
            if not is_number(dataset[field].value):
                dataset_dictionary[field] = INVALID_VALUE
                missing_fields.append(field)
                vprint(f"Missing required {field} value {dicom_filename}")
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "PixelSpacing":
            try:
                dataset_dictionary[field] = np.array(dataset[field].value)
            except Exception as e:
                dataset_dictionary[field] = INVALID_VALUE
                missing_fields.append(field)
                vprint(f"Missing required {field} value {dicom_filename}:\n{e}")
        else:
            # check that the field is not empty or None
            if dataset[field].value is None or str(dataset[field].value) == "":
                dataset_dictionary[field] = INVALID_VALUE
                missing_fields.append(field)
                vprint(f"Missing required field {dicom_filename}")
            else:
                dataset_dictionary[field] = dataset[field].value

    # set the default values for optional dicom fields
    for field in optional_info_list:
        DEFAULT_VALUE = -12345
        dataset_dictionary[field] = (
            DEFAULT_VALUE  # Every optional field will be set to the default -12345 and will be overriden if present and valid
        )
        try:
            dataset_value = dataset[field].value
        except Exception:
            dataset_value = DEFAULT_VALUE

        if field == "SAR":
            if field not in dataset or not is_number(dataset_value):
                # SAR is allowed to be empty or not a number because derived images often do not have SAR
                # for example, ADC images are derived images that are computed, so there is not
                # SAR impact on the patient for the derived image.
                # SAR Calculated whole body Specific Absorption Rate in watts/kilogram.
                # indicate that there is no SAR for the computed image
                _default_inferred_value = DEFAULT_VALUE
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = validate_numerical_dataset_element(
                    dataset_value
                )
        elif field == "Manufacturer":
            if field not in dataset or dataset_value is None:
                # Manufacturer is not a required field, it can be unknown
                _default_inferred_value = "UnknownManufacturer"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset_value
        elif field == "ImageType":
            if field not in dataset or dataset_value is None:
                # ImageType is not a required field, it can be unknown
                _default_inferred_value = "UnknownImageType"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset_value
        elif field == "ContrastBolusAgent":
            if field not in dataset or dataset_value is None:
                # The contrast is required but is empty if unknown but also may not be in every dataset
                dataset_dictionary[field] = DEFAULT_VALUE
            else:
                dataset_dictionary[field] = dataset_value
        elif field == "EchoNumbers":
            if field not in dataset:
                # EchoNumber(s) is not a required field, it can be unknown
                _default_inferred_value = DEFAULT_VALUE
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = validate_numerical_dataset_element(
                    dataset_value
                )
        elif field == "EchoTrainLength":
            if field not in dataset:
                # EchoTrainLength is not a required field, it can be unknown
                _default_inferred_value = DEFAULT_VALUE
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = validate_numerical_dataset_element(
                    dataset_value
                )
        elif field == "ScanningSequence":
            if field not in dataset or dataset_value is None:
                # ScanningSequence is not a required field, it can be unknown
                _default_inferred_value = "UnknownScanningSequence"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset_value
        elif field == "SequenceVariant":
            if field not in dataset or dataset_value is None:
                # SequenceVariant is not a required field, it can be unknown
                _default_inferred_value = "UnknownSequenceVariant"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset_value
        elif field == "InPlanePhaseEncodingDirection":
            if field not in dataset or dataset_value is None:
                # InplanePhaseEncodingDirection is not a required field, it can be unknown
                _default_inferred_value = "UnknownInplanePhaseEncodingDirection"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset_value
        elif field == "dBdt":
            if field not in dataset:
                # dBdt is not a required field, it can be unknown
                _default_inferred_value = DEFAULT_VALUE
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = validate_numerical_dataset_element(
                    dataset_value
                )
        elif field == "ImagingFrequency":
            if field not in dataset:
                # ImagingFrequency is not a required field, it can be unknown
                _default_inferred_value = DEFAULT_VALUE
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = validate_numerical_dataset_element(
                    dataset_value
                )
        elif field == "MRAcquisitionType":
            if field not in dataset or dataset_value is None:
                # MRAcquisitionType is not a required field, it can be unknown
                _default_inferred_value = "UnknownMRAcquisitionType"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset_value
        elif field == "NumberOfAverages":
            if field not in dataset:
                # NumberOfAverages is not a required field, it can be unknown
                _default_inferred_value = DEFAULT_VALUE
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = validate_numerical_dataset_element(
                    dataset_value
                )
        elif field == "InversionTime":
            if field not in dataset:
                # InversionTime is not a required field, it can be unknown
                _default_inferred_value = DEFAULT_VALUE
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = validate_numerical_dataset_element(
                    dataset_value
                )
        elif field == "VariableFlipAngleFlag":
            if field not in dataset or dataset_value is None:
                # VariableFlipAngleFlag is not a required field, it can be unknown
                _default_inferred_value = "UnknownVariableFlipAngleFlag"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset_value
        elif field == "AcquisitionTime":
            if field not in dataset:
                # AcquisitionTime is not a required field, it can be unknown
                _default_inferred_value = "000000.00"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = validate_numerical_dataset_element(
                    dataset_value
                )
        else:
            if field not in dataset:
                # field is not present, set to default value
                dataset_dictionary[field] = f"Unknown{field}"
            else:
                # field is present, set to value
                dataset_dictionary[field] = dataset_value

    # Warn the user if there are INVALID_VALUE fields
    if len(missing_fields) > 0:
        print(f"\n\n\n{missing_fields}\n\n\n")
        warnings.warn(
            f"\nWARNING: Required DICOM fields: {missing_fields} in {dicom_filename} are missing or have invalid values.\n"
        )
        return dataset_dictionary, False

    return dataset_dictionary, True


def validate_numerical_dataset_element(element: str | None) -> str | float:
    """
    Function to validate element can be converted to a float

    :param element: The element to validate.
    :type element: str

    :return: The element as a float if it can be converted, otherwise the element as a string.
    :rtype: str | float
    """
    try:
        float(element)
        return element
    except Exception:
        return -12345


# organize required features
image_type_features: list[str] = [
    field for field in features if ("ImageType_" in field and "ADC" not in field)
]
manufacturer_features: list[str] = [
    field for field in features if "Manufacturer_" in field
]
scanning_sequence_features: list[str] = [
    field for field in features if "ScanningSequence_" in field
]
scanning_variant_features: list[str] = [
    field for field in features if "SequenceVariant_" in field
]


def get_coded_dictionary_elements(
    dicom_sanitized_dataset: dict,
) -> dict[str, Any]:
    """
    Extract specific information from a DICOM fields dataset and create a coded dictionary with extracted features.

    :param dicom_sanitized_dataset: Sanitized dictionary containing required DICOM header information.
    :type dicom_sanitized_dataset: dict

    :return: A dictionary containing extracted information in a coded format.
    :rtype: dict

    """

    dataset_dictionary: dict[str, Any] = deepcopy(dicom_sanitized_dataset)
    for name, value in dicom_sanitized_dataset.items():
        if name == "PixelSpacing":
            if isinstance(value, np.ndarray):
                tuple_list = convert_array_to_index_value(name, value)
                for vv in tuple_list:
                    dataset_dictionary[vv[0]] = float(vv[1])
        elif value == "INVALID_VALUE":
            # Return a completely empty dictionary if any required values are missing from image
            # This should likely be more robust in the future
            return dict()
        elif name == "ImageType":
            lower_value_str: str = str(value).lower()
            if "adc" in lower_value_str:
                if "eadc" in lower_value_str:
                    dataset_dictionary["ImageType_EADC"] = 1
                    dataset_dictionary["ImageType_ADC"] = 0
                else:
                    dataset_dictionary["ImageType_ADC"] = 1
                    dataset_dictionary["ImageType_EADC"] = 0
            else:
                dataset_dictionary["ImageType_ADC"] = 0
                dataset_dictionary["ImageType_EADC"] = 0

            throw_away: int = len("ImageType_")
            for feature in image_type_features + ["ImageType_PERFUSION"]:
                # Note: this is a temporary fix to perfusion but does not include all cases
                # TODO: create a dataset containing perfusion data to retrain the model from scratch
                if feature[throw_away:].lower() in lower_value_str:
                    dataset_dictionary[feature] = 1
                else:
                    dataset_dictionary[feature] = 0
        elif name == "ImageOrientationPatient":
            tuple_list = convert_array_to_index_value(name, value)
            for vv in tuple_list:
                dataset_dictionary[vv[0]] = float(vv[1])
        elif name == "Manufacturer":
            lower_manufacturer_string: str = str(value).lower()
            throw_away: int = len("Manufacturer_")
            for feature in manufacturer_features:
                if feature[throw_away:].lower() in lower_manufacturer_string:
                    dataset_dictionary[feature] = 1
                else:
                    dataset_dictionary[feature] = 0
        elif name == "ScanningSequence":
            lower_scanning_sequence_string: str = str(value).lower()
            throw_away: int = len("ScanningSequence_")
            for feature in scanning_sequence_features:
                if feature[throw_away:].lower() in lower_scanning_sequence_string:
                    dataset_dictionary[feature] = 1
                else:
                    dataset_dictionary[feature] = 0
        elif name == "SequenceVariant":
            lower_sequence_variant_string: str = str(value).lower()
            throw_away: int = len("SequenceVariant_")
            for feature in scanning_variant_features:
                if feature[throw_away:].lower() in lower_sequence_variant_string:
                    dataset_dictionary[feature] = 1
                else:
                    dataset_dictionary[feature] = 0
        elif name == "ContrastBolusAgent":
            no_contrast_list = ["none", "no", "no contrast", "no_contrast", "n", ""]
            if str(value).lower() in no_contrast_list:
                dataset_dictionary["ContrastBolusAgent"] = "None"
            else:
                try:
                    dataset_dictionary["ContrastBolusAgent"] = str(value)
                except TypeError:
                    dataset_dictionary["ContrastBolusAgent"] = "INVALID_VALUE"

        else:
            dataset_dictionary[name] = str(value)

    return dataset_dictionary


def convert_array_to_min_max(name: str, value_list: list[int]) -> list:
    """
    Compute the minimum and maximum values of a DICOM array field.

    :param name: Original DICOM field name.
    :type name: str

    :param value_list: Original DICOM list values.
    :type value_list: list[int]

    :return: A list of (name, value) pairs representing the minimum and maximum values.
    :rtype: list

    """

    name = name.replace(" ", "")
    number_list = [float(x) for x in value_list]
    list_min = min(number_list)
    list_max = max(number_list)
    return [(name + "Min", list_min), (name + "Max", list_max)]


def convert_array_to_index_value(name: str, value_list: MultiValue | ndarray) -> list:
    """
    Takes a DICOM array and expands it to an indexed list of values.

    :param name: Original DICOM field name.
    :type name: str

    :param value_list: Original DICOM list values.
    :type value_list: MultiValue | ndarray

    :return: A list of (name, value) pairs representing the indexed values.
    :rtype: list
    """
    # If value is a multi-value field, then break it apart
    multi_value_list = value_list
    if isinstance(value_list, pydicom.multival.MultiValue):
        multi_value_list = tuple(value_list)
    else:
        # Some dicom have a string representation of the arrays
        if "/" in value_list:
            multi_value_list = value_list.split("/")
    del value_list

    name = name.replace(" ", "").replace("(", "").replace(")", "")
    # Note Absolute value as only the magnitude can have importance
    try:
        number_list = [abs(float(x)) for x in multi_value_list]
    except Exception as e:
        print(
            f"Failed conversion of {name} : {type(multi_value_list)} to float list {multi_value_list} "
        )
        raise e

    named_list = list()
    for index in range(0, len(number_list)):
        named_list.append((name + "_" + str(index), number_list[index]))
    return named_list


def check_two_images_have_same_physical_space(
    img1: FImageType, img2: FImageType
) -> bool:
    """
    Check if two images have the same physical space.

    :param img1: The first image.
    :type img1: FImageType
    :param img2: The second image.
    :type img2: FImageType
    :return: True if the images have the same physical space, False otherwise.
    :rtype: bool

    """
    same_size: bool = (
        img1.GetLargestPossibleRegion().GetSize()
        == img2.GetLargestPossibleRegion().GetSize()
    )
    same_space: bool = img1.GetSpacing() == img2.GetSpacing()
    same_origin: bool = img1.GetOrigin() == img2.GetOrigin()
    same_direction: bool = img1.GetDirection() == img2.GetDirection()

    return same_size and same_space and same_origin and same_direction


def parse_acquisition_datetime(ds: pydicom.Dataset) -> datetime:
    """
    Parse acquisition datetime with robust fallback strategies.

    Returns a consistent datetime object, using a fixed reference datetime
    when no valid datetime can be extracted.

    Args:
        ds (pydicom.Dataset): DICOM dataset

    Returns:
        datetime: Parsed or default datetime
    """
    # Reference datetime for sorting when no valid datetime is found
    # Dont love the double defaults but I like it better than having default values as strings
    DEFAULT_DATE = "19000101"  # YYYYMMDD
    DEFAULT_TIME = "000000"  # HHMMSS.FFFFFF
    DEFAULT_DATETIME = datetime(1900, 1, 1, 0, 0, 0)

    try:
        # Try AcquisitionDateTime first
        if hasattr(ds, "AcquisitionDateTime") and ds.AcquisitionDateTime:
            # https://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html#:~:text=A%20string%20of%20characters%20of,would%20represent%20August%2022%2C%201993.
            # A concatenated date-time character string in the format:
            #
            # YYYYMMDDHHMMSS.FFFFFF&ZZXX
            # FFFFFF = Fractional Second contains a fractional part of a second as small as 1 millionth of a second (range "000000" - "999999").
            #
            # &ZZXX is an optional suffix for offset from Coordinated Universal Time (UTC), where & = "+" or "-", and ZZ = Hours and XX = Minutes of offset.
            return datetime.strptime(str(ds.AcquisitionDateTime), "%Y%m%d%H%M%S")

        # Try combination of Date and Time
        date_str = (
            str(ds.AcquisitionDate) if hasattr(ds, "AcquisitionDate") else DEFAULT_DATE
        )
        time_str = (
            str(ds.AcquisitionTime) if hasattr(ds, "AcquisitionTime") else DEFAULT_TIME
        )

        # Pad or truncate to ensure correct format
        date_str = date_str.ljust(8, "0")[:8]
        time_str = time_str.ljust(6, "0")[:6]

        try:
            return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        except ValueError:
            return DEFAULT_DATETIME

    except Exception:
        return DEFAULT_DATETIME


def simple_classify_study(
    session_directory: Path | None = None,
    inferer: ImageTypeClassifierBase = ImageTypeClassifierBase(),
    json_dumppath: Path | None = None,
    output_dir: Path | None = None,
    nifti_dir: Path | None = None,
) -> dict[str, Any]:
    """

    Args:
        session_directory: Path to the patient session directory with dicom data
        inferer: Path to the model used for image type inference
        json_dumppath: Path to the output json file
        output_dir: Path to the output the newly organized dicom data or None to avoid organizing dicom data
        nifti_dir: Path to the directory where the NIFTI files are stored for each volume

    Returns:

    """
    if nifti_dir:
        import itk

        nifti_dir.mkdir(parents=True, exist_ok=True)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=session_directory, inferer=inferer
    )
    study.run_inference()
    list_of_inputs: list[dict[str, Any]] = []
    list_of_probabilities: list[pd.DataFrame] = []
    list_of_dictionaries: list[dict[str, str]] = []
    session_dictionary: dict[str, Any] = {}
    for series_number, series in study.series_dictionary.items():
        for index, volume in enumerate(series.get_volume_list()):
            invalid_volume: bool = False
            current_dict: dict[str, str] = ordered_dict()
            current_dict["Series#"] = str(series_number)
            dict_entry_name: str | None = None
            dictionary = {}

            # Volume Index
            try:
                vol_index: str = str(volume.get_volume_index())
                current_dict["Vol.#"] = vol_index
                dict_entry_name: str = f"{series_number}_{vol_index}"
            except AttributeError:
                current_dict["Vol.#"] = "None"

            # Volume Modality
            try:
                vol_modality: str = str(volume.get_volume_modality())
                current_dict["Volume Modality"] = vol_modality
                if vol_modality != "INVALID":
                    dictionary["VolumeModality"] = vol_modality
                else:
                    invalid_volume = True
            except AttributeError:
                current_dict["Volume Modality"] = "None"
                dictionary = {}

            # Series Modality
            try:
                series_modality: str = str(series.get_series_modality())
                current_dict["Series Modality"] = series_modality
                dictionary["SeriesModality"] = series_modality if dictionary else {}
            except AttributeError:
                current_dict["Series Modality"] = "None"
                dictionary = {}

            # Acquisition Plane
            try:
                acq_plane: str = str(volume.get_acquisition_plane())
                current_dict["Acq.Plane"] = acq_plane
                dictionary["AcqPlane"] = acq_plane if dictionary else {}
            except AttributeError:
                current_dict["Acq.Plane"] = "None"
                dictionary = {}

            # Isotropic
            try:
                isotropic: str = str(volume.get_is_isotropic())
                current_dict["Isotropic"] = isotropic
                dictionary["Isotropic"] = isotropic if dictionary else {}
            except AttributeError:
                current_dict["Isotropic"] = "None"
                dictionary["Isotropic"] = "None"

            # Modality Probabilities
            vol_probabilities = volume.get_modality_probabilities()
            for col in vol_probabilities.columns:
                # current_dict[col] = str(vol_probabilities[col].values[0])
                if "SeriesNumber" in col or "CODE" in col:
                    continue
                dictionary[col] = (
                    str(vol_probabilities[col].values[0]) if dictionary else {}
                )
            print(vol_probabilities.to_string(index=False))

            # Bvalue
            try:
                bval = str(volume.get_volume_bvalue())
                current_dict["Bvalue"] = bval
                dictionary["Bvalue"] = bval if dictionary else {}
            except AttributeError:
                current_dict["Bvalue"] = "None"
                dictionary["Bvalue"] = "None"

            # Series Description
            try:
                series_description: str = volume.get_dicom_field_by_name(
                    "SeriesDescription"
                )
                current_dict["SeriesDesc"] = series_description
                dictionary["SeriesDescription"] = (
                    series_description if dictionary else {}
                )
            except AttributeError:
                current_dict["SeriesDesc"] = "None"
                dictionary["SeriesDescription"] = "None"

            # Contrast Agent
            try:
                contrast = volume.get_has_contrast()
                dictionary["Contrast"] = str(contrast) if dictionary else {}

                if contrast:
                    contrast_agent = volume.get_contrast_agent()
                    # current_dict["ContrastAgent"] = contrast_agent
                    dictionary["ContrastAgent"] = contrast_agent if dictionary else {}
                else:
                    # current_dict["Contrast"] = "None"
                    dictionary["Contrast"] = "None"
            except AttributeError:
                # current_dict["Contrast"] = "None"
                dictionary["Contrast"] = "None"

            # Pixel Spacing
            try:
                pixel_spacing = volume.get_dicom_field_by_name("PixelSpacing")
                # current_dict["PixelSpacing"] = pixel_spacing
                dictionary["PixelSpacing_0"] = (
                    str(pixel_spacing[0]) if dictionary else {}
                )
                dictionary["PixelSpacing_1"] = (
                    str(pixel_spacing[1]) if dictionary else {}
                )
            except AttributeError:
                # current_dict["PixelSpacing"] = "None"
                dictionary["PixelSpacing_0"] = "None"
                dictionary["PixelSpacing_1"] = "None"

            for field in required_DICOM_fields + optional_DICOM_fields:
                if field in ["PixelSpacing", "ImageType"]:
                    continue
                try:
                    value = volume.get_dicom_field_by_name(field)
                    # current_dict[field] = str(value)
                    dictionary[field] = str(value) if dictionary else {}
                except AttributeError:
                    # current_dict[field] = "None"
                    dictionary[field] = "None"

            inputs_df: dict[str, Any] = volume.get_volume_dictionary()
            current_dict["ImageType"] = str(inputs_df.get("ImageType", "Unknown"))
            dictionary["ImageType"] = str(inputs_df.get("ImageType", "Unknown"))
            for unwanted in [
                "FileName",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "list_of_ordered_volume_files",
            ]:
                if unwanted in inputs_df:
                    inputs_df.pop(unwanted)

            list_of_inputs.append(inputs_df)

            prob_df: pd.DataFrame = volume.get_modality_probabilities()
            list_of_probabilities.append(prob_df)
            list_of_dictionaries.append(current_dict)
            if nifti_dir is not None:
                bvalue_suffix: str = (
                    f"_b{volume.get_volume_bvalue()}"
                    if volume.get_volume_bvalue() >= 0
                    else ""
                )
                image_file_name: str = (
                    f"{series_number:04}_{volume.get_volume_index():03}"
                    f"_{volume.get_series_modality()}{bvalue_suffix}.nii.gz"
                )
                itk_image = volume.get_itk_image()
                itk.imwrite(itk_image, nifti_dir / image_file_name)
            if output_dir is not None:
                bvalue_suffix: str = (
                    f"_b{volume.get_volume_bvalue()}"
                    if volume.get_volume_bvalue() >= 0
                    else ""
                )
                dcm_output_dir_name: str = (
                    f"{series_number:04}_{volume.get_volume_index():03}"
                    f"_{volume.get_series_modality()}{bvalue_suffix}"
                )
                output_dir_path: Path = output_dir / dcm_output_dir_name
                output_dir_path.mkdir(parents=True, exist_ok=True)
                for dcm_file in volume.one_volume_dcm_filenames:
                    output_file_path: Path = output_dir_path / dcm_file.name
                    print(f"Copying {dcm_file} to {output_file_path}")
                    shutil.copy(dcm_file, output_file_path, follow_symlinks=True)
                    # shutil.move(dcm_file, output_file_path)
            if dict_entry_name is not None:
                session_dictionary[dict_entry_name] = (
                    dictionary if not invalid_volume else {}  # set to empty if invalid
                )
    json_output_dict: dict[str, Any] = {str(session_directory): session_dictionary}
    # save the dictionary to a file
    if json_dumppath is not None:
        with open(json_dumppath, "w") as f:
            json.dump(json_output_dict, f, indent=4)
    df: pd.DataFrame = pd.DataFrame(list_of_dictionaries)
    df.sort_values(by=["Series#", "Vol.#"], inplace=True)
    table_msg: str = tabulate.tabulate(
        df, headers="keys", tablefmt="psql", showindex=False
    )
    print(table_msg)
    # all_inputs_df: pd.DataFrame = pd.DataFrame(list_of_inputs)
    # inputs_msg: str = tabulate.tabulate(
    #     all_inputs_df, headers="keys", tablefmt="psql", showindex=False
    # )
    # print(inputs_msg)
    #
    # all_prob_df: pd.DataFrame = pd.concat(list_of_probabilities, axis=0, ignore_index=True)
    # prob_msg: str = tabulate.tabulate(
    #     all_prob_df, headers="keys", tablefmt="psql", showindex=False
    # )
    # print(prob_msg)
    return json_output_dict
