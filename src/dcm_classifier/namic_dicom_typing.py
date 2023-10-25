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

from pathlib import Path
from typing import List, Dict, Any, Union
import collections
import pydicom
from copy import deepcopy
import itk
import warnings
import tempfile
import numpy as np


FImageType = itk.Image[itk.F, 3]
UCImageType = itk.Image[itk.UC, 3]


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


def get_bvalue(dicom_header_info, round_to_nearst_10=True) -> float:
    """
    Extract and compute the b-value from DICOM header information.

    How to compute b-values
    http://clinical-mri.com/wp-content/uploads/software_hardware_updates/Graessner.pdf

    How to compute b-values difference of non-zero values
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4610399/

    https://dicom.innolitics.com/ciods/enhanced-mr-image/enhanced-mr-image-multi-frame-functional-groups/52009229/00189117/00189087
    NOTE: Bvalue is conditionally required.  This script is to re-inject implied values based on manual inspection or other data sources

    `dicom_header_info = dicom.dcmread(dicom_file_name, stop_before_pixels=True)`

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
    # TODO: https://pydicom.github.io/pydicom/dev/auto_examples/metadata_processing/plot_add_dict_entries.html
    # Add these private tags to the DICOM dictionary for better processing

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
                # This is not supported yet
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


def check_for_diffusion_gradient(filenames: List[Path]) -> bool:
    """
    NAMIC Notes on DWI private fields:
    https://www.na-mic.org/wiki/NAMIC_Wiki:DTI:DICOM_for_DWI_and_DTI

    Args:
        filenames:

    Returns:

    """
    for file in filenames:
        ds = pydicom.dcmread(file.as_posix(), stop_before_pixels=True)
        image_type = ds[0x0008, 0x0008].value
        image_type_lower_str = str(image_type).lower()
        if "'TRACEW'".lower() in image_type_lower_str:
            return False
        # TODO: Adjust this for all manufacturers similarly to B-Value
        # Currently only supporting Siemens data
        try:
            gradient_direction_element = ds[0x0019, 0x100E]
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

            if np.linalg.norm(gradient_direction) > 1e-5:
                return True
        except KeyError:
            continue
    return False


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
    required_info_list: List[str],
    optional_info_list: List[str],
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
    all_candidate_info_fields: List[str] = required_info_list + optional_info_list

    for v in values:
        if isinstance(v, pydicom.dataelem.RawDataElement):
            e = pydicom.dataelem.DataElement_from_raw(v)
        else:
            e = v

        # process the name to match naming in required_info_list
        name = str(e.name).replace(" ", "").replace("(", "").replace(")", "")
        # Only add entities that are in the required or optional lists

        if name in all_candidate_info_fields:
            value = e.value
            dataset_dictionary[name] = value
    del all_candidate_info_fields

    # check if all fields in the required_info_list are present in dataset dictionary.
    # If fields are not present, or they are formatted incorrectly, add them with INVALID_VALUE
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
                vprint(f"Missing required SeriesNumber value {dicom_filename}")
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

    # set the default values for optional dicom fields
    for field in optional_info_list:
        if field == "SAR":
            if field not in dataset_dictionary.keys() or not is_number(
                dataset_dictionary[field]
            ):
                # SAR is allowed to be empty or not a number because derived images often do not have SAR
                # for example, ADC images are derived images that are computed, so there is not
                # SAR impact on the patient for the derived image.
                # SAR Calculated whole body Specific Absorption Rate in watts/kilogram.
                # indicate that there is no SAR for the computed image
                _default_inferred_value = -12345.0
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
        elif field == "Manufacturer":
            if field not in dataset_dictionary.keys():
                # Manufacturer is not a required field, it can be unknown
                _default_inferred_value = "UnknownManufacturer"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
        elif field == "ImageType":
            if field not in dataset_dictionary.keys():
                # ImageType is not a required field, it can be unknown
                _default_inferred_value = "UnknownImageType"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )

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
