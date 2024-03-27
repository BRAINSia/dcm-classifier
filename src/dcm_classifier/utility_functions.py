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
from typing import Any
import collections
import pydicom
from copy import deepcopy
import itk
import warnings
import tempfile
import numpy as np
import numbers
from numpy import ndarray
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from .dicom_config import inference_features as features


FImageType = itk.Image[itk.F, 3]
UCImageType = itk.Image[itk.UC, 3]


def itk_read_from_dicomfn_list(
    single_volume_dcm_files_list: list[str | Path],
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
    except Exception:
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
            try:
                result = float(value)
                if result > 5000:
                    return -12345
            except ValueError:
                print(
                    f"UNKNOWN CONVERSION OF VR={dicom_element.VR}: {type(dicom_element.value)} len={len(dicom_element.value)} ==> {dicom_element.value} to float"
                )
                return -12345
            if round_to_nearst_10:
                result = round(result / 10.0) * 10
            return result
    return -12345


def ensure_magnitude_of_1(vector: np.ndarray) -> bool:
    """
    Ensure that the magnitude of a vector is 1.

    Args:
        vector (np.ndarray): The input vector.

    Returns:
        bool: True if the magnitude of the vector is 1, False otherwise.
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
    Args:
        dicom_header_info: pydicom.Dataset object containing header information.

    Returns:
        numpy array containing the diffusion gradient direction.
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

    Args:
        filenames: list of DICOM file names corresponding to a single volume

    Returns:
        bool: True if the volume contains non-zero diffusion gradient directions

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

    Args:
        msg (str): The message to print.
        verbose (bool, optional): Whether to print the message. Default is False.
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

    Args:
        ro_dataset (pydicom.Dataset): The input DICOM dataset.
        required_info_list (list[str]): A list of required DICOM fields.
        optional_info_list (list[str]): A list of optional DICOM fields.

    Returns:
        tuple[dict, bool]: A dictionary containing the sanitized DICOM fields and a boolean indicating whether the validation was successful.

    Raises an exception if any required fields are missing.
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
        if field == "SAR":
            if field not in dataset or not is_number(dataset[field].value):
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
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "Manufacturer":
            if field not in dataset:
                # Manufacturer is not a required field, it can be unknown
                _default_inferred_value = "UnknownManufacturer"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "ImageType":
            if field not in dataset:
                # ImageType is not a required field, it can be unknown
                _default_inferred_value = "UnknownImageType"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "ContrastBolusAgent":
            if field not in dataset:
                # if (
                #     dataset_dictionary[field] is None
                #     or str(dataset_dictionary[field]) == ""
                # ):
                # The contrast is required but is empty if unknown but also may not be in every dataset
                dataset_dictionary[field] = "None"
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "EchoNumbers":
            if field not in dataset:
                # EchoNumber(s) is not a required field, it can be unknown
                _default_inferred_value = -12345
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "EchoTrainLength":
            if field not in dataset:
                # EchoTrainLength is not a required field, it can be unknown
                _default_inferred_value = -12345
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "ScanningSequence":
            if field not in dataset:
                # ScanningSequence is not a required field, it can be unknown
                _default_inferred_value = "UnknownScanningSequence"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "SequenceVariant":
            if field not in dataset:
                # SequenceVariant is not a required field, it can be unknown
                _default_inferred_value = "UnknownSequenceVariant"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "InPlanePhaseEncodingDirection":
            if field not in dataset:
                # InplanePhaseEncodingDirection is not a required field, it can be unknown
                _default_inferred_value = "UnknownInplanePhaseEncodingDirection"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "dBdt":
            if field not in dataset:
                # dBdt is not a required field, it can be unknown
                _default_inferred_value = -12345
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "ImagingFrequency":
            if field not in dataset:
                # ImagingFrequency is not a required field, it can be unknown
                _default_inferred_value = -12345
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "MRAcquisitionType":
            if field not in dataset:
                # MRAcquisitionType is not a required field, it can be unknown
                _default_inferred_value = "UnknownMRAcquisitionType"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "NumberOfAverages":
            if field not in dataset:
                # NumberOfAverages is not a required field, it can be unknown
                _default_inferred_value = -12345
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "InversionTime":
            if field not in dataset:
                # InversionTime is not a required field, it can be unknown
                _default_inferred_value = -12345
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "VariableFlipAngleFlag":
            if field not in dataset:
                # VariableFlipAngleFlag is not a required field, it can be unknown
                _default_inferred_value = "UnknownVariableFlipAngleFlag"
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value
        elif field == "AcquisitionTime":
            if field not in dataset:
                # AcquisitionTime is not a required field, it can be unknown
                _default_inferred_value = -12345
                dataset_dictionary[field] = _default_inferred_value
                vprint(
                    f"Inferring optional {field} value of '{_default_inferred_value}' for missing field in {dicom_filename}"
                )
            else:
                dataset_dictionary[field] = dataset[field].value

    # Warn the user if there are INVALID_VALUE fields
    if len(missing_fields) > 0:
        print(f"\n\n\n{missing_fields}\n\n\n")
        warnings.warn(
            f"\nWARNING: Required DICOM fields: {missing_fields} in {dicom_filename} are missing or have invalid values.\n"
        )
        return dataset_dictionary, False

    return dataset_dictionary, True


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

    Args:
        dicom_sanitized_dataset (dict): Sanitized dictionary containing required DICOM header information.

    Returns:
        Dict[str, Any]: A dictionary containing extracted information in a coded format.
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
            for feature in image_type_features:
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
            no_contrast_list = ["none", "no", "no contrast", "no_contrast", "n"]
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


def convert_array_to_index_value(name: str, value_list: MultiValue | ndarray) -> list:
    """
    Takes a DICOM array and expands it to an indexed list of values.

    Args:
        name: Original DICOM field name.
        value_list: Original DICOM list values.

    Returns:
        A list of (name, value) pairs, where each value in the original list is indexed with a unique name.
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
