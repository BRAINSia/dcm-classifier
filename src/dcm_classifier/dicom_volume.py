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

import collections
from pathlib import Path

import pandas as pd
import json

from typing import Dict, List, Any, Union, Optional
import pydicom
from dcm_classifier.namic_dicom_typing import (
    get_bvalue,
    FImageType,
    itk_read_from_dicomfn_list,
    vprint,
    get_coded_dictionary_elements,
    sanitize_dicom_dataset,
)

pydicom_read_cache_static_filename_dict: Dict[str, pydicom.Dataset] = dict()


def pydicom_read_cache(
    filename: Union[Path, str], stop_before_pixels=True
) -> pydicom.Dataset:
    """
    Reads a DICOM file header and caches the result to improve performance on subsequent reads.

    Args:
        filename: The path to the DICOM file to be read.
        stop_before_pixels: If True, stops reading before pixel data (default: True).
    Returns:
        (pydicom.Dataset): A pydicom.Dataset containing the DICOM file's header data.
    """

    global pydicom_read_cache_static_filename_dict
    lookup_filename: str = str(filename)
    if lookup_filename in pydicom_read_cache_static_filename_dict:
        # print(f"Using cached value for {lookup_filename}")
        pass
    else:
        pydicom_read_cache_static_filename_dict[lookup_filename] = pydicom.read_file(
            lookup_filename, stop_before_pixels=stop_before_pixels
        )
    return pydicom_read_cache_static_filename_dict.get(lookup_filename)


def merge_dictionaries(rw_dict_to_update, ro_dict):
    """
    Merges two dictionaries, updating the first dictionary with the contents of the second.

    Args:
        rw_dict_to_update: The dictionary to be updated.
        ro_dict: The dictionary containing read-only data to be merged into the first dictionary.

    Returns:
        The updated first dictionary after merging.
    """
    for key in ro_dict.keys():
        rw_dict_to_update[key] = ro_dict[key]
    return rw_dict_to_update


class DicomSingleVolumeInfoBase:
    """
    This class is used to store information about a single DICOM volume.

    Attributes:
        one_volume_dcm_filenames (List[Path]): A list of DICOM file paths representing a single volume.
        ro_user_supplied_dcm_filenames (List[Path]): A list of DICOM file paths representing a single volume.
        _pydicom_info (pydicom.Dataset): A pydicom.Dataset containing information about the DICOM volume.
        bvalue (float): The b-value of the DICOM volume.
        average_slice_spacing (float): The average slice spacing of the DICOM volume.
        volume_info_dict (Dict[str, Any]): A dictionary containing information about the DICOM volume.
        itk_image (Optional[FImageType]): The ITK image of the DICOM volume.
        modality (Optional[str]): The modality of the DICOM volume (e.g., "CT", "MRI").
        modality_probability (Optional[pd.DataFrame]): A DataFrame containing modality probabilities.
        acquisition_plane (Optional[str]): The acquisition plane of the DICOM volume (e.g., "Sagittal", "Axial").

    Methods:
        set_modality(self, modality: str) -> None:

        get_modality(self) -> str:

        set_modality_probabilities(self, modality_probability: pd.DataFrame) -> None:

        get_modality_probabilities(self) -> pd.DataFrame:

        set_acquisition_plane(self, acquisition_plane: str) -> None:

        get_acquisition_plane(self) -> str:

        get_volume_info_dict(self) -> Dict[str, Any]:

        get_primary_volume_info(self, vol_index: int = 0) -> Dict[str, str]:

        get_itk_image(self) -> FImageType:

        get_series_uid(self) -> str:

        get_study_uid(self) -> str:

        get_series_pixel_spacing(self) -> str:

        get_series_spacing_between_slices(self) -> str:

        get_series_size(self) -> str:

        get_one_volume_dcm_filenames(self) -> List[Path]:

        get_volume_dictionary(self) -> Dict[str, Any]:

        get_volume_datatype(self) -> str:

        get_volume_bvalue(self) -> float:

        get_series_number(self) -> int:

        is_MR_modality(self):

        _make_one_study_info_mapping_from_filelist(self) -> (str, dict):

        get_image_diagnostics(self) -> str:
    """

    def __init__(self, one_volume_dcm_filenames: List[Union[Path, str]]) -> None:
        """
        Initializes a DicomSingleVolumeInfoBase instance with a list of DICOM file paths.

        Args:
            one_volume_dcm_filenames (List[Union[Path, str]]): A list of DICOM file paths representing a single volume.
        """
        self.one_volume_dcm_filenames: List[Path] = [
            Path(x).resolve() for x in one_volume_dcm_filenames
        ]
        # The ro_user_supplied_dcm_filenames should never be overriden after initialization
        # it is needed for repeated validation calls
        self.ro_user_supplied_dcm_filenames = list(self.one_volume_dcm_filenames)

        _first_filename_for_volume: Path = self.one_volume_dcm_filenames[0]
        # print(f"USING REFERENCE VOLUME:  {_first_filename_for_volume} for pydicom info")
        self._pydicom_info: pydicom.Dataset = pydicom_read_cache(
            _first_filename_for_volume, stop_before_pixels=True
        )

        self.bvalue = get_bvalue(self._pydicom_info, round_to_nearst_10=True)
        self.average_slice_spacing = -12345.0

        self.modality: Optional[str] = None
        self.modality_probability: Optional[pd.DataFrame] = None
        self.acquisition_plane: Optional[str] = None
        self.itk_image: Optional[FImageType] = None
        (
            _one_study_found,
            self.volume_info_dict,
        ) = self._make_one_study_info_mapping_from_filelist()

    def set_modality(self, modality: str) -> None:
        """
        Sets the modality of the DICOM data.

        Args:
            modality (str): The modality information to be set.
        """
        self.modality = modality

    def get_modality(self) -> str:
        """
        Retrieves the modality of the DICOM data.

        Returns:
            str: The modality information.
        """
        return self.modality

    def set_modality_probabilities(self, modality_probability: pd.DataFrame) -> None:
        """
        Sets the modality probabilities for the DICOM data.

        Args:
            modality_probability (pd.DataFrame): A pandas DataFrame containing modality probabilities.
        """
        self.modality_probability = modality_probability

    def get_modality_probabilities(self) -> pd.DataFrame:
        """
        Get the modality probabilities DataFrame that returns probability per modality class.

        Returns:
            pd.DataFrame: A pandas DataFrame containing modality probabilities.
        """
        return self.modality_probability

    def set_acquisition_plane(self, acquisition_plane: str) -> None:
        """
        Sets the acquisition plane information for the DICOM data.

        Args:
            acquisition_plane (str): The acquisition plane information to be set.
        """
        self.acquisition_plane = acquisition_plane

    def get_acquisition_plane(self) -> str:
        """
        Retrieves the acquisition plane information for the DICOM data.

        Returns:
            str: The acquisition plane information.
        """
        return self.acquisition_plane

    def get_volume_info_dict(self) -> Dict[str, Any]:
        """
        Retrieves a dictionary containing volume information for the DICOM data.

        Returns:
            Dict[str, Any]: A dictionary containing volume information.
        """
        return self.volume_info_dict

    def get_primary_volume_info(self, vol_index: int = 0) -> Dict[str, str]:
        """
        Get primary volume information for the specified volume index.

        Args:
            vol_index (int) Optional: Index of the volume for which to retrieve information, defaults to 0.

        Returns:
            Dict[str, str]: A dictionary containing primary volume information.
        """
        fields_to_copy: Dict[str, str] = {
            "SeriesNumber": "SeriesNum",
            "Diffusionb-value": "Bval",
            "RepetitionTime": "TR",
            "EchoTime": "TE",
            "FlipAngle": "FA",
            "SAR": "SAR",
        }

        ref_vol_info = self.get_volume_dictionary()
        return_dict: Dict[str, Union[str, int]] = collections.OrderedDict()
        return_dict["vol_index"] = vol_index
        for refkey, return_key in fields_to_copy.items():
            value = ref_vol_info.get(refkey, "")
            if str(value) == "-12345":
                value = ""

            if isinstance(value, float):
                value = f"{value:#.3f}"
            return_dict[return_key] = value
        return return_dict

    def get_itk_image(self) -> FImageType:
        """
        Get the ITK image associated with the DICOM volume.

        Returns:
            FImageType: The ITK image of the DICOM volume with pixel type itk.F (float).
        """
        if self.itk_image is None:
            self.itk_image = itk_read_from_dicomfn_list(
                self.get_one_volume_dcm_filenames()
            )

        return self.itk_image

    def get_series_uid(self) -> str:
        """
        Get the Series Instance UID of the DICOM volume.

        Returns:
            str: The Series Instance UID.
        """
        return self._pydicom_info.SeriesInstanceUID

    def get_study_uid(self) -> str:
        """
        Get the Study Instance UID of the DICOM volume.

        Returns:
            str: The Study Instance UID.
        """
        return self._pydicom_info.StudyInstanceUID

    def get_series_pixel_spacing(self) -> str:
        """
        Get the pixel spacing of the DICOM series.

        Returns:
            str: The pixel spacing as a string.
        """
        return str(self._pydicom_info.PixelSpacing)

    def get_series_spacing_between_slices(self) -> str:
        """
        Get the spacing between slices in the DICOM series.

        Returns:
            str: The spacing between slices as a string.
        """
        return str(self.average_slice_spacing)

    def get_series_size(self) -> str:
        """
        Get the size of the DICOM series.

        Returns:
            str: The size of the DICOM series as a string.
        """
        size_list: List[int] = [
            self._pydicom_info.Rows,
            self._pydicom_info.Columns,
            len(self.one_volume_dcm_filenames),
        ]
        return str(size_list)

    def get_one_volume_dcm_filenames(self) -> List[Path]:
        """
        Get the list of DICOM file paths for the single DICOM volume.

        Returns:
            List[Path]: A list of file paths for the DICOM files in the single volume.
        """
        return self.one_volume_dcm_filenames

    def get_volume_dictionary(self) -> Dict[str, Any]:
        """
        Get the dictionary containing information about the DICOM volume.

        Returns:
            Dict[str, Any]: A dictionary containing information about the DICOM volume.
        """
        return self.volume_info_dict

    def get_volume_bvalue(self) -> float:
        """
        Get the b-value of the DICOM volume.

        Returns:
            float: The b-value of the DICOM volume as a float.
        """
        return self.bvalue

    def get_series_number(self) -> int:
        """
        Get the Series Number of the DICOM volume.

        Returns:
            int: The Series Number as an integer.
        """
        return int(self._pydicom_info.SeriesNumber)

    def is_MR_modality(self):
        """
        Check if the modality of the DICOM volume is MR (Magnetic Resonance).

        Returns:
            status (bool): True if the modality is MR, False otherwise.
        """
        status = bool(self._pydicom_info.Modality != "MR")
        if not status:
            vprint(f"Skipping non-MR modality : {self._pydicom_info.Modality}")
        return status

    def _make_one_study_info_mapping_from_filelist(self) -> (str, dict):
        """
        Create a dictionary containing information about the DICOM volume from the list of DICOM files.

        Returns:
            Tuple[str, dict]: A tuple containing the Study Instance UID and a dictionary with volume information.
                 The dictionary includes Series Number, Echo Time, SAR, b-values, file name,
                 Series and Study Instance UID, Series Description, and various indicators.
        """
        sanitized_dicom_dict, valid = sanitize_dicom_dataset(self._pydicom_info)
        if not valid:
            self.set_modality("INVALID")
            self.set_acquisition_plane("INVALID")

        volume_info_dict: Dict[str, Any] = get_coded_dictionary_elements(
            sanitized_dicom_dict
        )
        del sanitized_dicom_dict

        # add features related to b-values and diffusion
        bvalue_current_dicom: int = int(self.get_volume_bvalue())
        volume_info_dict["Diffusionb-value"] = bvalue_current_dicom
        volume_info_dict["Diffusionb-valueMax"] = bvalue_current_dicom
        if bvalue_current_dicom < -1:
            volume_info_dict["HasDiffusionGradientOrientation"] = 0
        else:
            volume_info_dict["HasDiffusionGradientOrientation"] = 1

        # those values are 1 in case of a single volume
        volume_info_dict["Diffusionb-valueCount"] = 1
        volume_info_dict["SeriesVolumeCount"] = 1
        # add list of dicom files for the volume
        volume_info_dict["list_of_ordered_volume_files"] = self.one_volume_dcm_filenames

        return self.get_study_uid, volume_info_dict

    def get_image_diagnostics(self) -> str:
        """
        Generates diagnostic information about the DICOM image.

        Returns:
            msg (str): Diagnostic information as a formatted string.
        """
        volume_info: str = json.dumps(
            self.get_primary_volume_info(), indent=4, sort_keys=True
        )
        msg = f"""
    {'*' * 40}
    {volume_info}
    {'*' * 40}
    """
        return msg
