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
from copy import deepcopy

from typing import Any, List, Dict, Optional
import pydicom
from .utility_functions import (
    get_bvalue,
    FImageType,
    itk_read_from_dicomfn_list,
    vprint,
    get_coded_dictionary_elements,
    sanitize_dicom_dataset,
    get_diffusion_gradient_direction,
)
from .dicom_config import (
    required_DICOM_fields,
    optional_DICOM_fields,
)


class DicomSingleVolumeInfoBase:
    """
    This class is used to store information about a single DICOM volume.

    Attributes:
        one_volume_dcm_filenames (List[Path]): A list of DICOM file paths representing a single volume.
        ro_user_supplied_dcm_filenames (List[Path]): A list of DICOM file paths representing a single volume.
        _pydicom_info (pydicom.Dataset): A pydicom.Dataset containing information about the DICOM volume.
        bvalue (float): The b-value of the DICOM volume.
        volume_info_dict (Dict[str, Any]): A dictionary containing information about the DICOM volume.
        itk_image (Optional[FImageType]): The ITK image of the DICOM volume.
        volume_modality (Optional[str]): The modality of the DICOM volume (e.g., "CT", "MRI").
        modality_probability (Optional[pd.DataFrame]): A DataFrame containing modality probabilities.
        acquisition_plane (Optional[str]): The acquisition plane of the DICOM volume (e.g., "Sagittal", "Axial").

    Methods:
        set_modality(self, modality: str) -> None:

        get_modality(self) -> str:

        set_modality_probabilities(self, modality_probability: pd.DataFrame) -> None:

        get_modality_probabilities(self) -> pd.DataFrame:

        set_acquisition_plane(self, acquisition_plane: str) -> None:

        get_acquisition_plane(self) -> str:

        get_volume_dictionary(self) -> Dict[str, Any]:

        get_primary_volume_info(self, vol_index: int = 0) -> Dict[str, str]:

        get_itk_image(self) -> FImageType:

        get_series_uid(self) -> str:

        get_study_uid(self) -> str:

        get_series_pixel_spacing(self) -> str:

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

    def __init__(self, one_volume_dcm_filenames: list[Path | str]) -> None:
        """
        Initializes a DicomSingleVolumeInfoBase instance with a list of DICOM file paths.

        Args:
            one_volume_dcm_filenames (List[Union[Path, str]]): A list of DICOM file paths representing a single volume.
        """
        self.one_volume_dcm_filenames: list[Path] = [
            Path(x).resolve() for x in one_volume_dcm_filenames
        ]
        if len(self.one_volume_dcm_filenames) == 0:
            raise ValueError("No file names provided list")

        # The ro_user_supplied_dcm_filenames should never be overriden after initialization
        # it is needed for repeated validation calls
        self.ro_user_supplied_dcm_filenames = list(self.one_volume_dcm_filenames)

        _first_filename_for_volume: Path = self.one_volume_dcm_filenames[0]
        # print(f"USING REFERENCE VOLUME:  {_first_filename_for_volume} for pydicom info")
        self._pydicom_info: pydicom.Dataset = pydicom.dcmread(
            _first_filename_for_volume, stop_before_pixels=True, force=True
        )

        # set diffusion information
        self.bvalue = get_bvalue(self._pydicom_info, round_to_nearst_10=True)
        self.diffusion_gradient = get_diffusion_gradient_direction(self._pydicom_info)
        if self.diffusion_gradient is not None:
            self.has_diffusion_gradient = True
        else:
            self.has_diffusion_gradient = False

        # set default values
        self.volume_index: int | None = None
        self.volume_modality: str = "INVALID"
        self.series_modality: str = "INVALID"
        self.modality_probability: pd.DataFrame | None = None
        self.average_slice_spacing = -12345.0
        self.acquisition_plane: str = "UNKNOWN"
        self.is_isotropic: bool = False
        self.has_contrast: bool = False
        self.itk_image: FImageType | None = None

        # process volume DICOM files
        (
            _one_study_found,
            self.volume_info_dict,
        ) = self._make_one_study_info_mapping_from_filelist()

    def set_volume_modality(self, modality: str) -> None:
        """
        Sets the modality of the DICOM data.

        Args:
            modality (str): The modality information to be set.
        """
        if not isinstance(modality, str):
            raise ValueError(
                f"ERROR: Can only set_modality with a string.  Got type(modality) = {type(modality)}."
            )
        self.volume_modality = modality

    def get_volume_modality(self) -> str:
        """
        Retrieves the modality of the DICOM data.

        Returns:
            str: The modality information.
        """
        return self.volume_modality

    def set_series_modality(self, modality: str) -> None:
        """
        Sets the modality of the DICOM data.

        Args:
            modality (str): The modality information to be set.
        """
        if not isinstance(modality, str):
            raise ValueError(
                f"ERROR: Can only set_modality with a string.  Got type(modality) = {type(modality)}."
            )
        self.series_modality = modality

    def get_series_modality(self) -> str:
        """
        Retrieves the modality of the DICOM data.

        Returns:
            str: The modality information.
        """
        return self.series_modality

    def set_is_isotropic(self, isotropic: bool) -> None:
        """
        Sets the isotropic flag of the DICOM data.

        Args:
            isotropic (bool): The isotropic flag to be set.
        """
        self.is_isotropic = isotropic

    def get_is_isotropic(self) -> bool:
        """
        Retrieves the isotropic flag of the DICOM data.

        Returns:
            bool: The isotropic flag.
        """
        return self.is_isotropic

    def set_has_contrast(self, contrast: bool) -> None:
        """
        Sets the contrast flag of the DICOM data.

        Args:
            contrast (bool): The contrast flag to be set.
        """
        self.has_contrast = contrast

    def get_has_contrast(self) -> bool:
        """
        Retrieves the contrast flag of the DICOM data.

        Returns:
            bool: The contrast flag.
        """
        return self.has_contrast

    def get_contrast_agent(self) -> str:
        """
        Retrieves the contrast agent of the DICOM data.

        Returns:
            str: The contrast agent.
        """
        if self.get_has_contrast():
            return self._pydicom_info.ContrastBolusAgent
        else:
            return "None"

    def set_modality_probabilities(
        self, modality_probability: pd.DataFrame | None
    ) -> None:
        """
        Sets the modality probabilities for the DICOM data.

        Args:
            modality_probability (pd.DataFrame): A pandas DataFrame containing modality probabilities.
        """
        if (
            not isinstance(modality_probability, pd.DataFrame)
            and modality_probability is not None
        ):
            raise ValueError(
                "ERROR: Can only set_modality_probabilities with a pd.DataFrame."
                f"Got type(modality_probability) = {type(modality_probability)}."
            )
        self.modality_probability = modality_probability

    def get_modality_probabilities(self) -> pd.DataFrame | None:
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

    def get_primary_volume_info(self, vol_index: int) -> dict[str, str]:
        """
        Get primary volume information for the specified volume index.

        Args:
            vol_index (int) Optional: Index of the volume for which to retrieve information, defaults to 0.

        Returns:
            Dict[str, str]: A dictionary containing primary volume information.
        """
        fields_to_copy: dict[str, str] = {
            "SeriesNumber": "SeriesNum",
            "Diffusionb-value": "Bval",
            "RepetitionTime": "TR",
            "EchoTime": "TE",
            "FlipAngle": "FA",
            "SAR": "SAR",
        }

        ref_vol_info: dict[str, Any] = self.get_volume_dictionary()
        return_dict: dict[str, str | int] = collections.OrderedDict()
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

    def get_series_size(self) -> str:
        """
        Get the size of the DICOM series.

        Returns:
            str: The size of the DICOM series as a string.
        """
        size_list: list[int] = [
            self._pydicom_info.Rows,
            self._pydicom_info.Columns,
            len(self.one_volume_dcm_filenames),
        ]
        return str(size_list)

    def get_one_volume_dcm_filenames(self) -> list[Path]:
        """
        Get the list of DICOM file paths for the single DICOM volume.

        Returns:
            List[Path]: A list of file paths for the DICOM files in the single volume.
        """
        return deepcopy(self.one_volume_dcm_filenames)

    def get_volume_dictionary(self) -> dict[str, Any]:
        """
        Get the dictionary containing information about the DICOM volume.

        Returns:
            Dict[str, Any]: A dictionary containing information about the DICOM volume.
        """
        return deepcopy(self.volume_info_dict)

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
        try:
            series_number_int: int = int(self._pydicom_info.SeriesNumber)
            return series_number_int
        except Exception as e:
            print(f"Can not convert to int {self._pydicom_info.SeriesNumber}: {e}")
        return -12345

    def get_volume_index(self) -> Optional[int]:
        """
        Get the Volume Index within the Series.

        Returns:
            int: The Volume Index as an integer.
        """
        return self.volume_index

    def set_volume_index(self, volume_index: int) -> None:
        """
        Set the Volume Index within the Series.

        Args:
            volume_index (int): The Volume Index within its series.
        """
        self.volume_index = volume_index

    def is_MR_modality(self) -> bool:
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
        sanitized_dicom_dict, valid = sanitize_dicom_dataset(
            ro_dataset=self._pydicom_info,
            required_info_list=required_DICOM_fields,
            optional_info_list=optional_DICOM_fields,
        )
        if not valid:
            self.set_volume_modality("INVALID")
            self.set_acquisition_plane("INVALID")

        volume_info_dict: dict[str, Any] = get_coded_dictionary_elements(
            sanitized_dicom_dict
        )
        del sanitized_dicom_dict

        # add features related to b-values and diffusion
        bvalue_current_dicom: int = int(self.get_volume_bvalue())
        volume_info_dict["Diffusionb-value"] = bvalue_current_dicom
        volume_info_dict["Diffusionb-valueBool"] = (
            0 if bvalue_current_dicom == -12345 else 1
        )
        volume_info_dict["has_b0"] = 1 if bvalue_current_dicom == 0 else 0
        volume_info_dict["has_pos_b0"] = 1 if bvalue_current_dicom > 0 else 0
        volume_info_dict["HasDiffusionGradientOrientation"] = int(
            self.has_diffusion_gradient
        )
        if (
            volume_info_dict["Image Type_DIFFUSION"] == 1
            or volume_info_dict["Image Type_ADC"] == 1
            or volume_info_dict["Image Type_EADC"] == 1
            or volume_info_dict["Image Type_TRACEW"] == 1
            or volume_info_dict["Image Type_FA"] == 1
            or volume_info_dict["Diffusionb-value"] > 0
            or volume_info_dict["HasDiffusionGradientOrientation"] == 1
        ):
            volume_info_dict["likely_diffusion"] = 1
        else:
            volume_info_dict["likely_diffusion"] = 0
        # those values are 1 in case of a single volume
        volume_info_dict["SeriesVolumeCount"] = 1
        # add list of dicom files for the volume
        volume_info_dict["list_of_ordered_volume_files"] = self.one_volume_dcm_filenames

        return self.get_study_uid, deepcopy(volume_info_dict)

    def get_image_diagnostics(self) -> str:
        """
        Generates diagnostic information about the DICOM image.

        Returns:
            msg (str): Diagnostic information as a formatted string.
        """
        volume_info: str = json.dumps(
            self.get_primary_volume_info(0), indent=4, sort_keys=True
        )
        msg = f"""
    {'*' * 40}
    {volume_info}
    {'*' * 40}
    """
        return msg
