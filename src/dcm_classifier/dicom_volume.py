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

from typing import Any
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

        is_isotropic (Optional[bool]): True if the DICOM volume is isotropic, False otherwise.

        has_contrast (Optional[bool]): True if the DICOM volume has contrast, False otherwise.

        parent_series (Optional[DicomSingleSeries]): The parent series of the DICOM volume.

        volume_index (Optional[int]): The index of the DICOM volume within its series.

        has_diffusion_gradient (bool): True if the DICOM volume has diffusion gradient, False otherwise.

    """

    def __init__(self, one_volume_dcm_filenames: list[Path | str]) -> None:
        """
        Initializes a DicomSingleVolumeInfoBase instance with a list of DICOM file paths.

        :param one_volume_dcm_filenames:
        :type one_volume_dcm_filenames: list[Path | str]

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
        self.parent_series = None
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

    def get_volume_series_description(self) -> str:
        """
        Get the Series Description of the DICOM volume.

        :return: The Series Description.
        :rtype: str
        """
        return self._pydicom_info.get("SeriesDescription", "UNKNOWN_SeriesDescription")

    def set_volume_modality(self, modality: str) -> None:
        """
        Sets the modality of the DICOM data.

        :param modality: The modality information to be set.
        :type modality: str
        """
        if not isinstance(modality, str):
            raise ValueError(
                f"ERROR: Can only set_modality with a string.  Got type(modality) = {type(modality)}."
            )
        self.volume_modality = modality

    def get_volume_modality(self) -> str:
        """
        Retrieves the modality of the DICOM data.

        :return: The modality information.
        :rtype: str
        """
        return self.volume_modality

    def get_series_modality(self) -> str:
        """
        Retrieves the modality of the DICOM data.

        :return: The modality information.
        :rtype: str
        """
        return self.parent_series.get_series_modality()

    def set_is_isotropic(self, isotropic: bool) -> None:
        """
        Sets the isotropic flag of the DICOM data.

        :param isotropic: The isotropic flag to be set.
        :type isotropic: bool
        """
        self.is_isotropic = isotropic

    def get_is_isotropic(self) -> bool:
        """
        Retrieves the isotropic flag of the DICOM data.

        :return: The isotropic flag.
        :rtype: bool
        """
        return self.is_isotropic

    def set_has_contrast(self, contrast: bool) -> None:
        """
        Sets the contrast flag of the DICOM data.

        :param contrast: The contrast flag to be set.
        :type contrast: bool
        """
        self.has_contrast = contrast

    def get_has_contrast(self) -> bool:
        """
        Retrieves the contrast flag of the DICOM data.

        :return: The contrast flag.
        :rtype: bool
        """
        return self.has_contrast

    def get_contrast_agent(self) -> str:
        """
        Retrieves the contrast agent of the DICOM data.

        :return: The contrast agent.
        :rtype: str
        """
        if self.get_has_contrast():
            return self._pydicom_info.get(
                "ContrastBolusAgent", "UNKNOWN_ContrastBolusAgent"
            )
        else:
            return "None"

    def set_parent_series(self, series) -> None:
        """
        Sets the parent series of the DICOM volume.

        :param series: The parent series object.
        :type series: DicomSingleSeries
        """
        self.parent_series = series

    def get_parent_series(self):
        """
        Retrieves the parent series of the DICOM volume.

        :return: The parent series object.
        :rtype: DicomSingleSeries
        """
        return self.parent_series

    def set_modality_probabilities(
        self, modality_probability: pd.DataFrame | None
    ) -> None:
        """
        Sets the modality probabilities for the DICOM data.

        :param modality_probability: The modality probabilities to be set.
        :type modality_probability: pd.DataFrame
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

        :return: The modality probabilities.
        :rtype: pd.DataFrame
        """
        return self.modality_probability

    def set_acquisition_plane(self, acquisition_plane: str) -> None:
        """
        Sets the acquisition plane information for the DICOM data.

        :param acquisition_plane: The acquisition plane information to be set.
        :type acquisition_plane: str

        """
        self.acquisition_plane = acquisition_plane

    def get_acquisition_plane(self) -> str:
        """
        Retrieves the acquisition plane information for the DICOM data.

        :return: The acquisition plane information.
        :rtype: str
        """
        return self.acquisition_plane

    def get_primary_volume_info(self, vol_index: int) -> dict[str, str]:
        """
        Get primary volume information for the specified volume index.

        :param vol_index: index of the volume for which to retrieve information
        :type vol_index: int, optional

        :return: A dictionary containing primary volume information.
        :rtype: Dict[str, str]
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

        :return: The ITK image of the DICOM volume.
        :rtype: FImageType

        """
        if self.itk_image is None:
            self.itk_image = itk_read_from_dicomfn_list(
                self.get_one_volume_dcm_filenames()
            )

        return self.itk_image

    def get_series_uid(self) -> str:
        """
        Get the Series Instance UID of the DICOM volume.

        :return: The Series Instance UID.
        :rtype: str
        """
        return self._pydicom_info.get("SeriesInstanceUID", "UNKNOWN_SeriesInstanceUID")

    def get_study_uid(self) -> str:
        """
        Get the Study Instance UID of the DICOM volume.

        :return: The Study Instance UID.
        :rtype: str
        """
        return self._pydicom_info.get("StudyInstanceUID", "UNKNOWN_StudyInstanceUID")

    def get_series_pixel_spacing(self) -> str:
        """
        Get the pixel spacing of the DICOM series.

        Returns:
            str: The pixel spacing as a string.
        """
        return str(self._pydicom_info.get("PixelSpacing", "[NA,NA]"))

    def get_series_size(self) -> str:
        """
        Get the size of the DICOM series.

        :return: The size of the DICOM series.
        :rtype: str
        """
        size_list: list[int] = [
            self._pydicom_info.get("Rows", 0),
            self._pydicom_info.get("Columns", 0),
            len(self.one_volume_dcm_filenames),
        ]
        return str(size_list)

    def get_one_volume_dcm_filenames(self) -> list[Path]:
        """
        Get the list of DICOM file paths for the single DICOM volume.

        :return: A list of file paths for the DICOM files in the single volume.
        :rtype: List[Path]
        """
        return deepcopy(self.one_volume_dcm_filenames)

    def get_volume_dictionary(self) -> dict[str, Any]:
        """
        Get the dictionary containing information about the DICOM volume.

        :return: A dictionary containing information about the DICOM volume.
        :rtype: Dict[str, Any]
        """
        return deepcopy(self.volume_info_dict)

    def get_volume_bvalue(self) -> float:
        """
        Get the b-value of the DICOM volume.

        :return: The b-value of the DICOM volume.
        :rtype: float

        """
        return self.bvalue

    def get_series_number(self) -> int:
        """
        Get the Series Number of the DICOM volume.

        :return: The Series Number.
        :rtype: int

        """
        try:
            series_number_int: int = int(self._pydicom_info.get("SeriesNumber", -12345))
            return series_number_int
        except Exception as e:
            if "SeriesNumber" not in self._pydicom_info:
                print("SeriesNumber not found in DICOM file")
            else:
                print(f"Can not convert to int {self._pydicom_info.SeriesNumber}: {e}")
        return -12345

    def get_volume_index(self) -> int | None:
        """
        Get the Volume Index within the Series.

        :return: The Volume Index within the Series.
        :rtype: int
        """
        return self.volume_index

    def set_volume_index(self, volume_index: int) -> None:
        """
        Set the Volume Index within the Series.

        :param volume_index: The Volume Index within its series.
        :type volume_index: int

        """
        self.volume_index = volume_index

    def is_MR_modality(self) -> bool:
        """
        Check if the modality of the DICOM volume is MR (Magnetic Resonance).

        :return: True if the modality is MR, False otherwise.
        :rtype: bool
        """
        status: bool = bool(
            self._pydicom_info.get("Modality", "Unknown_Modality") != "MR"
        )
        if not status:
            if "Modality" not in self._pydicom_info:
                vprint("Skipping DICOM file without Modality information")
            else:
                vprint(f"Skipping non-MR modality : {self._pydicom_info.Modality}")
        return status

    def _make_one_study_info_mapping_from_filelist(self) -> (str, dict):
        """
        Create a dictionary containing information about the DICOM volume from the list of DICOM files.

        :return: A tuple containing the Study Instance UID and a dictionary with volume information.
        :rtype: Tuple[str, dict]
        """
        # sanitize the DICOM dataset
        sanitized_dicom_dict, valid = sanitize_dicom_dataset(
            ro_dataset=self._pydicom_info,
            required_info_list=required_DICOM_fields,
            optional_info_list=optional_DICOM_fields,
        )
        # if the dataset is not valid, mark as INVALID and return an empty dictionary
        if not valid:
            self.set_volume_modality("INVALID")
            self.set_acquisition_plane("INVALID")
            return self.get_study_uid, dict()

        volume_info_dict: dict[str, Any] = get_coded_dictionary_elements(
            sanitized_dicom_dict
        )
        del sanitized_dicom_dict

        # ensure the volume_info_dict is not empty
        if not volume_info_dict:
            return self.get_study_uid, dict()

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
            volume_info_dict["ImageType_DIFFUSION"] == 1
            or volume_info_dict["ImageType_ADC"] == 1
            or volume_info_dict["ImageType_EADC"] == 1
            or volume_info_dict["ImageType_TRACEW"] == 1
            or volume_info_dict["ImageType_FA"] == 1
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

        :return: Diagnostic information about the DICOM image.
        :rtype: str
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
