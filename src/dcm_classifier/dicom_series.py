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

from dcm_classifier.dicom_volume import (
    DicomSingleVolumeInfoBase,
    merge_dictionaries,
)

import pandas as pd

from typing import List, Optional, Any, Dict


class DicomSingleSeries:
    """
    This class is used to store information about a single DICOM series. It organizes DicomSingleVolumeInfoBase objects
    by series.
    """

    def __init__(
        self,
        series_number: int,
    ) -> None:
        """
        Initialize a DicomSingleSeries instance with the provided series number.

        Args:
            series_number (int): The series number associated with this series.

        Attributes:
            series_number (int): The series number.
            volume_info_list (List[DicomSingleVolumeInfoBase]): A list to store
                DicomSingleVolumeInfoBase objects for this series.
            modality (Optional[str]): The modality of the series (e.g., "CT", "MRI").
            modality_probability (Optional[pd.DataFrame]): A DataFrame containing modality
                probabilities.
            acquisition_plane (Optional[str]): The acquisition plane of the series (e.g.,
                "Sagittal", "Axial").
        """
        self.series_number: int = series_number
        self.volume_info_list: List[DicomSingleVolumeInfoBase] = list()
        self.modality: Optional[str] = None
        self.modality_probability: Optional[pd.DataFrame] = None
        self.acquisition_plane: Optional[str] = None

    def get_series_number(self) -> int:
        """
        Get the series number of this DICOM series.

        Returns:
            int: The series number.
        """
        return self.series_number

    def set_modality(self, modality: str) -> None:
        """
        Set the modality of the DICOM series.

        Args:
            modality (str): The modality of the series (e.g., "CT", "MRI").
        """
        self.modality = modality

    def get_modality(self) -> str:
        """
        Get the modality of the DICOM series.

        Returns:
            str: The modality.
        """
        return self.modality

    def set_modality_probabilities(self, modality_probability: pd.DataFrame) -> None:
        """
        Set the modality probabilities DataFrame.

        Args:
            modality_probability (pd.DataFrame): A DataFrame containing modality probabilities.
        """
        self.modality_probability = modality_probability

    def get_modality_probabilities(self) -> pd.DataFrame:
        """
        Get the modality probabilities DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing modality probabilities.
        """
        return self.modality_probability

    def set_acquisition_plane(self, acquisition_plane: str) -> None:
        """
        Set the acquisition plane of the DICOM series.

        Args:
            acquisition_plane (str): The acquisition plane (e.g., "Sagittal", "Axial").

        """
        self.acquisition_plane = acquisition_plane

    def get_acquisition_plane(self) -> str:
        """
        Get the acquisition plane of the DICOM series.

        Returns:
            str: The acquisition plane.
        """
        return self.acquisition_plane

    def get_volume_list(self) -> List[DicomSingleVolumeInfoBase]:
        """
        Get the list of DicomSingleVolumeInfoBase objects for this series.

        Returns:
            List[DicomSingleVolumeInfoBase]: A list of volume information objects.
        """
        return self.volume_info_list

    def add_volume_to_series(self, new_volume_info: DicomSingleVolumeInfoBase) -> None:
        """
        Add a DicomSingleVolumeInfoBase object to the series.

        Args:
            new_volume_info (DicomSingleVolumeInfoBase): The volume information to add.

        """
        self.volume_info_list.append(new_volume_info)
        # Sort subvolumes based on bvalues similar to dcm2niix
        sorted(self.volume_info_list, key=lambda x: x.get_volume_bvalue())

    def get_series_info_dict(self) -> Dict[str, Any]:
        # """
        # Returns a dictionary with information about the series.
        # It merges the info_dicts of all volumes in the series and adds the bvalue-count, bvalue-max
        # and series count fields.
        # Returns:
        #
        # """
        """
        Get a dictionary with information about the series.

        This method merges information dictionaries from all volumes in the series and adds
        additional fields such as b-value statistics and volume count.

        Returns:
            Dict[str, Any]: A dictionary with information about the series.
        """
        if len(self.volume_info_list) == 1:
            info_dict = self.volume_info_list[0].get_volume_dictionary()
            info_dict["Diffusionb-valueCount"] = 1
            info_dict["Diffusionb-valueMax"] = self.volume_info_list[
                0
            ].get_volume_bvalue()
        else:
            bvals = []
            info_dict = self.volume_info_list[0].get_volume_dictionary()
            bvals.append(self.volume_info_list[0].get_volume_bvalue())
            for volume in self.volume_info_list[1:]:
                bvals.append(volume.get_volume_bvalue())
                current_dict = volume.get_volume_dictionary()
                info_dict = merge_dictionaries(info_dict, current_dict)
            info_dict["Diffusionb-valueSet"] = list(set(bvals))
            info_dict["Diffusionb-valueCount"] = len(list(set(bvals)))
            info_dict["Diffusionb-valueMax"] = max(bvals)
            info_dict["Diffusionb-valueMin"] = min(bvals)
        info_dict["SeriesVolumeCount"] = len(self.volume_info_list)
        return info_dict
