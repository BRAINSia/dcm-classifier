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
from src.namic_dicom.namic_dicom_typing import (
    get_bvalue,
    FImageType,
    itk_read_from_dicomfn_list,
    vprint,
    get_coded_dictionary_elements,
)

pydicom_read_cache_static_filename_dict: Dict[str, pydicom.Dataset] = dict()


def pydicom_read_cache(
    filename: Union[Path, str], stop_before_pixels=True
) -> pydicom.Dataset:
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
    for key in ro_dict.keys():
        rw_dict_to_update[key] = ro_dict[key]
    return rw_dict_to_update


class DicomSingleVolumeInfoBase:
    def __init__(self, one_volume_dcm_filenames: List[Union[Path, str]]) -> None:
        self.one_volume_dcm_filenames: List[Path] = [
            Path(x).resolve() for x in one_volume_dcm_filenames
        ]
        # The ro_user_supplied_dcm_filenames should never be overriden after initialization
        # it is needed for repeated validation calls
        self.ro_user_supplied_dcm_filenames = list(self.one_volume_dcm_filenames)

        _first_filename_for_volume: Path = self.one_volume_dcm_filenames[0]
        # print(f"USING REFERENCE VOLUME:  {_first_filename_for_volume} for pydicom info")
        # TODO: Remove self.pydicom_info from this section, it is useful for debugging
        self.pydicom_info: pydicom.Dataset = pydicom_read_cache(
            _first_filename_for_volume, stop_before_pixels=True
        )

        self.bvalue = get_bvalue(self.pydicom_info, round_to_nearst_10=True)
        self.average_slice_spacing = -12345.0
        (
            _one_study_found,
            self.volume_info_dict,
        ) = self._make_one_study_info_mapping_from_filelist()

        self.itk_image: Optional[FImageType] = None

        self.modality: Optional[str] = None
        self.modality_probability: Optional[pd.DataFrame] = None
        self.acquisition_plane: Optional[str] = None

    def set_modality(self, modality: str) -> None:
        self.modality = modality

    def get_modality(self) -> str:
        return self.modality

    def set_modality_probabilities(self, modality_probability: pd.DataFrame) -> None:
        self.modality_probability = modality_probability

    def get_modality_probabilities(self) -> pd.DataFrame:
        return self.modality_probability

    def set_acquisition_plane(self, acquisition_plane: str) -> None:
        self.acquisition_plane = acquisition_plane

    def get_acquisition_plane(self) -> str:
        return self.acquisition_plane

    def get_volume_info_dict(self) -> Dict[str, Any]:
        return self.volume_info_dict

    # This is a bad measurement!!
    # def get_slice_thickness(self) -> float:
    #     return self.pydicom_info.SliceThickness
    def get_primary_volume_info(self, vol_index: int = 0) -> Dict[str, str]:
        fields_to_copy: Dict[str, str] = {
            "SeriesNumber": "SeriesNum",
            "Diffusionb-value": "Bval",
            "SeriesDescription": "SeriesDescription",
            # "ImageTypeADC",
            # "ImageTypeTrace",
            # "AxialIndicator",
            # "CoronalIndicator",
            # "SaggitalIndicator",
            # "IsDerivedImageType",
            "RepetitionTime": "TR",
            "EchoTime": "TE",
            "FlipAngle": "FA",
            "SAR": "SAR",
            # "SeriesVolumeCount": "Vols",
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
        if self.itk_image is None:
            self.itk_image = itk_read_from_dicomfn_list(
                self.get_one_volume_dcm_filenames()
            )

        return self.itk_image

    def get_series_uid(self) -> str:
        return self.pydicom_info.SeriesInstanceUID

    def get_study_uid(self) -> str:
        return self.pydicom_info.StudyInstanceUID

    def get_series_pixel_spacing(self) -> str:
        return str(self.pydicom_info.PixelSpacing)

    def get_series_spacing_between_slices(self) -> str:
        return str(self.average_slice_spacing)

    def get_series_size(self) -> str:
        size_list: List[int] = [
            self.pydicom_info.Rows,
            self.pydicom_info.Columns,
            len(self.one_volume_dcm_filenames),
        ]
        return str(size_list)

    def get_one_volume_dcm_filenames(self) -> List[Path]:
        return self.one_volume_dcm_filenames

    def get_volume_dictionary(self) -> Dict[str, Any]:
        return self.volume_info_dict
        # for f in fileNames:
        #     print(f"\t{f}")
        # print(len(self.one_volume_dcm_filenames))
        # print(
        #     f'XXXXXXXXXXXXXXXXXXX  {series_number}: {len(self.one_volume_dictionary["list_of_ordered_volume_files"])}'
        # )

    # def get_volume_datatype(self) -> str:
    #     return "unknown"

    def get_volume_bvalue(self) -> float:
        return self.bvalue

    def get_series_number(self) -> int:
        return int(self.pydicom_info.SeriesNumber)

    def is_MR_modality(self):
        status = bool(self.pydicom_info.Modality != "MR")
        if not status:
            vprint(f"Skipping non-MR modality : {self.pydicom_info.Modality}")
        return status

    def _make_one_study_info_mapping_from_filelist(self) -> (str, dict):
        dicom_file_name: Path = Path(self.pydicom_info.filename)

        volume_info_dict = dict()

        volume_info_dict["SeriesNumber"] = self.get_series_number()
        if "EchoTime" not in self.pydicom_info:
            vprint(f"Missing required echo time value {dicom_file_name}")
            volume_info_dict["EchoTime"] = -123456789.0
        if "SAR" not in self.pydicom_info:
            # Some derived datasets do not have SAR listed, so fill with zero
            vprint(f"Missing required SAR value {dicom_file_name}")
            volume_info_dict["SAR"] = -123456789.0
        bvalue_current_dicom: int = int(self.get_volume_bvalue())
        volume_info_dict["Diffusionb-value"] = bvalue_current_dicom
        volume_info_dict["Diffusionb-valueMax"] = bvalue_current_dicom

        if bvalue_current_dicom < -1:
            volume_info_dict["HasDiffusionGradientOrientation"] = 0
        else:
            volume_info_dict["HasDiffusionGradientOrientation"] = 1

        volume_info_dict["FileName"] = dicom_file_name.as_posix()
        volume_info_dict["StudyInstanceUID"] = str(self.pydicom_info.StudyInstanceUID)
        volume_info_dict["SeriesInstanceUID"] = str(self.pydicom_info.SeriesInstanceUID)
        volume_info_dict["SeriesDescription"] = str(self.pydicom_info.SeriesDescription)
        volume_info_dict["SeriesNumber"] = self.get_series_number()
        missing_info_flag: int = -1
        volume_info_dict["ImageTypeADC"] = missing_info_flag
        volume_info_dict["ImageTypeTrace"] = missing_info_flag
        volume_info_dict["AxialIndicator"] = missing_info_flag
        volume_info_dict["CoronalIndicator"] = missing_info_flag
        volume_info_dict["SaggitalIndicator"] = missing_info_flag
        volume_info_dict["IsDerivedImageType"] = missing_info_flag
        # values_dict[series_key]["IsSecondaryOrProcessed"] = missing_info_flag
        volume_info_dict["ImageType"] = "NOT_PROVIDED"

        curr_prostat_encoded_dict: Dict[str, Any] = get_coded_dictionary_elements(
            self.pydicom_info, True
        )
        # those values are 1 in case of a single volume
        curr_prostat_encoded_dict["Diffusionb-valueCount"] = 1
        curr_prostat_encoded_dict["SeriesVolumeCount"] = 1

        merge_dictionaries(volume_info_dict, curr_prostat_encoded_dict)

        volume_info_dict["list_of_ordered_volume_files"] = self.one_volume_dcm_filenames

        return self.get_study_uid, volume_info_dict

    def get_image_diagnostics(self) -> str:
        volume_info: str = json.dumps(
            self.get_primary_volume_info(), indent=4, sort_keys=True
        )
        msg = f"""
    {'*' * 40}
    {volume_info}
    {'*' * 40}
    """
        return msg
