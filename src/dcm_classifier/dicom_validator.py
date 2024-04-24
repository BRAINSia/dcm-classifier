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

from .dicom_volume import DicomSingleVolumeInfoBase

from pathlib import Path

import json


class DicomValidatorBase:
    """
    This is a baseclass with almost no functionality to facilitate the creation of Validator classes for
    specific user projects

    Attributes:
        single_volume_info (DicomSingleVolumeInfoBase): An instance of DicomSingleVolumeInfoBase containing information about a single DICOM volume.

        _validation_failure_reports (List[str]): A list of validation failure messages.

    """

    def __init__(self, single_volume_info: DicomSingleVolumeInfoBase) -> None:
        """
        Initialize the DicomValidatorBase.

        :param single_volume_info: An instance of DicomSingleVolumeInfoBase containing information about a single DICOM volume.
        :type single_volume_info: DicomSingleVolumeInfoBase
        """
        self.single_volume_info: DicomSingleVolumeInfoBase = single_volume_info
        self._validation_failure_reports: list[str] = list()

    def append_to_validation_failure_reports(self, msg: str) -> None:
        """
        Append a validation failure message to the internal list.

        :param msg: A validation failure message to append.
        :type msg: str
        """
        if msg not in self._validation_failure_reports:
            self._validation_failure_reports.append(msg)

    def generate_validation_report_str(self, verbose_reporting: bool = False) -> str:
        """
        Generate a validation report as a formatted string.

        :param verbose_reporting: If True, includes verbose information in the report.
        :type verbose_reporting: bool
        :return: A formatted validation report string.
        :rtype: str
        """
        msg: str = ""
        if len(self._validation_failure_reports) > 0:
            validation_failure_listing: str = "\n".join(
                [str(f"\t{x}") for x in self._validation_failure_reports]
            )

            long_form_data: str
            if verbose_reporting:
                long_form_data = f"""
    Identified PrimaryInfo:
    {self.single_volume_info.get_image_diagnostics()}
    Identified dictionary:
    {json.dumps(self.single_volume_info.volume_info_dict, indent=2, sort_keys=True, default = lambda o: str(o))}
    """
            else:
                long_form_data = ""
            msg = f"""
    {"!" * 60}
    {"!" * 60}
    Identified image type: {self.single_volume_info.get_volume_modality()}-{self.single_volume_info.get_acquisition_plane()}
    Identified bvalue: {self.single_volume_info.get_volume_bvalue()}
    Identified SeriesNumber: {self.single_volume_info.get_series_number()}

    Failure Messages:
    {validation_failure_listing}
    {long_form_data}
    {"^" * 60}
    {"^" * 60}

    """
        return msg

    def write_validation_report(self, report_filename_to_append: Path | None) -> None:
        """
        Write the validation report to a file or print it.

        :param report_filename_to_append: The filename to write the report to. If None, the report will be printed to the console.
        :type report_filename_to_append: Path | None
        """
        msg: str = self.generate_validation_report_str()

        if report_filename_to_append is None:
            print(f"{msg}")
        else:
            with open(report_filename_to_append, "a") as vfid:
                vfid.write(f"{msg}\n")

    def validate(self) -> bool:
        """
        Perform validation checks on the DICOM volume.

        :return: True if the volume passes all validation criteria, false otherwise.
        :rtype: bool
        """
        return True
