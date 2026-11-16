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
import logging
import re
from collections import defaultdict
from datetime import datetime
from itertools import pairwise
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict

import numpy as np
import pydicom

from .dicom_config import MISSING_VALUE
from .dicom_series import DicomSingleSeries
from .dicom_volume import DicomSingleVolumeInfoBase
from .image_type_inference import ImageTypeClassifierBase
from .utility_functions import (
    ORIENTATION_TOLERANCE,
    SliceOrdering,
    check_two_images_have_same_physical_space,
    fallback_slice_ordering,
    geometric_slice_order,
    is_mr_sop_class,
    new_gdcm_series_file_names,
    parse_acquisition_datetime,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _series_number_order(item: tuple[int, Any]) -> tuple[bool, int]:
    """Sort key: numbered series ascending, then unnumbered ones as filed."""
    number = item[0]
    unnumbered = number <= MISSING_VALUE
    return unnumbered, -number if unnumbered else number


class _FileRecord(TypedDict):
    image_position: Any
    orientation: Any
    acquisition_date_time: datetime
    acquisition_time: datetime | None
    acquisition_number: int
    instance_number: int
    tags: dict[str, str | None]
    filename: str
    # The header itself is used only by geometric_slice_order and fallback_slice_ordering.
    dataset: pydicom.Dataset


_SPLIT_TAGS = ("TemporalPositionIdentifier", "AcquisitionNumber")


def _acquisition_order(record: _FileRecord) -> tuple[datetime, int, int]:
    return (
        record["acquisition_date_time"],
        record["acquisition_number"],
        record["instance_number"],
    )


def _tag_value(ds: pydicom.Dataset, keyword: str) -> Any:
    try:
        return ds.get(keyword)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        logger.debug("Unreadable %s: %s", keyword, error)
        return None


def _tag_text(ds: pydicom.Dataset, keyword: str) -> str | None:
    value = _tag_value(ds, keyword)
    text = "" if value is None else str(value).strip()
    return text or None


def _int_or_missing(ds: pydicom.Dataset, keyword: str) -> int:
    text = _tag_text(ds, keyword)
    try:
        return int(text) if text is not None else MISSING_VALUE
    except ValueError:
        return MISSING_VALUE


_DICOM_DATE = re.compile(r"\d{8}")
_DICOM_TIME = re.compile(r"(\d{2})(?:(\d{2})(?:(\d{2})(?:\.(\d{1,6}))?)?)?")
_DICOM_DATETIME = re.compile(
    r"(\d{8})(\d{2}(?:\d{2}(?:\d{2}(?:\.\d{1,6})?)?)?)?([+-]\d{2})?(\d{2})?"
)


def _acquisition_time(ds: pydicom.Dataset) -> datetime | None:
    """AcquisitionDateTime, or AcquisitionDate with AcquisitionTime; None unless it parses."""
    text = _tag_text(ds, "AcquisitionDateTime")
    if text is not None:
        match = _DICOM_DATETIME.fullmatch(text)
        if match is None:
            return None
        day, clock, offset_hours, offset_minutes = match.groups()
        if (offset_hours is None) != (offset_minutes is None):
            return None
        if offset_hours is not None:
            sign = -1 if offset_hours.startswith("-") else 1
            minutes = sign * (abs(int(offset_hours)) * 60 + int(offset_minutes))
            # DICOM limits the UTC offset to -1200 through +1400.
            if int(offset_minutes) >= 60 or not -720 <= minutes <= 840:
                return None
        # A date without a time cannot order slices acquired the same day.
        if clock is None:
            return None
    else:
        day, clock = _tag_text(ds, "AcquisitionDate"), _tag_text(ds, "AcquisitionTime")
        if day is None or clock is None or not _DICOM_DATE.fullmatch(day):
            return None
    parts = _DICOM_TIME.fullmatch(clock)
    if parts is None:
        return None
    hour, minute, second, fraction = parts.groups()
    try:
        return datetime.strptime(
            f"{day}{hour}{minute or '00'}{second or '00'}{(fraction or '').ljust(6, '0')}",
            "%Y%m%d%H%M%S%f",
        )
    except ValueError:
        return None


def _file_record(ds: pydicom.Dataset, filename: str) -> _FileRecord:
    """Every header value the volume splitters use, read once."""
    return {
        "image_position": ds.get("ImagePositionPatient"),
        "orientation": _tag_value(ds, "ImageOrientationPatient"),
        "acquisition_date_time": parse_acquisition_datetime(ds),
        "acquisition_time": _acquisition_time(ds),
        "acquisition_number": _int_or_missing(ds, "AcquisitionNumber"),
        "instance_number": _int_or_missing(ds, "InstanceNumber"),
        "tags": {tag: _tag_text(ds, tag) for tag in _SPLIT_TAGS},
        "filename": filename,
        "dataset": ds,
    }


def _position_ranks(
    by_position: dict[tuple[Any, ...], list[_FileRecord]],
) -> dict[tuple[Any, ...], int] | None:
    """Rank of each position along the shared slice normal, or None when it is unknown."""
    try:
        cosines = [
            np.asarray(r["orientation"], float)
            for records in by_position.values()
            for r in records
        ]
        points = {key: np.asarray(key, float) for key in by_position}
    except (AttributeError, TypeError, ValueError):
        return None
    if any(c.shape != (6,) for c in cosines) or any(
        v.shape != (3,) for v in points.values()
    ):
        return None
    if any(np.max(np.abs(c - cosines[0])) > ORIENTATION_TOLERANCE for c in cosines):
        return None
    normal = np.cross(cosines[0][:3], cosines[0][3:])
    along = {key: float(np.dot(v, normal)) for key, v in points.items()}
    return {key: rank for rank, key in enumerate(sorted(along, key=along.__getitem__))}


def _split_by_repeat(
    by_position: dict[tuple[Any, ...], list[_FileRecord]],
) -> list[list[_FileRecord]] | None:
    """Split by repeat rank at each position, when every position repeats equally."""
    counts = {len(records) for records in by_position.values()}
    if len(counts) != 1:
        return None
    numbered = {
        r["acquisition_number"] != MISSING_VALUE
        for records in by_position.values()
        for r in records
    }
    # Repeats cannot be ranked when only some acquisition numbers can be read.
    if len(numbered) != 1:
        return None
    records = [r for group in by_position.values() for r in group]
    if any(r["instance_number"] == MISSING_VALUE for r in records):
        return None
    repeats = counts.pop()
    parsed = {id(r): r["acquisition_time"] for r in records}
    clock = {key: when for key, when in parsed.items() if when is not None}

    def split(
        sort_key: "Callable[[_FileRecord], Any]",
        order_key: "Callable[[_FileRecord], Any]",
    ) -> list[list[_FileRecord]] | None:
        ranked = [sorted(group, key=sort_key) for group in by_position.values()]
        parts = [[group[i] for group in ranked] for i in range(repeats)]
        bounds = [(min(map(order_key, p)), max(map(order_key, p))) for p in parts]
        # Each repeat must finish before the next starts, or ranks do not identify acquisitions.
        separated = len(by_position) == 1 or all(
            a[1] < b[0] for a, b in pairwise(bounds)
        )
        return parts if separated else None

    by_instance = split(
        lambda r: (r["acquisition_number"], r["instance_number"]),
        lambda r: r["instance_number"],
    )
    if by_instance is None:
        return None
    # Times that parse cross-check InstanceNumber; any disagreement means one is wrong.
    timed = [[r for r in group if id(r) in clock] for group in by_position.values()]
    if any(
        [r["instance_number"] for r in sorted(group, key=lambda r: clock[id(r)])]
        != sorted(r["instance_number"] for r in group)
        for group in timed
    ):
        return None
    if len(clock) < len(records):
        return by_instance
    by_time = split(
        lambda r: (clock[id(r)], r["acquisition_number"], r["instance_number"]),
        lambda r: clock[id(r)],
    )
    if by_time is not None and [{id(r) for r in p} for p in by_time] != [
        {id(r) for r in p} for p in by_instance
    ]:
        return None
    return by_instance


def _split_by_acquisition(
    by_position: dict[tuple[Any, ...], list[_FileRecord]],
) -> tuple[list[list[_FileRecord]] | None, bool]:
    """Split by acquisition tags giving contiguous slabs without repeats; True if tags conflict."""
    ranks = _position_ranks(by_position)
    if ranks is None:
        return None, False
    keyed = [(key, r) for key, records in by_position.items() for r in records]
    splits: list[list[list[_FileRecord]]] = []
    for tag in _SPLIT_TAGS:
        values = [r["tags"][tag] for _, r in keyed]
        if any(v is None for v in values):
            continue
        groups: defaultdict[str, list[tuple[tuple[Any, ...], _FileRecord]]] = (
            defaultdict(list)
        )
        for value, item in zip(values, keyed, strict=True):
            groups[str(value)].append(item)
        if not all(len({key for key, _ in g}) == len(g) for g in groups.values()):
            continue
        # A tag numbering every slice does not identify volumes, except for one-position series.
        if len(by_position) > 1 and min(len(g) for g in groups.values()) < 2:
            continue
        spans = [sorted(ranks[key] for key, _ in g) for g in groups.values()]
        if any(s[-1] - s[0] + 1 != len(s) for s in spans):
            continue
        parts = [[r for _, r in g] for g in groups.values()]
        splits.append(sorted(parts, key=lambda p: min(map(_acquisition_order, p))))
    # When both tags separate the repeats they must agree, or one of them is wrong.
    volumes = {tuple(frozenset(id(r) for r in part) for part in s) for s in splits}
    if len(volumes) > 1:
        return None, True
    return (splits[0] if splits else None), False


class ProcessOneDicomStudyToVolumesMappingBase:
    """
    Base class for processing a DICOM study. A study is typically all MRI scans for a single patient's scanning session.

    Attributes:
        series_restrictions_list_dwi_subvolumes (List[str]): List of DICOM tags used for filtering DWI sub-volumes.

        study_directory (Path): The path to the DICOM study directory as a pathlib Path object.

        search_series (Optional[Dict[str, int]]): A dictionary of series to search within the study.

        series_dictionary (Dict[int, DicomSingleSeries]): A dictionary mapping series numbers to DicomSingleSeries objects.

        inferer (Optional[ImageTypeClassifierBase]): An image type classifier for inference.
    """

    series_restrictions_list_dwi_subvolumes: ClassVar[list[str]] = [
        # https://www.na-mic.org/wiki/NAMIC_Wiki:DTI:DICOM_for_DWI_and_DTI
        # STANDARD
        "0018|9075",  # S 1 Diffusion Directionality
        "0018|9076",  # SQ 1 Diffusion Gradient Direction Sequence
        "0018|9087",  # FD 1 Diffusion b - value
        "0018|9089",  # F,D 3 Diffusion Gradient Orientation
        "0018|9117",  # SQ 1 MR Diffusion Sequence
        "0018|9147",  # CS 1 Diffusion Anisotropy Type
        "0018|9602",  # FD 1 DiffusionBValueXX
        "0018|9603",  # FD 1 DiffusionBValueXY
        "0018|9604",  # FD 1 DiffusionBValueXZ
        "0018|9605",  # FD 1 DiffusionBValueYY
        "0018|9606",  # FD 1 DiffusionBValueYZ
        "0018|9607",  # FD 1 DiffusionBValueZZ
        "0018,0086",  # IS 1-n EchoNumber  (needed for fieldmaps, PD/T2, T2* parametric imaging, and others)
        # Private vendor: GE
        "0019|10e0",  # DTI diffusion directions (release 10.0 & above)
        "0019|10df",  # DTI diffusion directions (release 9.0 & below)
        "0019|10d9",  # Concatenated SAT {# DTI Diffusion Dir., release 9.0 & below}
        "0021|105A",  # diffusion direction
        "0043|1039",  # Diffusion b-value Slop_int_6... slop_int_9: (in the GEMS_PARM_01 block)
        # Private vendor: Siemens
        "0019|100C",  # Siemens # 2,1 B_value
        "0019|100d",  # Siemens # 8,1 DiffusionDirectionality
        "0019|100e",  # Siemens # 24,3 DiffusionGradientDirection
        "0019|1027",  # Siemens # 48,6 B_matrix
        # Private vendor: Siemens(Historical)
        "0029|1010",  # NOT SUPPORTED Siemens Historical Diffusion b-value
        "0019|000A",  # ;SIEMENS MR HEADER  ;NumberOfImagesInMosaic          ;1;US;1
        "0019|000B",  # ;SIEMENS MR HEADER  ;SliceMeasurementDuration        ;1;DS;1
        "0019|000C",  # ;SIEMENS MR HEADER  ;B_value                         ;1;IS;1
        "0019|000D",  # ;SIEMENS MR HEADER  ;DiffusionDirectionality         ;1;CS;1
        "0019|000E",  # "SIEMENS MR HEADER  ;DiffusionGradientDirection      ;1;FD;3
        "0019|000F",  # ;SIEMENS MR HEADER  ;GradientMode                    ;1;SH;1
        "0019|0027",  # ;SIEMENS MR HEADER  ;B_matrix                        ;1;FD;6
        "0019|0028",  # ;SIEMENS MR HEADER  ;BandwidthPerPixelPhaseEncode    ;1;FD;1"
        # Private vendor: Philips
        "2001|1003",  # FL : Diffusion B-Factor
        "2001|1004",  # CS : Diffusion Direction
        "2005|10B0",  # FL : Diffusion Direction RL
        "2005|10B1",  # FL : Diffusion Direction AP
        "2005|10B2",  # FL : Diffusion Direction FH"
        # Private vendor: Toshiba
        "0029|1001",  # : Private Sequence" Gradients
        "0029|1090",  # : Private Byte Data
        # Private vendor: UIH
        "0065|1009",  # FD 929.75018313375267  # 8, 1 B_value
        "0065|1037",  # FD 0.219573072677\0.9618\0.1632  # 24, 3 DiffusionDirectionality
    ]

    @staticmethod
    def _is_pathlike_object(path_rep: str | Path | PurePath) -> bool:
        """
        Check if the given object represents a valid path or path-like object.

        This method determines whether the provided object is an instance of pathlib's PurePath or a string.

        :param path_rep: str | Path | PurePath: The object to check for path-likeness.
        :return: True if the object is a valid path or path-like object, False otherwise.
        :rtype: bool

        """
        # checks if the variable is any instance of pathlib
        return bool(isinstance(path_rep, (PurePath, str)))

    def __init__(
        self,
        study_directory: str | Path,
        search_series: dict[str, int] | None = None,
        inferer: ImageTypeClassifierBase | None = None,
        raise_error_on_failure: bool = False,
        mr_sop_class_only: bool = False,
        allow_fallback_ordering: bool = False,
    ) -> None:
        """
        Initialize an instance of ProcessOneDicomStudyToVolumesMappingBase.

        This constructor sets up the object with the provided parameters, including the study directory,
        search_series dictionary, and optional image type classifier.

        :param study_directory: str | Path: The path to the DICOM study directory.
        :param search_series: Optional[Dict[str, int]]: A dictionary of series to search within the study.
        :param inferer: Optional[ImageTypeClassifierBase]: An image type classifier for inference.
        :param mr_sop_class_only: bool: Optional. Only process series whose first file has an MR SOP Class UID.
        :param allow_fallback_ordering: bool: Optional. By default, raise ValueError for a
            volume whose slices could not be ordered by position. True accepts such
            volumes with a warning; their images may have wrong geometry.

        """
        if ProcessOneDicomStudyToVolumesMappingBase._is_pathlike_object(
            study_directory
        ):
            self.study_directory = Path(study_directory)  # coerce to path object
            del study_directory
        else:
            msg = f"study_directory must be path-like; got {study_directory!r}"
            raise TypeError(msg)
        self.raise_error_on_failure: bool = raise_error_on_failure
        self._unnumbered_series: dict[str, int] = {}
        self.search_series: dict[str, int] | None = search_series
        self.series_dictionary: dict[int, DicomSingleSeries] = (
            self.__identify_single_volumes(self.study_directory, mr_sop_class_only)
        )
        for series_number, series in self.series_dictionary.items():
            fallbacks = series.get_slice_orderings() - {
                SliceOrdering.GEOMETRIC,
                SliceOrdering.GEOMETRIC_SUBVOLUME,
            }
            if fallbacks:
                message = (
                    f"SeriesNumber {series_number} in {self.study_directory} has volumes "
                    f"whose slices could not be ordered by position: "
                    f"{', '.join(sorted(fallbacks))}"
                )
                if not allow_fallback_ordering:
                    msg = (
                        f"{message}. Pass allow_fallback_ordering=True to accept them."
                    )
                    raise ValueError(msg)
                logger.warning(message)
            if series.get_is_mosaic():
                logger.warning(
                    "SeriesNumber %s in %s contains Siemens mosaic images. "
                    "Mosaic images are not unpacked into slices, so the geometry and "
                    "classification of this series may be wrong.",
                    series_number,
                    self.study_directory,
                )
        self.inferer: ImageTypeClassifierBase | None = inferer

    def get_list_of_primary_volume_info(self) -> list[dict[str, str | int]]:
        """
        Retrieve a list of dictionaries containing primary volume information from all series.

        This method iterates through the series stored in the class's series_dictionary and
        extracts primary volume information from each subseries within those series. It then
        compiles this information into a list of dictionaries, with each dictionary representing
        primary volume information for a single subseries.

        :return: A list of dictionaries containing primary volume information.
        :rtype: list[dict[str, str]]
        """
        list_of_volume_info_dictionaries: list[dict[str, str | int]] = []
        for series_object in self.series_dictionary.values():
            for vol_index, subseries_vol_info in enumerate(
                series_object.volume_info_list
            ):
                primary_volume_info: dict[str, str | int] = (
                    subseries_vol_info.get_primary_volume_info(vol_index)
                )
                list_of_volume_info_dictionaries.append(primary_volume_info)
        return list_of_volume_info_dictionaries

    def get_study_dictionary(self) -> dict[int, DicomSingleSeries]:
        """
        Get the dictionary mapping series numbers to DicomSingleSeries objects.

        This method provides access to the internal dictionary that maps series numbers to
        DicomSingleSeries objects within the instance of the class. The dictionary stores
        information about DICOM series associated with the DICOM study.

        :return: dict[int, DicomSingleSeries]: A dictionary mapping series numbers to DicomSingleSeries objects containing information about each series.

        :Example:
            Retrieve information for a specific series number from the study dictionary:

            >>> study = ProcessOneDicomStudyToVolumesMappingBase(study_directory_path)
            >>> study_dict = study.get_study_dictionary()
            >>> series_info = study_dict.get(1)  # Retrieve information for series number 1
        """
        return self.series_dictionary

    def set_inferer(self, inferer: ImageTypeClassifierBase) -> None:
        """
        Set the image type classifier for inference.

        This method allows you to set the image type classifier (inferer) to be used for
        inference on the DICOM series. The provided `inferer` should be an instance of
        a class that derives from `ImageTypeClassifierBase`.

        :param inferer: An image type classifier object for inference.
        :type inferer: ImageTypeClassifierBase

        :Example:
            Setting an image type classifier for inference:

            >>> study = ProcessOneDicomStudyToVolumesMappingBase(study_directory_path)
            >>> image_classifier = MyImageTypeClassifier()  # Replace with your classifier instance
            >>> study.set_inferer(image_classifier)
        """
        self.inferer = inferer

    def run_inference(self) -> None:
        """
        Run inference on each DICOM series using the specified image type classifier.

        This method iterates through the DICOM series stored in the class's series_dictionary
        and performs inference on each series using the configured image type classifier
        (inferer). It sets the current series for inference using `set_series` and then
        invokes the `run_inference` method of the image type classifier for each series.

        .. note::
            Ensure that you have configured an image type classifier (inferer) using the
            `set_inferer` method before calling this method.

        """
        if self.inferer is None:
            msg = "No image type classifier is set; call set_inferer first."
            raise AttributeError(msg)
        for series_object in self.series_dictionary.values():
            self.inferer.set_series(series_object)
            self.inferer.run_inference()
        self.fixup_adjacent_series()

    def fixup_adjacent_series(self) -> None:
        """
        Fix up adjacent series that are part of the same family of images.

        This method iterates through the DICOM series stored in the class's series_dictionary
        and fixes up cases where adjacent series are part of the same family of images. It
        checks for cases where a bvalue=0 image is part of a family of tracew images and
        updates the series modality to "tracew" for the adjacent series.

        """
        # Fix up case where adjacent volumes make up a family of tracew images and one of them is a bvalue=0 image
        for current_series_number, series_obj in self.get_study_dictionary().items():
            if series_obj.get_volume_list()[
                0
            ].get_volume_bvalue() == 0 and series_obj.get_series_modality() not in [
                "tracew",
                "adc",
                "fa",
                "eadc",
                "dwig",
            ]:
                for (
                    adjacent_series_number,
                    adjacent_series_obj,
                ) in self.get_study_dictionary().items():
                    if adjacent_series_number != current_series_number:
                        list_of_candidate_adjacent = [
                            int(current_series_number + 1),
                            int(current_series_number - 1),
                        ]
                        if (
                            adjacent_series_number in list_of_candidate_adjacent
                            and adjacent_series_obj.get_series_modality() == "tracew"
                        ):
                            # if an adjacent image is a tracew image, and has same physical space as
                            # the current image then assume it is also a tracew image
                            itk_volume = series_obj.get_volume_list()[0].get_itk_image()
                            itk_adjacent_volume = adjacent_series_obj.get_volume_list()[
                                0
                            ].get_itk_image()

                            same_physical_space: bool = (
                                check_two_images_have_same_physical_space(
                                    itk_volume, itk_adjacent_volume
                                )
                            )

                            if same_physical_space:
                                series_obj.set_series_modality("tracew")
                            else:
                                logger.warning(
                                    "Adjacent series %s does not have same physical space as %s",
                                    adjacent_series_number,
                                    current_series_number,
                                )
                                logger.debug("%s", itk_volume)
                                logger.debug("%s", itk_adjacent_volume)

    def validate(self) -> None:
        pass

    def __identify_single_volumes(
        self,
        study_directory: Path,
        mr_sop_class_only: bool = False,
    ) -> dict[int, DicomSingleSeries]:
        """
        Identify and map single volumes within the DICOM study directory.

        This method scans the provided DICOM study directory for series that match the specified
        restrictions and organizes them into a dictionary, mapping series numbers to
        DicomSingleSeries objects.

        :param study_directory: The path to the DICOM study directory.
        :type study_directory: Path

        :param mr_sop_class_only: Optional. Only retrieve series whose first file has an MR SOP Class UID.
        :type mr_sop_class_only: bool

        :return: A dictionary mapping series numbers to DicomSingleSeries objects.
        :rtype: dict[int, DicomSingleSeries]

        .. notes::
            - This method uses ITK's GDCMSeriesFileNames to identify series within the study
              based on defined restrictions.
            - Series matching the restrictions are organized into DicomSingleSeries objects
              within the returned dictionary.

        :Example:
            Identify and retrieve information for single volumes within a DICOM study directory:

            >>> study = ProcessOneDicomStudyToVolumesMappingBase(study_directory_path)
            >>> series_mapping = study.__identify_single_volumes(study_directory_path)
            >>> series_info = series_mapping.get(1)  # Retrieve information for series number 1
        """
        namesGenerator = new_gdcm_series_file_names()
        namesGenerator.SetLoadPrivateTags(True)
        namesGenerator.SetRecursive(True)
        # namesGenerator.AddSeriesRestriction("0008|0021")  # Date restriction
        # namesGenerator.AddSeriesRestriction("0020|0013") # For testing, SeriesInstance results in images with 1 value
        namesGenerator.SetGlobalWarningDisplay(False)
        namesGenerator.SetDirectory(study_directory.as_posix())

        seriesUID = namesGenerator.GetSeriesUIDs()

        if len(seriesUID) < 1:
            msg: str = f"No DICOMs in: {study_directory} (__identify_single_volumes)"
            if self.raise_error_on_failure:
                raise FileNotFoundError(msg)
            logger.warning(
                "No readable dicoms DICOMs in: %s (__identify_single_volumes)",
                study_directory,
            )
        else:
            logger.info(
                "The directory: %s contains %d DICOM series",
                study_directory,
                len(seriesUID),
            )

        volumes_dictionary: dict[int, DicomSingleSeries] = {}

        # Sorted so unnumbered series are numbered the same on every ITK version.
        # TODO Parallelize this loop to speed up the process NOT A PRIORITY
        for seriesIdentifier in sorted(seriesUID):
            # Get all filenames belonging to the current sub-series
            subseries_filenames: list[str] = namesGenerator.GetFileNames(
                seriesIdentifier
            )
            itk_fallback = bool(
                getattr(namesGenerator, "GetDidUseAmbiguousOrdering", lambda: False)()
            )

            if (
                mr_sop_class_only
                and subseries_filenames
                and not is_mr_sop_class(
                    pydicom.dcmread(
                        subseries_filenames[0], stop_before_pixels=True, force=True
                    )
                )
            ):
                continue

            # Use our pydicom-based splitter to handle multi-volume logic
            sub_volumes = self._identify_and_split_sub_volumes_pydicom(
                subseries_filenames, itk_fallback=itk_fallback
            )

            # Organize these sub-volumes by their SeriesNumber
            sn: int = -1
            for volume_obj in sub_volumes:
                sn = self._series_number_of(volume_obj)
                if sn not in volumes_dictionary:
                    volumes_dictionary[sn] = DicomSingleSeries(series_number=sn)
                volumes_dictionary[sn].add_volume_to_series(volume_obj)
            logger.info("\t%s has %d subvolumes", sn, len(sub_volumes))

        volumes_dictionary = dict(
            sorted(volumes_dictionary.items(), key=_series_number_order)
        )
        for series in volumes_dictionary.values():
            series.order_volumes_by_acquisition()

        # Optionally filter to only the user-requested series numbers
        if self.search_series is not None:
            candidate_series_numbers: list[int] = [
                int(x) for x in self.search_series.values()
            ]
            series_numbers_to_remove: list[int] = [
                sn for sn in volumes_dictionary if sn not in candidate_series_numbers
            ]
            for series_number in series_numbers_to_remove:
                del volumes_dictionary[series_number]
            del series_numbers_to_remove
        return volumes_dictionary

    def _series_number_of(self, volume: DicomSingleVolumeInfoBase) -> int:
        """
        Get the SeriesNumber a volume is filed under.

        A volume without a usable SeriesNumber is filed by its SeriesInstanceUID:
        each such series gets its own number, counting down from MISSING_VALUE.

        :param volume: The volume to file.
        :return: The SeriesNumber, or a number at or below MISSING_VALUE when it is
            absent or not an integer.
        :raises ValueError: If the SeriesNumber is unusable and raise_error_on_failure is set.
        """
        try:
            return int(volume.get_dicom_field_by_name("SeriesNumber"))
        except (TypeError, ValueError):
            message = (
                "SeriesNumber is missing or not an integer in "
                f"{volume.get_one_volume_dcm_filenames()[0]}"
            )
            if self.raise_error_on_failure:
                raise ValueError(message) from None
            series_uid = str(volume.get_dicom_field_by_name("SeriesInstanceUID"))
            number = self._unnumbered_series.setdefault(
                series_uid, MISSING_VALUE - len(self._unnumbered_series)
            )
            logger.warning("%s; filed under series %s", message, number)
            return number

    def _identify_and_split_sub_volumes_pydicom(
        self, subseries_filenames: list[str], itk_fallback: bool = False
    ) -> list[DicomSingleVolumeInfoBase]:
        """
        Identify and split multi-volume DICOM sub-series into separate volumes using pydicom.
        This approach primarily uses unique ImagePositionPatient entries to detect
        when multiple volumes are present (e.g. repeated slices over time or b-values).

        Steps:
            1) Read minimal header info for each file (stop_before_pixels=True).
            2) Collect ImagePositionPatient for each file to determine the number of unique slice positions.
            3) Sort the files (by InstanceNumber or any stable criterion).
            4) If (# unique positions) < (# total files), assume multiple volumes and slice up accordingly.
            5) Return a list of DicomSingleVolumeInfoBase objects.

        :param subseries_filenames: List of DICOM file paths from one sub-series.
        :param itk_fallback: ITK could not order these files by position.
        :return: A list of DicomSingleVolumeInfoBase objects, each representing a 3D volume.

        ASSUMPTIONS:

        -- All subvolumes have the same number of slices and same ImagePositionPatient values.

        Need to handle the case where position "0" is taken for all sub volumes before moving to the next position.
        Currently this would results in Non valid values for the position.

        Create a data dict with each

        dict[IPP_tuple,list[dict[str, any]]] = {
        IPP_tuple: [
                {
                "filename": str,
                "acquisition_date_time": datetime,
                "acquisition_time": str,
                "instance_number": int
                },
                ...
            ]
        }
        for each key in the dict
            -- sort the list by DateTime Object that we create, instance_number, acquisition_number
            -- Sanity check that all have the same number of slices (Unless Mosaic which will be handled separately)

        For number of slices you have pull that index value off and create the DICOMSingleVolumeInfoBase object

        return the list of DICOMSingleVolumeInfoBase objects

        """
        if not subseries_filenames:
            return []
        # Name order makes the read order, and the error fallback, OS independent.
        subseries_filenames = sorted(subseries_filenames)

        file_info_dict: defaultdict[tuple[Any, ...], list[_FileRecord]] = defaultdict(
            list
        )
        ordered_filenames: list[Path | str] = list(subseries_filenames)
        single_volume_ordering = SliceOrdering.FALLBACK_FILENAME

        try:
            file_records: list[_FileRecord] = []
            for dcm_file in subseries_filenames:
                try:
                    ds = pydicom.dcmread(dcm_file, stop_before_pixels=True)

                    # Validate ImagePositionPatient
                    img_position = ds.get("ImagePositionPatient")
                    if img_position is None and self.raise_error_on_failure:
                        msg = f"ImagePositionPatient not found in {dcm_file}"
                        raise ValueError(msg)

                    file_records.append(_file_record(ds, dcm_file))

                except Exception as file_error:
                    if self.raise_error_on_failure:
                        raise
                    logger.warning("Error processing file %s: %s", dcm_file, file_error)
                    # If error processing a single file, continue with others
                    continue

            # Slices are ordered by InstanceNumber.
            file_records.sort(key=lambda r: (r["instance_number"], str(r["filename"])))
            read_filenames = {str(r["filename"]) for r in file_records}
            ordered_filenames = [r["filename"] for r in file_records]
            ordered_filenames += [
                f for f in subseries_filenames if str(f) not in read_filenames
            ]
            for file_info in file_records:
                img_position = file_info["image_position"]
                if img_position is None:
                    logger.warning(
                        "ImagePositionPatient not found in %s. Using default positioning.",
                        file_info["filename"],
                    )
                    img_position = [0, 0, len(file_info_dict)]
                file_info_dict[tuple(img_position)].append(file_info)

            # Sort files within each position group
            for position_records in file_info_dict.values():
                position_records.sort(
                    key=lambda x: (
                        x["acquisition_date_time"],
                        x["acquisition_number"],
                        x["instance_number"],
                    )
                )

            all_read = len(file_records) == len(subseries_filenames)
            if 0 < len(file_info_dict) < len(file_records):
                groups = None
                if all_read:
                    groups, conflict = _split_by_acquisition(file_info_dict)
                    if groups is None and not conflict:
                        groups = _split_by_repeat(file_info_dict)
                if groups is None:
                    msg = (
                        f"Repeated slice positions cannot be separated into volumes: "
                        f"{len(subseries_filenames)} files, "
                        f"{len(file_info_dict)} positions"
                    )
                    if self.raise_error_on_failure:
                        raise ValueError(msg)
                    logger.warning(msg)
                    return [
                        DicomSingleVolumeInfoBase(
                            one_volume_dcm_filenames=ordered_filenames,
                            slice_ordering=SliceOrdering.FALLBACK_REPEATED_POSITIONS,
                        )
                    ]

                sub_volume_info_list: list[DicomSingleVolumeInfoBase] = []
                for group in groups:
                    datasets = [r["dataset"] for r in group]
                    order = geometric_slice_order(datasets)
                    if order is None:
                        ordering = fallback_slice_ordering(datasets)
                        by_instance = ordering == SliceOrdering.FALLBACK_INSTANCE_NUMBER
                        records = sorted(
                            group,
                            key=lambda r: (
                                r["instance_number"] if by_instance else 0,
                                str(r["filename"]),
                            ),
                        )
                    else:
                        records = [group[j] for j in order]
                        ordering = SliceOrdering.GEOMETRIC_SUBVOLUME
                    sub_volume_info_list.append(
                        DicomSingleVolumeInfoBase(
                            one_volume_dcm_filenames=[r["filename"] for r in records],
                            slice_ordering=ordering,
                        )
                    )
                return sub_volume_info_list

            datasets = [r["dataset"] for r in file_records]
            if (
                not itk_fallback
                and all_read
                and geometric_slice_order(datasets) is not None
            ):
                return [
                    DicomSingleVolumeInfoBase(
                        one_volume_dcm_filenames=ordered_filenames,
                        slice_ordering=SliceOrdering.GEOMETRIC,
                    )
                ]
            if all_read:
                single_volume_ordering = fallback_slice_ordering(datasets)

        except Exception as global_error:
            if self.raise_error_on_failure:
                raise
            logger.warning("Error in volume splitting: %s", global_error)

        if single_volume_ordering == SliceOrdering.FALLBACK_FILENAME:
            ordered_filenames = list(subseries_filenames)
        return [
            DicomSingleVolumeInfoBase(
                one_volume_dcm_filenames=ordered_filenames,
                slice_ordering=single_volume_ordering,
            )
        ]
