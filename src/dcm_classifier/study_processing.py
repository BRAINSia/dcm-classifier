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

from pathlib import Path, PurePath
import itk

from .dicom_volume import DicomSingleVolumeInfoBase
from .dicom_series import DicomSingleSeries
from .image_type_inference import ImageTypeClassifierBase


class ProcessOneDicomStudyToVolumesMappingBase:
    """
    Base class for processing a DICOM study. A study is typically all MRI scans for a single patient's scanning session.

    Attributes:
        series_restrictions_list_dwi_subvolumes (List[str]): List of DICOM tags used for filtering DWI sub-volumes.

    Methods:
        _is_pathlike_object(path_rep: Union[str, Path, PurePath]) -> bool:

        __init__(
            self,
            study_directory: Union[str, Path],
            search_series: Optional[Dict[str, int]] = None,
            inferer: Optional[ImageTypeClassifierBase] = None,
        ):

        get_list_of_primary_volume_info(self) -> List[Dict[str, str]]:

        get_study_dictionary(self):

        set_inferer(self, inferer: ImageTypeClassifierBase):

        run_inference(self):

        validate(self):

        __identify_single_volumes(
            self,
            study_directory: Path,
        ) -> Dict[int, DicomSingleSeries]:

        series_inference(self):
    """

    series_restrictions_list_dwi_subvolumes: list[str] = [
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

        Args:
            path_rep (Union[str, Path, PurePath]): The object to check for path-likeness.

        Returns:
            bool: True if the object is a valid path or path-like object, False otherwise.
        """
        # checks if the variable is any instance of pathlib
        if isinstance(path_rep, PurePath) or isinstance(path_rep, str):
            return True
        return False

    def __init__(
        self,
        study_directory: str | Path,
        search_series: dict[str, int] | None = None,
        inferer: ImageTypeClassifierBase | None = None,
        raise_error_on_failure: bool = False,
    ) -> None:
        """
        Initialize an instance of ProcessOneDicomStudyToVolumesMappingBase.

        This constructor sets up the object with the provided parameters, including the study directory,
        search_series dictionary, and optional image type classifier.

        Args:
            study_directory (Union[str, Path]): The path to the DICOM study directory.
            search_series (Optional[Dict[str, int]]): A dictionary of series to search within the study.
            inferer (Optional[ImageTypeClassifierBase]): An image type classifier for inference.

        Raises:
            ValueError: If the provided study_directory is not a valid path or path-like object.

        Attributes:
            study_directory (Path): The path to the DICOM study directory as a pathlib Path object.
            search_series (Optional[Dict[str, int]]): A dictionary of series to search within the study.
            series_dictionary (Dict[int, DicomSingleSeries]): A dictionary mapping series numbers to DicomSingleSeries objects.
            inferer (Optional[ImageTypeClassifierBase]): An image type classifier for inference.
        """
        if ProcessOneDicomStudyToVolumesMappingBase._is_pathlike_object(
            study_directory
        ):
            self.study_directory = Path(study_directory)  # coerce to path object
            del study_directory
        else:
            print(f"ERROR:  {self.study_directory} is not pathlike")
        self.raise_error_on_failure: bool = raise_error_on_failure
        self.search_series: dict[str, int] | None = search_series
        self.series_dictionary: dict[int, DicomSingleSeries] = (
            self.__identify_single_volumes(self.study_directory)
        )
        self.inferer: ImageTypeClassifierBase | None = inferer

    def get_list_of_primary_volume_info(self) -> list[dict[str, str]]:
        """
        Retrieve a list of dictionaries containing primary volume information from all series.

        This method iterates through the series stored in the class's series_dictionary and
        extracts primary volume information from each subseries within those series. It then
        compiles this information into a list of dictionaries, with each dictionary representing
        primary volume information for a single subseries.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing primary volume information.
                Each dictionary includes keys and values for primary volume attributes. See get_primary_volume_info function in dicom_volume.py

        """
        list_of_volume_info_dictionaries: list[dict[str, str]] = list()
        for (
            series_number,
            series_object,
        ) in self.series_dictionary.items():
            for vol_index, subseries_vol_info in enumerate(
                series_object.volume_info_list
            ):
                primary_volume_info: dict[str, str] = (
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

        Returns:
            Dict[int, DicomSingleSeries]: A dictionary where keys are series numbers (integers)
            and values are DicomSingleSeries objects containing information about each series.

        Example:
            Example usage to retrieve series information:
            study = ProcessOneDicomStudyToVolumesMappingBase(study_directory_path)
            study_dict = study.get_study_dictionary()
            series_info = study_dict.get(1)  # Retrieve information for series number 1
        """
        return self.series_dictionary

    def set_inferer(self, inferer: ImageTypeClassifierBase) -> None:
        """
        Set the image type classifier for inference.

        This method allows you to set the image type classifier (inferer) to be used for
        inference on the DICOM series. The provided `inferer` should be an instance of
        a class that derives from `ImageTypeClassifierBase`.

        Args:
            inferer (ImageTypeClassifierBase): An image type classifier object for inference.

        Example:
            Example usage to set an image type classifier:
            study = ProcessOneDicomStudyToVolumesMappingBase(study_directory_path)
            image_classifier = MyImageTypeClassifier()  # Replace with your classifier instance
            study.set_inferer(image_classifier)
        """
        self.inferer = inferer

    def run_inference(self) -> None:
        """
        Run inference on each DICOM series using the specified image type classifier.

        This method iterates through the DICOM series stored in the class's series_dictionary
        and performs inference on each series using the configured image type classifier
        (inferer). It sets the current series for inference using `set_series` and then
        invokes the `run_inference` method of the image type classifier for each series.

        Note:
            Ensure that you have configured an image type classifier (inferer) using the
            `set_inferer` method before calling this method.

        """
        for (
            series_number,
            series_object,
        ) in self.series_dictionary.items():
            self.inferer.set_series(series_object)
            self.inferer.run_inference()

    def validate(self) -> None:
        pass

    def __identify_single_volumes(
        self,
        study_directory: Path,
    ) -> dict[int, DicomSingleSeries]:
        """
        Identify and map single volumes within the DICOM study directory.

        This method scans the provided DICOM study directory for series that match the specified
        restrictions and organizes them into a dictionary, mapping series numbers to
        DicomSingleSeries objects.

        Args:
            study_directory (Path): The path to the DICOM study directory.

        Returns:
            Dict[int, DicomSingleSeries]: A dictionary mapping series numbers (integers)
            to DicomSingleSeries objects that represent the identified series.

        Notes:
            - This method uses ITK's GDCMSeriesFileNames to identify series within the study
              based on defined restrictions.
            - Series matching the restrictions are organized into DicomSingleSeries objects
              within the returned dictionary.

        Example:
            Example usage to identify and map DICOM series:
            study = ProcessOneDicomStudyToVolumesMappingBase(study_directory_path)
            series_mapping = study.__identify_single_volumes(study_directory_path)
            series_info = series_mapping.get(1)  # Retrieve information for series number 1
        """
        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        namesGenerator.SetLoadPrivateTags(True)
        namesGenerator.SetRecursive(True)
        for (
            bvalue_restrictions
        ) in (
            ProcessOneDicomStudyToVolumesMappingBase.series_restrictions_list_dwi_subvolumes
        ):
            namesGenerator.AddSeriesRestriction(bvalue_restrictions)
        # namesGenerator.AddSeriesRestriction("0008|0021")  # Date restriction
        # namesGenerator.AddSeriesRestriction("0020|0013") # For testing, SeriesInstance results in images with 1 value
        namesGenerator.SetGlobalWarningDisplay(False)
        namesGenerator.SetDirectory(study_directory.as_posix())

        seriesUID = namesGenerator.GetSeriesUIDs()

        if len(seriesUID) < 1:
            msg: str = f"No DICOMs in: {study_directory} (__identify_single_volumes)"
            if self.raise_error_on_failure:
                raise FileNotFoundError(msg)
            else:
                print(
                    f"No readable dicoms DICOMs in: {study_directory} (__identify_single_volumes)"
                )
        else:
            print(
                f"The directory: {study_directory} contains {len(seriesUID)} DICOM sub-volumes"
            )
        # print(f"Contains the following {len(seriesUID)} DICOM Series: ")
        # for uid in seriesUID:
        #     print(uid)

        volumes_dictionary: dict[int, DicomSingleSeries] = dict()

        for seriesIdentifier in seriesUID:
            # print("Reading: " + seriesIdentifier)
            subseries_filenames: list[str] = namesGenerator.GetFileNames(
                seriesIdentifier
            )
            subseries_info: DicomSingleVolumeInfoBase = DicomSingleVolumeInfoBase(
                one_volume_dcm_filenames=subseries_filenames
            )
            sn = subseries_info.get_series_number()
            if sn not in volumes_dictionary:
                volumes_dictionary[sn] = DicomSingleSeries(series_number=sn)
            volumes_dictionary[sn].add_volume_to_series(subseries_info)

            # reader = itk.ImageSeriesReader[ImageType].New()
            # dicomIO = itk.GDCMImageIO.New()
            # reader.SetImageIO(dicomIO)
            # reader.SetFileNames(fileNames)
            # reader.ForceOrthogonalDirectionOff()
            #
            # writer = itk.ImageFileWriter[ImageType].New()
            # outFileName = os.path.join(dirName, seriesIdentifier + ".nrrd")
            # if args.output_image:
            #     outFileName = args.output_image
            # writer.SetFileName(outFileName)
            # writer.UseCompressionOn()
            # writer.SetInput(reader.GetOutput())
            # print("Writing: " + outFileName)
            # writer.Update()
        # import json
        # msg=json.dumps(volumes_dictionary, indent=2)
        # print(msg)
        # volumes_dictionary = collections.defaultdict(list)
        # for _slice_loc, slice_list in slice_location_dictionary.items():
        #     for index in range(0, len(slice_list)):
        #         volumes_dictionary[index].append(slice_list[index])

        if self.search_series is not None:
            cadidate_series_numbers: list[int] = [
                int(x) for x in self.search_series.values()
            ]

            series_numbers_to_remove: list[int] = list()
            for series_number in volumes_dictionary.keys():
                if series_number not in cadidate_series_numbers:
                    series_numbers_to_remove.append(series_number)
            for series_number in series_numbers_to_remove:
                del volumes_dictionary[series_number]
            del series_numbers_to_remove
        return volumes_dictionary
