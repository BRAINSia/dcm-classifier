import json
import logging
import shutil
from collections import OrderedDict as ordered_dict
from pathlib import Path
from typing import Any, cast

import itk
import pandas as pd
import tabulate

from dcm_classifier.dicom_config import optional_DICOM_fields, required_DICOM_fields
from dcm_classifier.dicom_volume import DicomSingleVolumeInfoBase
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase

logger = logging.getLogger(__name__)


def _classification_failure_reason(volume: DicomSingleVolumeInfoBase) -> str:
    if not volume.get_volume_dictionary():
        return "Required DICOM fields are missing or invalid"
    return "Inference features are missing or invalid"


def simple_classify_study(
    session_directory: str | Path,
    inferer: ImageTypeClassifierBase | None = None,
    json_dumppath: Path | None = None,
    output_dir: Path | None = None,
    nifti_dir: Path | None = None,
    mr_sop_class_only: bool = False,
    allow_fallback_ordering: bool = False,
) -> dict[str, Any]:
    """

    Args:
        session_directory: Path to the patient session directory with dicom data
        inferer: Classifier used for image type inference, created when omitted
        json_dumppath: Path to the output json file
        output_dir: Path to the output the newly organized dicom data or None to avoid organizing dicom data
        nifti_dir: Path to the directory where the NIFTI files are stored for each volume
        mr_sop_class_only: Only process series whose first file has an MR SOP Class UID
        allow_fallback_ordering: Accept volumes whose slices could not be ordered by position

    Returns:

    """
    if inferer is None:
        inferer = ImageTypeClassifierBase()
    if nifti_dir:
        nifti_dir.mkdir(parents=True, exist_ok=True)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=session_directory,
        inferer=inferer,
        mr_sop_class_only=mr_sop_class_only,
        allow_fallback_ordering=allow_fallback_ordering,
    )
    study.run_inference()
    list_of_inputs: list[dict[str, Any]] = []
    list_of_probabilities: list[pd.DataFrame | None] = []
    list_of_dictionaries: list[dict[str, str]] = []
    session_dictionary: dict[str, Any] = {}
    for series_number, series in study.series_dictionary.items():
        for volume in series.get_volume_list():
            invalid_volume: bool = False
            current_dict: dict[str, str] = ordered_dict()
            current_dict["Series#"] = str(series_number)
            dict_entry_name: str | None = None
            dictionary = {}

            # Volume Index
            try:
                vol_index: str = str(volume.get_volume_index())
                current_dict["Vol.#"] = vol_index
                dict_entry_name = f"{series_number}_{vol_index}"
            except AttributeError:
                current_dict["Vol.#"] = "None"

            # Volume Modality
            try:
                vol_modality: str = str(volume.get_volume_modality())
                current_dict["Volume Modality"] = vol_modality
                invalid_volume = vol_modality == "INVALID"
            except AttributeError:
                vol_modality = "None"
                current_dict["Volume Modality"] = "None"
            dictionary["VolumeModality"] = vol_modality
            slice_ordering = str(volume.get_slice_ordering())
            dictionary["SliceOrdering"] = slice_ordering
            current_dict["Order"] = slice_ordering
            if invalid_volume:
                dictionary["ClassificationFailureReason"] = (
                    _classification_failure_reason(volume)
                )

            # Series Modality
            try:
                series_modality: str = str(series.get_series_modality())
                current_dict["Series Modality"] = series_modality
                dictionary["SeriesModality"] = series_modality
            except AttributeError:
                current_dict["Series Modality"] = "None"
                dictionary["SeriesModality"] = "None"

            # Acquisition Plane
            try:
                acq_plane: str = str(volume.get_acquisition_plane())
                current_dict["Acq.Plane"] = acq_plane
                dictionary["AcqPlane"] = acq_plane
            except AttributeError:
                current_dict["Acq.Plane"] = "None"
                dictionary["AcqPlane"] = "None"

            # Isotropic
            try:
                isotropic: str = str(volume.get_is_isotropic())
                current_dict["Isotropic"] = isotropic
                dictionary["Isotropic"] = isotropic
            except AttributeError:
                current_dict["Isotropic"] = "None"
                dictionary["Isotropic"] = "None"

            # Modality Probabilities
            vol_probabilities = volume.get_modality_probabilities()
            if vol_probabilities is not None:
                if not invalid_volume:
                    for col in vol_probabilities.columns:
                        if "SeriesNumber" in col or "CODE" in col:
                            continue
                        dictionary[col] = str(vol_probabilities[col].to_numpy()[0])
                logger.info("%s", vol_probabilities.to_string(index=False))

            # Bvalue
            try:
                bval = str(volume.get_volume_bvalue())
                current_dict["Bvalue"] = bval
                dictionary["Bvalue"] = bval
            except AttributeError:
                current_dict["Bvalue"] = "None"
                dictionary["Bvalue"] = "None"

            # Series Description
            try:
                series_description: str = volume.get_dicom_field_by_name(
                    "SeriesDescription"
                )
                current_dict["SeriesDesc"] = series_description
                dictionary["SeriesDescription"] = series_description
            except AttributeError:
                current_dict["SeriesDesc"] = "None"
                dictionary["SeriesDescription"] = "None"

            # Contrast Agent
            try:
                contrast = volume.get_has_contrast()
                if contrast:
                    dictionary["Contrast"] = str(contrast)
                    dictionary["ContrastAgent"] = volume.get_contrast_agent()
                else:
                    dictionary["Contrast"] = "None"
            except AttributeError:
                dictionary["Contrast"] = "None"

            # Pixel Spacing
            try:
                pixel_spacing: str | list[str] = volume.get_dicom_field_by_name(
                    "PixelSpacing"
                )
                if isinstance(pixel_spacing, str):
                    pixel_spacing = [pixel_spacing, pixel_spacing]
                dictionary["PixelSpacing_0"] = str(pixel_spacing[0])
                dictionary["PixelSpacing_1"] = str(pixel_spacing[1])
            except AttributeError:
                dictionary["PixelSpacing_0"] = "None"
                dictionary["PixelSpacing_1"] = "None"

            for field in required_DICOM_fields + optional_DICOM_fields:
                if field in ["PixelSpacing", "ImageType"]:
                    continue
                try:
                    value = volume.get_dicom_field_by_name(field)
                    dictionary[field] = str(value)
                except AttributeError:
                    dictionary[field] = "None"

            inputs_df: dict[str, Any] = volume.get_volume_dictionary()
            current_dict["ImageType"] = str(inputs_df.get("ImageType", "Unknown"))
            dictionary["ImageType"] = str(inputs_df.get("ImageType", "Unknown"))
            for unwanted in [
                "FileName",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "list_of_ordered_volume_files",
            ]:
                if unwanted in inputs_df:
                    inputs_df.pop(unwanted)

            list_of_inputs.append(inputs_df)

            prob_df: pd.DataFrame | None = volume.get_modality_probabilities()
            list_of_probabilities.append(prob_df)
            list_of_dictionaries.append(current_dict)
            if nifti_dir is not None:
                bvalue_suffix: str = (
                    f"_b{volume.get_volume_bvalue()}"
                    if volume.get_volume_bvalue() >= 0
                    else ""
                )
                image_file_name: str = (
                    f"{series_number:04}_{volume.get_volume_index():03}"
                    f"_{volume.get_series_modality()}{bvalue_suffix}.nii.gz"
                )
                itk_image = volume.get_itk_image()
                itk.imwrite(itk_image, nifti_dir / image_file_name)
            if output_dir is not None:
                bvalue_suffix = (
                    f"_b{volume.get_volume_bvalue()}"
                    if volume.get_volume_bvalue() >= 0
                    else ""
                )
                dcm_output_dir_name: str = (
                    f"{series_number:04}_{volume.get_volume_index():03}"
                    f"_{volume.get_series_modality()}{bvalue_suffix}"
                )
                output_dir_path: Path = output_dir / dcm_output_dir_name
                output_dir_path.mkdir(parents=True, exist_ok=True)
                for dcm_file in volume.one_volume_dcm_filenames:
                    output_file_path: Path = output_dir_path / dcm_file.name
                    logger.info("Copying %s to %s", dcm_file, output_file_path)
                    shutil.copy(dcm_file, output_file_path, follow_symlinks=True)
            if dict_entry_name is not None:
                session_dictionary[dict_entry_name] = dictionary
    json_output_dict: dict[str, Any] = {str(session_directory): session_dictionary}
    # save the dictionary to a file
    if json_dumppath is not None:
        with Path(json_dumppath).open("w") as f:
            json.dump(json_output_dict, f, indent=4, sort_keys=True)

    if len(list_of_dictionaries) > 0:  # Check if the list is not empty
        df: pd.DataFrame = pd.DataFrame(list_of_dictionaries)
        try:
            df = df.sort_values(by=["Series#", "Vol.#"])
            # tabulate accepts a DataFrame, but its stubs do not say so.
            table_msg: str = tabulate.tabulate(
                cast("Any", df), headers="keys", tablefmt="psql", showindex=False
            )
            logger.info("%s", table_msg)
        except KeyError:
            logger.warning("No series found in the study.")
    else:
        logger.warning("No series found in the study.")
    # all_inputs_df: pd.DataFrame = pd.DataFrame(list_of_inputs)
    # inputs_msg: str = tabulate.tabulate(
    #     all_inputs_df, headers="keys", tablefmt="psql", showindex=False
    # )
    # print(inputs_msg)
    #
    # all_prob_df: pd.DataFrame = pd.concat(list_of_probabilities, axis=0, ignore_index=True)
    # prob_msg: str = tabulate.tabulate(
    #     all_prob_df, headers="keys", tablefmt="psql", showindex=False
    # )
    # print(prob_msg)
    return json_output_dict
