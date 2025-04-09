import json
import shutil
from collections import OrderedDict as ordered_dict
from pathlib import Path
from typing import Any

import pandas as pd
import tabulate

from dcm_classifier.dicom_config import required_DICOM_fields, optional_DICOM_fields
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase


def simple_classify_study(
    session_directory: Path | None = None,
    inferer: ImageTypeClassifierBase = ImageTypeClassifierBase(),
    json_dumppath: Path | None = None,
    output_dir: Path | None = None,
    nifti_dir: Path | None = None,
) -> dict[str, Any]:
    """

    Args:
        session_directory: Path to the patient session directory with dicom data
        inferer: Path to the model used for image type inference
        json_dumppath: Path to the output json file
        output_dir: Path to the output the newly organized dicom data or None to avoid organizing dicom data
        nifti_dir: Path to the directory where the NIFTI files are stored for each volume

    Returns:

    """
    if nifti_dir:
        import itk

        nifti_dir.mkdir(parents=True, exist_ok=True)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=session_directory, inferer=inferer
    )
    study.run_inference()
    list_of_inputs: list[dict[str, Any]] = []
    list_of_probabilities: list[pd.DataFrame] = []
    list_of_dictionaries: list[dict[str, str]] = []
    session_dictionary: dict[str, Any] = {}
    for series_number, series in study.series_dictionary.items():
        for index, volume in enumerate(series.get_volume_list()):
            invalid_volume: bool = False
            current_dict: dict[str, str] = ordered_dict()
            current_dict["Series#"] = str(series_number)
            dict_entry_name: str | None = None
            dictionary = {}

            # Volume Index
            try:
                vol_index: str = str(volume.get_volume_index())
                current_dict["Vol.#"] = vol_index
                dict_entry_name: str = f"{series_number}_{vol_index}"
            except AttributeError:
                current_dict["Vol.#"] = "None"

            # Volume Modality
            try:
                vol_modality: str = str(volume.get_volume_modality())
                current_dict["Volume Modality"] = vol_modality
                if vol_modality != "INVALID":
                    dictionary["VolumeModality"] = vol_modality
                else:
                    invalid_volume = True
            except AttributeError:
                current_dict["Volume Modality"] = "None"
                dictionary = {}

            # Series Modality
            try:
                series_modality: str = str(series.get_series_modality())
                current_dict["Series Modality"] = series_modality
                dictionary["SeriesModality"] = series_modality if dictionary else {}
            except AttributeError:
                current_dict["Series Modality"] = "None"
                dictionary = {}

            # Acquisition Plane
            try:
                acq_plane: str = str(volume.get_acquisition_plane())
                current_dict["Acq.Plane"] = acq_plane
                dictionary["AcqPlane"] = acq_plane if dictionary else {}
            except AttributeError:
                current_dict["Acq.Plane"] = "None"
                dictionary = {}

            # Isotropic
            try:
                isotropic: str = str(volume.get_is_isotropic())
                current_dict["Isotropic"] = isotropic
                dictionary["Isotropic"] = isotropic if dictionary else {}
            except AttributeError:
                current_dict["Isotropic"] = "None"
                dictionary["Isotropic"] = "None"

            # Modality Probabilities
            vol_probabilities = volume.get_modality_probabilities()
            for col in vol_probabilities.columns:
                # current_dict[col] = str(vol_probabilities[col].values[0])
                if "SeriesNumber" in col or "CODE" in col:
                    continue
                dictionary[col] = (
                    str(vol_probabilities[col].values[0]) if dictionary else {}
                )
            print(vol_probabilities.to_string(index=False))

            # Bvalue
            try:
                bval = str(volume.get_volume_bvalue())
                current_dict["Bvalue"] = bval
                dictionary["Bvalue"] = bval if dictionary else {}
            except AttributeError:
                current_dict["Bvalue"] = "None"
                dictionary["Bvalue"] = "None"

            # Series Description
            try:
                series_description: str = volume.get_dicom_field_by_name(
                    "SeriesDescription"
                )
                current_dict["SeriesDesc"] = series_description
                dictionary["SeriesDescription"] = (
                    series_description if dictionary else {}
                )
            except AttributeError:
                current_dict["SeriesDesc"] = "None"
                dictionary["SeriesDescription"] = "None"

            # Contrast Agent
            try:
                contrast = volume.get_has_contrast()
                dictionary["Contrast"] = str(contrast) if dictionary else {}

                if contrast:
                    contrast_agent = volume.get_contrast_agent()
                    # current_dict["ContrastAgent"] = contrast_agent
                    dictionary["ContrastAgent"] = contrast_agent if dictionary else {}
                else:
                    # current_dict["Contrast"] = "None"
                    dictionary["Contrast"] = "None"
            except AttributeError:
                # current_dict["Contrast"] = "None"
                dictionary["Contrast"] = "None"

            # Pixel Spacing
            try:
                pixel_spacing = volume.get_dicom_field_by_name("PixelSpacing")
                # current_dict["PixelSpacing"] = pixel_spacing
                dictionary["PixelSpacing_0"] = (
                    str(pixel_spacing[0]) if dictionary else {}
                )
                dictionary["PixelSpacing_1"] = (
                    str(pixel_spacing[1]) if dictionary else {}
                )
            except AttributeError:
                # current_dict["PixelSpacing"] = "None"
                dictionary["PixelSpacing_0"] = "None"
                dictionary["PixelSpacing_1"] = "None"

            for field in required_DICOM_fields + optional_DICOM_fields:
                if field in ["PixelSpacing", "ImageType"]:
                    continue
                try:
                    value = volume.get_dicom_field_by_name(field)
                    # current_dict[field] = str(value)
                    dictionary[field] = str(value) if dictionary else {}
                except AttributeError:
                    # current_dict[field] = "None"
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

            prob_df: pd.DataFrame = volume.get_modality_probabilities()
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
                bvalue_suffix: str = (
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
                    print(f"Copying {dcm_file} to {output_file_path}")
                    shutil.copy(dcm_file, output_file_path, follow_symlinks=True)
                    # shutil.move(dcm_file, output_file_path)
            if dict_entry_name is not None:
                session_dictionary[dict_entry_name] = (
                    dictionary if not invalid_volume else {}  # set to empty if invalid
                )
    json_output_dict: dict[str, Any] = {str(session_directory): session_dictionary}
    # save the dictionary to a file
    if json_dumppath is not None:
        with open(json_dumppath, "w") as f:
            json.dump(json_output_dict, f, indent=4, sort_keys=True)

    if len(list_of_dictionaries) > 0:  # Check if the list is not empty
        df: pd.DataFrame = pd.DataFrame(list_of_dictionaries)
        try:
            df.sort_values(by=["Series#", "Vol.#"], inplace=True)
            table_msg: str = tabulate.tabulate(
                df, headers="keys", tablefmt="psql", showindex=False
            )
            print(table_msg)
        except KeyError:
            print("No series found in the study.")
    else:
        print("No series found in the study.")
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
