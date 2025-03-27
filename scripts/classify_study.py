#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
from typing import Any
import pandas as pd
import tabulate
from collections import OrderedDict as ordered_dict
from pathlib import Path

try:
    from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
    from dcm_classifier.image_type_inference import ImageTypeClassifierBase
    from dcm_classifier.dicom_config import required_DICOM_fields, optional_DICOM_fields
except Exception as e:
    print(f"Missing module import {e}")
    print(
        f"Try setting export PYTHONPATH={Path(__file__).parent.parent.as_posix()}/src"
    )
    sys.exit(255)


def generate_separator(column_width):
    return "|" + ("-" * (column_width + 2) + "|") * 4


def generate_row(*args, column_width):
    return "| " + " | ".join(arg.ljust(column_width) for arg in args) + " |"


def get_new_base_name(
    user_prefix: str,
    series_number: int,
    bvalue_name_str: str,
    series_modality: str,
    *,
    volume_index: int = -1,
) -> str:
    """
    The Goal of this function is to create a new base name for the output files that provide a standardized interface.
    Stealing the idea from bids we should be able to easily create a dictionary from the name of the file.
    key-value pairs are separated by underscores, and the key, values are separated by dashes.
    The "image type" is the last location and doesn't have a KEY associated with it.
    It is assumed

    Args:
        user_prefix:
        series_number:
        bvalue_name_str:
        series_modality:
        volume_index:

    Returns:

    """
    if "_" not in user_prefix[-1] and user_prefix != "":
        user_prefix += "_"  # Add an underscore if it is missing
    # TODO Make sure this is robust enough to handle all cases

    if volume_index >= 0:
        return f"{user_prefix}seriesnum-{series_number:05}_volidx-{volume_index:03}_{bvalue_name_str}{series_modality}"
    return (
        f"{user_prefix}seriesnum-{series_number:05}_{bvalue_name_str}{series_modality}"
    )


def get_dicom_bvalue_name_str(series) -> str:
    """
    Extracts and returns a b-value string formatted as bvalue-b50b400b800
    from all volumes in a series.
    """
    # TODO maybe make this check only even allow for b-values to be included if it is a valid series modality like "DWI"
    bvals = []
    for vol in series.get_volume_list():
        try:
            val = int(vol.get_volume_bvalue())
            if val >= 0:
                bvals.append(val)
        except Exception:
            continue
    unique_sorted = sorted(set(bvals))
    if unique_sorted:
        return "bvalue-" + "b" + "b".join(str(b) for b in unique_sorted) + "_"
    else:
        return ""


def main():
    # Set up argparse
    description = (
        "Authors: Michal Brzus, Hans J. Johnson\n"
        "Classify image modality and acquisition plane of DICOM MR data by series and volumes.\n"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-d",
        "--session_directory",
        required=True,
        help="Path to the patient session directory with dicom data",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=False,
        help="Path to the model used for image type inference",
    )
    parser.add_argument(
        "-n",
        "--nifti_dir",
        required=False,
        default=None,
        help="Path to the directory where the NIFTI files are stored for each volume",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=None,
        help="Path to the output the newly organized dicom data",
    )
    parser.add_argument(
        "-j",
        "--json",
        required=False,
        default=None,
        help="Path to the output json file",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix to add to the output file names for Nifti or the prefix to add to the output directory for dicom files",
        required=False,
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move the original DICOM files rather than copying them",
    )  # TODO Maybe don't need this option in the future but need it for Bot currently

    args = parser.parse_args()
    prefix = args.prefix
    nifti_dir: Path | None = Path(args.nifti_dir) if args.nifti_dir else None
    if nifti_dir:
        import itk

        nifti_dir.mkdir(parents=True, exist_ok=True)

    output_dir: Path | None = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(description)

    if args.model is None:
        inferer = ImageTypeClassifierBase()
    else:
        inferer = ImageTypeClassifierBase(classification_model_filename=args.model)
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=args.session_directory, inferer=inferer
    )
    study.run_inference()

    list_of_inputs: list[dict[str, Any]] = []
    list_of_probabilities: list[pd.DataFrame] = []
    list_of_dictionaries: list[dict[str, str]] = []
    session_dictionary: dict[str, Any] = {}

    for series_number, series in study.series_dictionary.items():
        series_modality = str(series.get_series_modality())
        bvalue_name_str = get_dicom_bvalue_name_str(series)
        # Volume-level loop for metadata and nifti output
        for index, volume in enumerate(series.get_volume_list()):
            invalid_volume: bool = False
            current_dict: dict[str, str] = ordered_dict()
            current_dict["Series#"] = str(series_number)
            dict_entry_name: str | None = None
            dictionary = {}

            try:
                vol_index: str = str(volume.get_volume_index())
                current_dict["Vol.#"] = vol_index
                dict_entry_name: str = f"{series_number}_{vol_index}"
            except AttributeError:
                current_dict["Vol.#"] = "None"

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

            current_dict["Series Modality"] = series_modality
            dictionary["SeriesModality"] = series_modality if dictionary else {}

            try:
                acq_plane: str = str(volume.get_acquisition_plane())
                current_dict["Acq.Plane"] = acq_plane
                dictionary["AcqPlane"] = acq_plane if dictionary else {}
            except AttributeError:
                current_dict["Acq.Plane"] = "None"
                dictionary = {}

            try:
                isotropic: str = str(volume.get_is_isotropic())
                current_dict["Isotropic"] = isotropic
                dictionary["Isotropic"] = isotropic if dictionary else {}
            except AttributeError:
                current_dict["Isotropic"] = "None"
                dictionary["Isotropic"] = "None"

            vol_probabilities = volume.get_modality_probabilities()
            for col in vol_probabilities.columns:
                if "SeriesNumber" in col or "CODE" in col:
                    continue
                dictionary[col] = (
                    str(vol_probabilities[col].values[0]) if dictionary else {}
                )
            print(vol_probabilities.to_string(index=False))

            try:
                bval = str(volume.get_volume_bvalue())
                current_dict["Bvalue"] = bval
                dictionary["Bvalue"] = bval if dictionary else {}
            except AttributeError:
                current_dict["Bvalue"] = "None"
                dictionary["Bvalue"] = "None"

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

            try:
                contrast = volume.get_has_contrast()
                dictionary["Contrast"] = str(contrast) if dictionary else {}
                if contrast:
                    contrast_agent = volume.get_contrast_agent()
                    dictionary["ContrastAgent"] = contrast_agent if dictionary else {}
                else:
                    dictionary["Contrast"] = "None"
            except AttributeError:
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
            list_of_probabilities.append(volume.get_modality_probabilities())
            list_of_dictionaries.append(current_dict)
            if nifti_dir is not None:
                nifti_file_base_name = get_new_base_name(
                    prefix,
                    series_number,
                    bvalue_name_str,
                    series_modality,
                    volume_index=index,
                )  # THIS HAS to have the volume index to be unique or you will overwrite
                image_file_name: str = f"{nifti_file_base_name}.nii.gz"
                itk_image = volume.get_itk_image()
                itk.imwrite(itk_image, nifti_dir / image_file_name)

                # Make output directory once per series
            if output_dir is not None:
                # TODO Allow for changing this to be a per volume output
                SPLIT_DCM_OUTPUT_BY_VOLUME = False
                if SPLIT_DCM_OUTPUT_BY_VOLUME:
                    dcm_output_dir_name: str = get_new_base_name(
                        prefix,
                        series_number,
                        bvalue_name_str,
                        series_modality,
                        volume_index=index,
                    )
                else:
                    dcm_output_dir_name: str = get_new_base_name(
                        prefix,
                        series_number,
                        bvalue_name_str,
                        series_modality,
                    )
                # TODO Allow for changing this via command line
                output_dir_path: Path = output_dir / dcm_output_dir_name
                output_dir_path.mkdir(parents=True, exist_ok=True)
                for dcm_file in volume.one_volume_dcm_filenames:
                    output_file_path: Path = output_dir_path / dcm_file.name
                    print(f"Copying {dcm_file} to {output_file_path}")
                    if output_file_path.exists():
                        if output_file_path.absolute() != dcm_file.absolute():
                            print(f"File {output_file_path} already exists, skipping")
                            raise FileExistsError(
                                f"File {output_file_path} already exists"
                            )  # TODO Handle this more gracefully
                    else:
                        if args.move:
                            shutil.move(dcm_file, output_file_path)
                        else:
                            shutil.copy(
                                dcm_file, output_file_path, follow_symlinks=True
                            )
            if dict_entry_name is not None:
                session_dictionary[dict_entry_name] = (
                    dictionary if not invalid_volume else {}  # set to empty if invalid
                )

    json_output_dict: dict[str, Any] = {str(args.session_directory): session_dictionary}

    # save the dictionary to a file
    if args.json is not None:
        with open(args.json, "w") as f:
            json.dump(json_output_dict, f, indent=4)

    df: pd.DataFrame = pd.DataFrame(list_of_dictionaries)
    df.sort_values(by=["Series#", "Vol.#"], inplace=True)

    table_msg: str = tabulate.tabulate(
        df, headers="keys", tablefmt="psql", showindex=False
    )
    print(table_msg)

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


if __name__ == "__main__":
    # Execute the script
    main()
