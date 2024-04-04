#!/usr/bin/env python3

import argparse
import sys
from typing import Any
import pandas as pd
import tabulate
from collections import OrderedDict as ordered_dict
from pathlib import Path

try:
    from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
    from dcm_classifier.image_type_inference import ImageTypeClassifierBase
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

    args = parser.parse_args()

    nifti_dir: Path | None = Path(args.nifti_dir) if args.nifti_dir else None
    if nifti_dir:
        import itk

        nifti_dir.mkdir(parents=True, exist_ok=True)

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

    for series_number, series in study.series_dictionary.items():
        for index, volume in enumerate(series.get_volume_list()):
            current_dict: dict[str, str] = ordered_dict()
            current_dict["Series#"] = str(series_number)
            current_dict["Vol.#"] = str(volume.get_volume_index())
            current_dict["Volume Modality"] = str(volume.get_volume_modality())
            current_dict["Series Modality"] = str(series.get_series_modality())
            current_dict["Acq.Plane"] = str(volume.get_acquisition_plane())
            current_dict["Isotropic"] = str(volume.get_is_isotropic())
            print(volume.get_modality_probabilities().to_string(index=False))
            current_dict["Bvalue"] = str(volume.get_volume_bvalue())
            try:
                current_dict["SeriesDesc"] = volume.get_volume_series_description()
            except AttributeError:
                current_dict["SeriesDesc"] = "None"
            inputs_df: dict[str, Any] = volume.get_volume_dictionary()
            current_dict["ImageType"] = str(inputs_df.get("ImageType", "Unknown"))
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
