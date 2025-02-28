#!/usr/bin/env python3
"""
Authors: Michal Brzus, Hans J. Johnson, Ivan Johnson-Eversoll.

This script classifies image modality and acquisition plane of DICOM MR data by series
and volumes. It supports handling multiple study directories in a single run and saves
the results to a master DataFrame (CSV). It also retains the ability to output DICOM
files to a specified output directory and optionally write NIFTI volumes for each study.

Usage:
  python3 multi_study_processor.py \
    -d /path/to/study1 /path/to/study2 ... \
    [-m /path/to/model] \
    [-n /path/to/nifti_output_dir] \
    [-o /path/to/dicom_output_dir] \
    [-r /path/to/results_csv]

Required Arguments:
  -d, --session_directories
    A list of one or more paths to patient sessions that contain DICOM data.

Optional Arguments:
  -m, --model
    Path to the model used for image type inference. If not provided, the base classifier
    with default model is used.
  -n, --nifti_dir
    Path to the directory where the NIFTI files will be stored for each volume.
  -o, --output
    Path to the directory where the newly organized DICOM data will be copied.
  -r, --results_csv
    Path to a CSV file where the concatenated classification results for all studies
    will be saved (default: ./multi_study_results.csv).
"""

import argparse
import shutil
import sys
import pandas as pd
import tabulate
from collections import OrderedDict as ordered_dict
from pathlib import Path
from itk import imwrite

try:
    from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
    from dcm_classifier.image_type_inference import ImageTypeClassifierBase
except Exception as e:
    print(f"Missing module import {e}")
    print(
        f"Try setting export PYTHONPATH={Path(__file__).parent.parent.as_posix()}/src"
    )
    sys.exit(255)


def process_single_study(
    session_directory: Path,
    inferer: ImageTypeClassifierBase,
    nifti_dir: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Process a single study directory, classify volumes, save optional NIFTI outputs,
    copy DICOMs to organized directories, and return a DataFrame summarizing the results.
    """

    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=session_directory, inferer=inferer
    )
    study.run_inference()

    list_of_dictionaries: list[dict[str, str]] = []

    for series_number, series in study.series_dictionary.items():
        for index, volume in enumerate(series.get_volume_list()):
            current_dict: dict[str, str] = ordered_dict()
            # current_dict["StudyPath"] = str(session_directory)
            current_dict["Series#"] = str(series_number)
            try:
                current_dict["Vol.#"] = str(volume.get_volume_index())
            except AttributeError:
                current_dict["Vol.#"] = "None"
            try:
                current_dict["Volume Modality"] = str(volume.get_volume_modality())
            except AttributeError:
                current_dict["Volume Modality"] = "None"
            try:
                current_dict["Series Modality"] = str(series.get_series_modality())
            except AttributeError:
                current_dict["Series Modality"] = "None"
            try:
                current_dict["Acq.Plane"] = str(volume.get_acquisition_plane())
            except AttributeError:
                current_dict["Acq.Plane"] = "None"
            try:
                current_dict["Isotropic"] = str(volume.get_is_isotropic())
            except AttributeError:
                current_dict["Isotropic"] = "None"

            vol_probabilities = volume.get_modality_probabilities()
            # Print probability DataFrame to console for debugging/inspection
            print(vol_probabilities.to_string(index=False))

            try:
                current_dict["Bvalue"] = str(volume.get_volume_bvalue())
            except AttributeError:
                current_dict["Bvalue"] = "None"
            try:
                current_dict["SeriesDesc"] = volume.get_dicom_field_by_name(
                    "SeriesDescription"
                )
            except AttributeError:
                current_dict["SeriesDesc"] = "None"

            # We'll store some volume info in case needed
            volume_dict = volume.get_volume_dictionary()
            current_dict["ImageType"] = str(volume_dict.get("ImageType", "Unknown"))

            # Save NIFTI file if requested
            if nifti_dir is not None:
                bvalue_suffix: str = (
                    f"_b{volume.get_volume_bvalue()}"
                    if volume.get_volume_bvalue() >= 0
                    else ""
                )
                image_file_name: str = (
                    f"{series_number:04}_{volume.get_volume_index():03}"
                    f"_{series.get_series_modality()}{bvalue_suffix}.nii.gz"
                )
                itk_image = volume.get_itk_image()
                imwrite(itk_image, nifti_dir / image_file_name)

            # Copy out DICOM files if requested
            if output_dir is not None:
                bvalue_suffix: str = (
                    f"_b{volume.get_volume_bvalue()}"
                    if volume.get_volume_bvalue() >= 0
                    else ""
                )
                dcm_output_dir_name: str = (
                    f"{series_number:04}_{volume.get_volume_index():03}"
                    f"_{series.get_series_modality()}{bvalue_suffix}"
                )
                output_dir_path: Path = output_dir / dcm_output_dir_name
                output_dir_path.mkdir(parents=True, exist_ok=True)
                for dcm_file in volume.one_volume_dcm_filenames:
                    output_file_path: Path = output_dir_path / dcm_file.name
                    print(f"Copying {dcm_file} to {output_file_path}")
                    shutil.copy(dcm_file, output_file_path, follow_symlinks=True)

            list_of_dictionaries.append(current_dict)

    # Create a DataFrame for the single study
    print(f"Finished processing study at: {session_directory}")
    df_study: pd.DataFrame = pd.DataFrame(list_of_dictionaries)
    df_study.sort_values(by=["Series#", "Vol.#"], inplace=True)

    # Print a nice summary table
    table_msg: str = tabulate.tabulate(
        df_study, headers="keys", tablefmt="psql", showindex=False
    )
    print(table_msg)

    return df_study


def main() -> None:
    description = (
        "Authors: Michal Brzus, Hans J. Johnson\n"
        "Modifications for multi-study handling by Ivan Johnson-Eversoll\n"
        "Classify image modality and acquisition plane of DICOM MR data by series and volumes.\n"
        "Supports multiple DICOM study directories in one run."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-d",
        "--session_directories",
        nargs="+",
        required=True,
        help="One or more paths to patient session directories with dicom data.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=False,
        help="Path to the model used for image type inference (optional).",
    )
    parser.add_argument(
        "-n",
        "--nifti_dir",
        required=False,
        default=None,
        help="Directory where the NIFTI files are stored for each volume (optional).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=None,
        help="Directory to output the newly organized dicom data (optional).",
    )
    parser.add_argument(
        "-r",
        "--results_csv",
        required=False,
        default="multi_study_results.csv",
        help="Path to the CSV file to store classification results for all studies.",
    )

    args = parser.parse_args()

    nifti_dir: Path | None = Path(args.nifti_dir) if args.nifti_dir else None
    if nifti_dir:
        nifti_dir.mkdir(parents=True, exist_ok=True)

    output_dir: Path | None = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    results_csv: Path = Path(args.results_csv)

    print(description)
    if args.model is None:
        inferer = ImageTypeClassifierBase()
    else:
        inferer = ImageTypeClassifierBase(classification_model_filename=args.model)

    # Process each study directory and collect results
    all_studies_df_list = []
    for study_path_str in args.session_directories:
        study_path = Path(study_path_str)
        if not study_path.is_dir():
            print(f"WARNING: {study_path} is not a valid directory. Skipping.")
            continue

        print(f"\n--- Processing study at: {study_path} ---")
        df_study = process_single_study(
            session_directory=study_path,
            inferer=inferer,
            nifti_dir=nifti_dir / study_path.name if nifti_dir else None,
            output_dir=output_dir / study_path.name if output_dir else None,
        )
        df_study["StudyPath"] = str(study_path)
        df_study["StudyPathName"] = study_path.name
        all_studies_df_list.append(df_study)

    if not all_studies_df_list:
        print("No valid study directories found or processed. Exiting.")
        sys.exit(0)

    # Concatenate all results into one master DataFrame
    master_df = pd.concat(all_studies_df_list, ignore_index=True)
    master_df.sort_values(by=["StudyPathName", "Series#", "Vol.#"], inplace=True)

    # Save the master DataFrame to CSV
    master_df.to_csv(results_csv, index=False)
    print(f"\nMaster results saved to: {results_csv}")

    # Optionally, print the master summary to console
    printable_df = master_df.copy()
    printable_df.drop(columns=["StudyPath"], inplace=True)

    table_msg: str = tabulate.tabulate(
        printable_df,
        headers="keys",
        tablefmt="pretty",
        showindex=False,
        numalign="center",
    )
    print("\n=== Final Master Table ===")
    print(table_msg)


if __name__ == "__main__":
    main()
