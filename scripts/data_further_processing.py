#!/usr/bin/env python3
"""
Prepare data for training a new classifier from image content.

For every series of every study directory below the data directory, writes

    <output>/<StudyInstanceUID>/<SeriesNumber>_<SeriesInstanceUID>/
        <SeriesModality>_<SeriesInstanceUID>.json
        <VolumeModality>_<SeriesInstanceUID>_<VolumeIndex:03>.nii.gz

The JSON file has one entry per sub-volume with its image type, the
features the current model classified it from, and the class probabilities.

Example:
    python3 scripts/data_further_processing.py -d <studies_directory> -o <output_directory>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import itk

try:
    from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
    from dcm_classifier.image_type_inference import ImageTypeClassifierBase
    from dcm_classifier.dicom_config import inference_features
except Exception as e:
    print(f"Missing module import {e}")
    print(
        f"Try setting export PYTHONPATH={Path(__file__).parent.parent.as_posix()}/src"
    )
    sys.exit(255)

non_class_probability_columns: tuple[str, ...] = (
    "GUESS_ONNX",
    "GUESS_ONNX_CODE",
    "SeriesNumber",
)


def to_json_value(value: Any) -> Any:
    """Convert numpy scalars, and anything else json cannot write, to plain values."""
    return value.item() if hasattr(value, "item") else str(value)


def create_subvolume_entry(volume) -> dict[str, Any]:
    """
    Describe one sub-volume for the series JSON file.

    :param volume: A classified DicomSingleVolumeInfoBase.
    :return: The image type, the inference features, and the class probabilities.
    """
    volume_dictionary: dict[str, Any] = volume.get_volume_dictionary()
    probabilities = volume.get_modality_probabilities()
    return {
        "ImageType": volume.get_volume_modality(),
        "TrainingData": {
            key: value
            for key, value in volume_dictionary.items()
            if key in inference_features
        },
        "OutputProbabilities": (
            {}
            if probabilities is None
            else {
                column.removeprefix("GUESS_ONNX_"): probabilities[column].iloc[0]
                for column in probabilities.columns
                if column not in non_class_probability_columns
            }
        ),
    }


def prepare_study(
    study_directory: Path,
    output_base_directory: Path,
    inferer: ImageTypeClassifierBase,
) -> list[str]:
    """
    Classify one study and write the JSON and NIfTI files for each of its series.

    A sub-volume whose image cannot be written is left out of the JSON file, so every
    JSON entry has a NIfTI file. A series whose output directory already exists is
    skipped, so a second copy of a series never overwrites the first.

    :param study_directory: Directory with the DICOM files of one study.
    :param output_base_directory: Directory below which the study output is written.
    :param inferer: Classifier used to label each sub-volume.
    :return: One message per series or sub-volume that was not written.
    """
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=study_directory, inferer=inferer
    )
    study.run_inference()

    problems: list[str] = []
    for series_number, series in study.series_dictionary.items():
        series_uid: str = series.get_series_uid()
        output_directory: Path = (
            output_base_directory
            / series.get_study_uid()
            / f"{series_number}_{series_uid}"
        )
        if output_directory.exists():
            problems.append(f"Series already prepared, skipped: {output_directory}")
            continue
        output_directory.mkdir(parents=True)

        sub_volumes_data: dict[int, dict[str, Any]] = {}
        for volume in series.get_volume_list():
            volume_index: int = volume.get_volume_index()
            nifti_file: Path = (
                output_directory
                / f"{volume.get_volume_modality()}_{series_uid}_{volume_index:03}.nii.gz"
            )
            try:
                itk.imwrite(volume.get_itk_image(), nifti_file)
            except Exception as e:
                problems.append(f"No image written for {nifti_file}: {e}")
                continue
            sub_volumes_data[volume_index] = create_subvolume_entry(volume)

        json_file: Path = (
            output_directory / f"{series.get_series_modality()}_{series_uid}.json"
        )
        with open(json_file, "w") as f:
            json.dump(sub_volumes_data, f, indent=4, default=to_json_value)
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "-d",
        "--data_directory",
        required=True,
        help="Path to a directory with one sub-directory of DICOM data per study",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=False,
        help="Path to the model used for image type inference",
    )
    parser.add_argument(
        "-o",
        "--output_base_directory",
        required=True,
        help="Path to the directory where the output files are stored",
    )
    args = parser.parse_args(argv)

    if args.model is None:
        inferer = ImageTypeClassifierBase()
    else:
        inferer = ImageTypeClassifierBase(classification_model_filename=args.model)
    output_base_directory = Path(args.output_base_directory)

    problems: list[str] = []
    for study_directory in sorted(Path(args.data_directory).iterdir()):
        if not study_directory.is_dir():
            continue
        try:
            problems += prepare_study(study_directory, output_base_directory, inferer)
        except Exception as e:
            problems.append(f"Study {study_directory} could not be prepared: {e}")
    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} studies, series, or sub-volumes were not prepared")


if __name__ == "__main__":
    main()
