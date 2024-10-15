#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
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

# TODO: example command for running this script adapted from classify_study.py:
#   python3 scripts/data_further_processing.py -d <path_to_input_directory> -o <path_to_output_base_directory>


def write_json(subvolumes_data, output_path):
    """
    Writes a JSON file based on subvolumes data.

    :param subvolumes_data: A dictionary where keys are subvolume numbers,
                            and values are dictionaries containing the subvolume data.
    :param output_path: Path to save the generated JSON file.
    """
    with open(output_path, 'w') as json_file:
        json.dump(subvolumes_data, json_file, indent=4)


def create_subvolume_entry(subvolume_number, image_type, training_data: dict, output_probabilities: dict):
    """
    Creates a dictionary entry for a single subvolume.

    :param subvolume_number: The subvolume number (key in JSON).
    :param image_type: The image type (e.g., "t1w").
    :param training_data: A dictionary of training data for the subvolume.
    :param output_probabilities: A dictionary of output probabilities for different image types.
    :return: A dictionary representing the subvolume data.
    """
    return {
        subvolume_number: {
            "ImageType": image_type,
            "TrainingData": training_data,
            "OutputProbabilities": output_probabilities
        }
    }


def main():
    # Set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_directory",
        required=True,
        help="Path to the dicom data directory",
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
        required=False,
        default=None,
        help="Path to the directory where the the output files are stored",
    )

    args = parser.parse_args()

    base_output_directory: Path | None = Path(args.output_base_directory) if args.output_base_directory else None
    if base_output_directory:
        base_output_directory.mkdir(parents=True, exist_ok=True)

    for subject in Path(args.data_directory).iterdir():

        if args.model is None:
            inferer = ImageTypeClassifierBase()
        else:
            inferer = ImageTypeClassifierBase(classification_model_filename=args.model)
        study = ProcessOneDicomStudyToVolumesMappingBase(
            study_directory=subject.as_posix(), inferer=inferer
        )
        study.run_inference()

        for series_number, series in study.series_dictionary.items():
            sub_volumes_data = {}

            study_uid: str = series.get_study_uid()  # get series and study uid
            series_uid: str = series.get_series_uid()

            # create output directory in form ${OUTPUTBASEDIR}/[StudyUID]/[SeriesNumber_SeriesUID]/
            output_directory: Path = Path(args.output_base_directory) / study_uid / f"{series_number}_{series_uid}"
            if not output_directory.exists():
                output_directory.mkdir(parents=True)

            series_modality: str = series.get_series_modality()

            json_file_name: str = f"{series_modality}_{series_uid}.json"

            for index, volume in enumerate(series.get_volume_list()):
                sub_volume_modality: str = volume.get_volume_modality()
                sub_volume_index: int = volume.get_volume_index()

                # get nifti image path in form ${ImageType}_${SeriesUID}_${SubvolumeNumber:03}.nii.gz
                output_nifti_file: Path = output_directory / f"{sub_volume_modality}_{series_uid}_{sub_volume_index:03}.nii.gz"

                # get training data dictionary and output probabilities
                training_data_dictionary: dict = volume.get_volume_dictionary()
                output_probabilities: dict = volume.get_modality_probabilities().to_dict()

                filtered_output_probabilities = {key.split('_')[-1]: value.get(0, 0.0) for key, value in output_probabilities.items() if key not in ["GUESS_ONNX", "SeriesNumber", "GUESS_ONNX_CODE"]}

                # TODO: if we want the entire volume dictionary dont use the filtered dictionary
                # for unwanted in [
                #     "FileName",
                #     "StudyInstanceUID",
                #     "SeriesInstanceUID",
                #     "list_of_ordered_volume_files",
                # ]:
                #     if unwanted in training_data_dictionary:
                #         training_data_dictionary.pop(unwanted)
                filtered_training_data = {key: value for key, value in training_data_dictionary.items() if key in inference_features}

                itk_image = volume.get_itk_image()
                itk.imwrite(itk_image, output_nifti_file)

                sub_volumes_data.update(create_subvolume_entry(sub_volume_index, sub_volume_modality, filtered_training_data, filtered_output_probabilities))

            # write the json file
            write_json(sub_volumes_data, output_directory / json_file_name)


if __name__ == "__main__":
    # Execute the script
    main()
