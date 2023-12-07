import sys
from typing import Any

import pandas as pd
import tabulate

# from tabulate import tabulate

try:
    from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
    from dcm_classifier.image_type_inference import ImageTypeClassifierBase
except Exception as e:
    from pathlib import Path

    print(f"Missing module import {e}")
    print(
        f"Try setting export PYTHONPATH={Path(__file__).parent.parent.as_posix()}/src"
    )
    sys.exit(255)


import argparse


def generate_separator(column_width):
    return "|" + ("-" * (column_width + 2) + "|") * 4


def generate_row(*args, column_width):
    return "| " + " | ".join(arg.ljust(column_width) for arg in args) + " |"


# Set up argparse
description = (
    f"Authors: Michal Brzus, Hans J. Johnson\n"
    f"Classify image modality and acquisition plane of DICOM MR data by series and volumes.\n"
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
    required=True,
    help="Path to the model used for image type inference",
)

args = parser.parse_args()

print(description)
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
        # modality = volume.get_modality()
        #         plane = volume.get_acquisition_plane()
        #         print(generate_row(str(series_number), modality, plane, iso, column_width=col_width))
        current_dict: dict[str, str] = {}
        current_dict["Series#"] = str(series_number)
        current_dict["Vol.#"] = str(index)
        current_dict["Modality"] = str(volume.get_modality())
        current_dict["Acq.Plane"] = str(volume.get_acquisition_plane())
        current_dict["Isotropic"] = str(volume.get_is_isotropic())
        # info_dict = series.get_series_info_dict()
        inputs_df: dict[str, Any] = volume.get_volume_info_dict()
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


df: pd.DataFrame = pd.DataFrame(list_of_dictionaries)
table_msg: str = tabulate.tabulate(df, headers="keys", tablefmt="psql", showindex=False)
print(table_msg)

all_inputs_df: pd.DataFrame = pd.DataFrame(list_of_inputs)
inputs_msg: str = tabulate.tabulate(
    all_inputs_df, headers="keys", tablefmt="psql", showindex=False
)
print(inputs_msg)

all_prob_df: pd.DataFrame = pd.concat(list_of_probabilities, axis=0, ignore_index=True)
prob_msg: str = tabulate.tabulate(
    all_prob_df, headers="keys", tablefmt="psql", showindex=False
)
print(prob_msg)


# # below is the code to run inference on volume level
# print("\n\nInference per Volume\n\n")
# inferer = ImageTypeClassifierBase(
#     classification_model_filename=args.model, mode="volume"
# )
# study = ProcessOneDicomStudyToVolumesMappingBase(
#     study_directory=args.session_directory, inferer=inferer
# )
# study.run_inference()
#
# print(generate_separator(col_width))
# print(
#     generate_row(
#         "Series number", "Modality", "Acquisition Plane", "Isotropic", column_width=col_width
#     )
# )
# print(generate_separator(col_width))
# for series_number, series in study.series_dictionary.items():
#     for volume in series.get_volume_list():
#         modality = volume.get_modality()
#         plane = volume.get_acquisition_plane()
#         print(generate_row(str(series_number), modality, plane, iso, column_width=col_width))


if __name__ == "__main__":
    # Execute the script
    pass  # The main logic is already written above, so this is just a placeholder.
