#!/usr/bin/env python3

import argparse
import sys
from typing import Any
import pandas as pd
import tabulate
from collections import OrderedDict as ordered_dict

# warnings.simplefilter(action="ignore", category=FutureWarning)

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


def generate_separator(column_width):
    return "|" + ("-" * (column_width + 2) + "|") * 4


def generate_row(*args, column_width):
    return "| " + " | ".join(arg.ljust(column_width) for arg in args) + " |"


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
        current_dict: dict[str, str] = ordered_dict()
        current_dict["Series#"] = str(series_number)
        current_dict["Vol.#"] = str(volume.get_volume_index())
        current_dict["Volume Modality"] = str(volume.get_volume_modality())
        current_dict["Series Modality"] = str(series.get_series_modality())
        current_dict["Acq.Plane"] = str(volume.get_acquisition_plane())
        current_dict["Isotropic"] = str(volume.get_is_isotropic())
        print(volume.get_modality_probabilities().to_string(index=False))
        current_dict["Bvalue"] = str(volume.get_volume_bvalue())
        current_dict["SeriesDesc"] = volume.get_volume_series_description()
        # info_dict = series.get_series_info_dict()
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


df: pd.DataFrame = pd.DataFrame(list_of_dictionaries)
table_msg: str = tabulate.tabulate(df, headers="keys", tablefmt="psql", showindex=False)
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
    pass  # The main logic is already written above, so this is just a placeholder.
