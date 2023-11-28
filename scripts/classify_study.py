import sys

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

col_width = 25
print(generate_separator(col_width))
print(
    generate_row(
        "Series number",
        "Modality",
        "Acquisition Plane",
        "Isotropic",
        column_width=col_width,
    )
)
print(generate_separator(col_width))
for series_number, series in study.series_dictionary.items():
    modality = str(series.get_modality())
    plane = str(series.get_acquisition_plane())
    iso = str(series.get_is_isotropic())
    print(
        generate_row(str(series_number), modality, plane, iso, column_width=col_width)
    )


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
