from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
import argparse


def generate_separator(column_width):
    return "|" + ("-" * (column_width + 2) + "|") * 3


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

col_width = 17
print(generate_separator(col_width))
print(
    generate_row(
        "Series number", "Modality", "Acquisition Plane", column_width=col_width
    )
)
print(generate_separator(col_width))
for series_number, series in study.series_dictionary.items():
    modality = series.get_modality()
    plane = series.get_acquisition_plane()
    print(generate_row(str(series_number), modality, plane, column_width=col_width))


if __name__ == "__main__":
    # Execute the script
    pass  # The main logic is already written above, so this is just a placeholder.
