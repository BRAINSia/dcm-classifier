#!/usr/bin/env python3

from pathlib import Path
import xlwt
import argparse
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase

inference_model_path = list(
    Path(__file__).parent.parent.rglob("models/rf_classifier.onnx")
)[0]

current_file_path: Path = Path(__file__).resolve()

"""
Instructions:
1. Run this script from the command line with the following arguments:
    --dir/-d: The directory of DICOM files to be classified
    --output/-o: The output path and file name (directory/filename)
2. The output file will be saved in the current working directory unless specified otherwise
"""


# The StudyClassification class is used to run the classification on a single study and has the option to
# return the study ProcessOneDicomStudyToVolumesMappingBase object or the study dictionary
class StudyClassification:
    def __init__(self, study_directory: Path):
        self.study_directory: Path = study_directory
        self.study: ProcessOneDicomStudyToVolumesMappingBase = self.run_study(
            study_directory
        )

    @staticmethod
    def run_study(study_directory: Path) -> ProcessOneDicomStudyToVolumesMappingBase:
        inferer = ImageTypeClassifierBase(
            classification_model_filename=inference_model_path, mode="volume"
        )
        study = ProcessOneDicomStudyToVolumesMappingBase(
            study_directory=study_directory, inferer=inferer
        )
        study.run_inference()
        return study

    def get_study_dict(self):
        return self.study.get_study_dictionary()

    def get_study(self):
        return self.study


def output_data(directory: str, output_file: str):
    wb = xlwt.Workbook()
    sheet = wb.add_sheet("Dicom Classifications")

    # Fields that are not needed in the output file, can be changed if needed
    not_needed_fields = ["FileName", "ManufacturerCode", "list_of_ordered_volume_files"]

    field_mapping: dict[str:int] = {
        "SeriesNumber": 3,
        "SeriesDescription": 4,
        "StudyInstanceUID": 5,
        "SeriesInstanceUID": 6,
        "PixelSpacing": 7,
        "ImageOrientationPatient": 8,
        "PixelBandwidth": 9,
        "EchoTime": 10,
        "RepetitionTime": 11,
        "FlipAngle": 12,
        "EchoNumbers": 13,
        "ContrastBolusAgent": 14,
    }

    bold_style = xlwt.easyxf("font: bold 1")
    sheet.write(0, 0, "Directory Path", bold_style)
    sheet.write(0, 1, "Modality", bold_style)
    sheet.write(0, 2, "Plane", bold_style)

    j = 3
    for field, index in field_mapping.items():
        sheet.write(0, index, field, bold_style)
        j += 1

    i = 1
    for item in Path(directory).rglob("*"):
        list_files = list(item.glob("*.dcm"))
        if item.is_dir() and len(list_files) > 0:
            # returns a dictionary of series numbers and series objects
            study_dict = StudyClassification(item).get_study_dict()

            for series_num, series in study_dict.items():
                # Write the path, modality, and acquisition plane for each series
                sheet.write(i, 0, str(item.absolute()))
                sheet.write(i, 1, series.get_series_modality())
                sheet.write(i, 2, series.get_acquisition_plane())

                # Write the series info for each series
                for field, value in series.get_series_info_dict().items():
                    try:
                        sheet.write(i, field_mapping[field], str(value))
                    except KeyError:
                        if field not in not_needed_fields:
                            field_mapping[field] = j
                            sheet.write(0, j, field, bold_style)
                            sheet.write(i, j, str(value))
                            j += 1
                i += 1

    sheet.col(0).width = 10000
    for i in range(len(sheet.cols)):
        sheet.col(i).width = 10000

    wb.save(output_file + ".xls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add directory argument
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        required=True,
        help="Enter directory of DICOM files to be classified",
    )
    # Add output argument
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Enter the name of the output excel file",
    )

    args = parser.parse_args()  # Execute the parse_args() method

    output_data(args.dir, args.output)

    print(f"Files classified in {args.output}.xls")
