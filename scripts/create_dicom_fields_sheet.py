import sys
from pathlib import Path

import pandas as pd
from glob import glob
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase

import pydicom


phi_tags = (
    "Acquisition Date",
    "Acquisition Duration",
    "Acquisition Time",
    "Additional Patient History",
    "Admitting Diagnoses Description",
    "Allergies",
    "Anatomic Region Sequence",
    "Code Meaning",
    "Comments on the Performed Procedure Step",
    "Content Date",
    "Contributing Equipment Sequence",
    "Current Patient Location",
    "De-identification Method",
    "Device Serial Number",
    "Ethnic Group",
    "Heart Rate",
    "Image Comments",
    "Images in Acquisition",
    "Imaging Service Request Comments",
    "In-Stack Position Number",
    "Instance Creation Time",
    "Instance Creator UID",
    "Instance Number",
    "Institution Address",
    "Institution Name",
    "Institutional Department Name",
    "Issue Date of Imaging Service Request",
    "Issue Time of Imaging Service Request",
    "Lossy Image Compression",
    "Medical Alerts",
    "Military Rank",
    "Name of Physician(s) Reading Study",
    "Occupation",
    "Operators' Name",
    "Order Callback Phone Number",
    "Order Enterer's Location",
    "Other Patient IDs",
    "Patient Comments",
    "Patient ID",
    "Patient Identity Removed",
    "Patient Position",
    "Patient State",
    "Patient Transport Arrangements",
    "Patient's Address",
    "Patient's Age",
    "Patient's Birth Date",
    "Patient's Name",
    "Patient's Sex",
    "Patient's Size",
    "Patient's Telephone Numbers",
    "Patient's Weight",
    "Performed Location",
    "Performed Procedure Step End Date",
    "Performed Procedure Step End Time",
    "Performed Procedure Step Start Date",
    "Performed Procedure Step Start Time",
    "Performed Station AE Title",
    "Performed Station Name",
    "Performing Physician's Name",
    "Physician(s) of Record",
    "Pregnancy Status",
    "Procedure Code Sequence",
    "Reason for Study",
    "Reason for the Imaging Service Request",
    "Reason for the Requested Procedure",
    "Referenced SOP Instance UID",
    "Referring Physician's Name",
    "Requested Procedure Code Sequence",
    "Requested Procedure Description",
    "Requesting Physician Identification Sequence",
    "Requesting Physician",
    "SOP Instance UID",
    "Scheduled Performing Physician's Name",
    "Series Date",
    "Series Instance UID",
    "Series Number",
    "Series Time",
    "Slice Location",
    "Station Name",
    "Study Date",
    "Study ID",
    "Study Instance UID",
    "Study Time",
    "Temporal Position Identifier",
    "Timezone Offset From UTC",
    "Trigger Time",
    "Type of Patient ID",
    "Video Image Format Acquired",
    "Window Center & Width Explanation",
)


output_additional_flags = [
    "FileName",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "PROSTAT_TYPE",
    "PROSTAT_TYPE_SeriesDescription",
    "SeriesDescription",
    "ImageTypeADC",
    "ImageTypeFA",
    "ImageTypeTrace",
    "ImageType",
    "ScanOptions",
    "SeriesNumber",
    "Manufacturer'sModelName",
    "ManufacturerCode",
    "HasDiffusionGradientOrientation",
    "Diffusionb-valueSet",
    "Diffusionb-valueCount",
    "Diffusionb-valueMax",
    "IsDerivedImageType",
    "AxialIndicator",
    "CoronalIndicator",
    "SaggitalIndicator",
    "ImageOrientationPatient_0",
    "ImageOrientationPatient_1",
    "ImageOrientationPatient_2",
    "ImageOrientationPatient_3",
    "ImageOrientationPatient_4",
    "ImageOrientationPatient_5",
    "SliceThickness",
]


# PROSTAT_TYPE is a simple application of the series description classifier rule set.
result_columns = ["PROSTAT_TYPE"]


def make_unique_ordered_list(seq):
    # https://stackoverflow.com/a/480227
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def data_set_to_dict(ds):
    information = dict()
    for elem in ds:
        if elem.VR in ["SQ", "OB", "OW", "OF", "UT", "UN"]:
            continue
        key: str = elem.name
        if key in phi_tags:
            continue
        value: str | None = None
        try:
            value = str(elem.value).strip()
        except Exception as _e:
            pass
        if value is not None or value == "":
            information[key] = value
    return information


def generate_dicom_dataframe(
    session_dirs: list, output_file: str | None, save_to_excel: bool = True
) -> None | pd.DataFrame:
    """
    Generate a dataframe from DICOM data

    Args:
        session_dirs: list: list of session directories
        output_file: str | None: path to the output file
        save_to_excel: bool: whether to save the dataframe to an excel file

    Returns:
        None | pd.DataFrame: if save_to_excel is True, None is returned, otherwise the dataframe is returned
    """
    dfs = [pd.DataFrame.from_dict({})]
    for ses_dir in session_dirs:
        study = ProcessOneDicomStudyToVolumesMappingBase(study_directory=ses_dir)
        for series_number, series in study.series_dictionary.items():
            for index, volume in enumerate(series.volume_info_list):
                ds = pydicom.dcmread(
                    volume.one_volume_dcm_filenames[0], stop_before_pixels=True
                )
                img_dict = data_set_to_dict(ds)

                try:
                    img_dict["FileName"] = volume.one_volume_dcm_filenames[0]
                except Exception:
                    img_dict["FileName"] = ""
                for k, v in volume.get_volume_dictionary().items():
                    img_dict[k] = v
                volume_df = pd.DataFrame.from_dict(data=img_dict, orient="index").T
                dfs.append(volume_df)

    if len(dfs) > 1:
        df = pd.concat(dfs, axis=0, ignore_index=True)

        all_columns = list(df.columns)
        ordered_columns = [
            "FileName",
        ] + [x for x in output_additional_flags if x in all_columns]

        prefered_odering = make_unique_ordered_list(ordered_columns + all_columns)

        df = df[prefered_odering]

        if save_to_excel:
            df.to_excel(output_file, index=False)
        else:
            return df
    else:
        print(f"NO MR DICOM DATA FOUND IN {session_dirs}")


if __name__ == "__main__":
    import argparse

    current_file: Path = Path(__file__).resolve()
    root_dir: Path = current_file.parent.parent
    test_data_dir: Path = (
        root_dir
        / "tests"
        / "testing_data"
        / "anonymized_testing_data"
        / "anonymized_data"
    )

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--dicom_path",
        type=str,
        default=test_data_dir.as_posix(),
        help="Path to DICOM directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./testfile.xlsx",
        help="Path to output excel file",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=root_dir / "models" / "rf_classifier.onnx",
        help="Path to the model used for image type inference",
    )

    args = parser.parse_args()
    dicom_path = args.dicom_path
    model: Path = Path(args.model)
    out = args.out

    path_dirs = sorted(list(glob(f"{dicom_path}/*")))
    ses_dirs = [x for x in path_dirs if Path(x).is_dir()]
    if not model.exists():
        print(f"Model {model} does not exist")
        sys.exit(255)
    generate_dicom_dataframe(session_dirs=ses_dirs, output_file=out)
    print(out)
