#!/usr/bin/env python3

import sys
from pathlib import Path

import pandas as pd
from glob import glob
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from dcm_classifier.image_type_inference import ImageTypeClassifierBase

import pydicom


# Higher numbers are my (Hans) best guess at being more important for classifying the images
importance_values = [
    "5:Acquisition Matrix",
    "5:Image Orientation (Patient)",
    "5:In-plane Phase Encoding Direction",
    "5:Laterality",
    "5:Scan Options",
    "5:Scanning Sequence",
    "5:Sequence Variant",
    "6:Diffusion Gradient Orientation",
    "6:Diffusion b-value",
    "6:Number of Phase Encoding Steps",
    "6:dB/dt",
    "7:Columns",
    "7:Echo Train Length",
    "7:Imaging Frequency",
    "7:Number of Averages",
    "7:Pixel Spacing",
    "7:Rows",
    "7:Slice Thickness",
    "8:Echo Time",
    "8:Inversion Time",
    "8:Magnetic Field Strength",
    "8:Percent Phase Field of View",
    "8:Percent Sampling",
    "8:Pixel Bandwidth",
    "8:Reconstruction Diameter",
    "8:Repetition Time",
    "8:SAR",
    "8:Spacing Between Slices",
    "9:Flip Angle",
    "9:Modality",
    "Diffusionb-value",
    "DiffusionGradientOrientation",
    # "AcquisitionMatrix",
    # "PixelSpacing",
    # "ReconstructionDiameter",
    # "InversionTime",
]

feature_columns_type_image_type_id_new = [
    "ImageTypeADC",
    "ImageTypeFA",
    "ImageTypeTrace",
    "SeriesVolumeCount",
    #  DOES NOT WORK "IsSecondaryOrProcessed",  # A fallback for very old data,  almost certainly an ADC
    "EchoTime",
    "RepetitionTime",
    "FlipAngle",
    "PixelBandwidth",
    #    "EchoTrainLength",
    "SAR",
    #    "PixelSpacingMin",
    # Optional missing in for ADC "NumberofPhaseEncodingSteps",
    # "PercentSampling",
    "Diffusionb-valueCount",
    "Diffusionb-valueMax",
    # REDUNDANT with ImageTypeADC ImageTypeFA ImageTypeTrace "IsDerivedImageType",
    # NI - "MagneticFieldStrength",
    # NI - "HasDiffusionGradientOrientation",
    # NI - "PercentPhaseFieldofView",
    # NI - "ImagingFrequency",
    # Redundant - "PixelSpacingMax",
    # NI - "SliceThickness",
    # "NumberofAverages",
    # "SpacingBetweenSlices",
    # "dB/dt",
    "ImageOrientationPatient_0",
    "ImageOrientationPatient_1",
    "ImageOrientationPatient_2",
    "ImageOrientationPatient_3",
    "ImageOrientationPatient_4",
    "ImageOrientationPatient_5",
    # "AxialIndicator",
    # "CoronalIndicator",
    # "SaggitalIndicator"
    # "Rows",
    # "Columns",
]

# The feature_columns are roughly order by importance based on a trivial RandomForest classifier test case.
# NI - not important
# Redundant - may be redundant with other values
feature_columns_type_image_type_id_20230127 = [
    "ImageTypeADC",
    "ImageTypeTrace",
    "SeriesVolumeCount",
    #  DOES NOT WORK "IsSecondaryOrProcessed",  # A fallback for very old data,  almost certainly an ADC
    "EchoTime",
    "RepetitionTime",
    "FlipAngle",
    "PixelBandwidth",
    #    "EchoTrainLength",
    "SAR",
    #    "PixelSpacingMin",
    # Optional missing in for ADC "NumberofPhaseEncodingSteps",
    # "PercentSampling",
    "Diffusionb-valueCount",
    "Diffusionb-valueMax",
    # REDUNDANT with ImageTypeADC ImageTypeFA ImageTypeTrace "IsDerivedImageType",
    # NI - "MagneticFieldStrength",
    # NI - "HasDiffusionGradientOrientation",
    # NI - "PercentPhaseFieldofView",
    # NI - "ImagingFrequency",
    # Redundant - "PixelSpacingMax",
    # NI - "SliceThickness",
    # "NumberofAverages",
    # "SpacingBetweenSlices",
    # "dB/dt",
    "ImageOrientationPatient_0",
    "ImageOrientationPatient_1",
    "ImageOrientationPatient_2",
    "ImageOrientationPatient_3",
    "ImageOrientationPatient_4",
    "ImageOrientationPatient_5",
    # "AxialIndicator",
    # "CoronalIndicator",
    # "SaggitalIndicator"
    # "Rows",
    # "Columns",
]


all_found_tags = (
    "PROSTAT_Extended_TYPE",
    "SeriesDescription",
    "ImageType",
    "Accession Number",
    "Acquisition Matrix",
    "Acquisition Number",
    "Angio Flag",
    "B1rms",
    "Bits Allocated",
    "Bits Stored",
    "Body Part Examined",
    "Cardiac Number of Images",
    "Code Value",
    "Coding Scheme Designator",
    "Columns",
    "Content Time",
    "Contrast/Bolus Agent",
    "Conversion Type",
    "De-identification Method Code Sequence",
    "Diffusion Gradient Orientation",
    "Diffusion b-value",
    "Digital Image Format Acquired",
    "Echo Number(s)",
    "Echo Time",
    "Echo Train Length",
    "Flip Angle",
    "Frame of Reference UID",
    "High Bit",
    "High R-R Value",
    "Image Orientation (Patient)",
    "Image Position (Patient)",
    "Image Type",
    "Imaged Nucleus",
    "Imaging Frequency",
    "In-plane Phase Encoding Direction",
    "Instance Creation Date",
    "Intervals Acquired",
    "Intervals Rejected",
    "Inversion Time",
    "Largest Image Pixel Value",
    "Laterality",
    "Longitudinal Temporal Information Modified",
    "Low R-R Value",
    "MR Acquisition Type",
    "Magnetic Field Strength",
    "Manufacturer",
    "Manufacturer's Model Name",
    "Modality",
    "Number of Averages",
    "Number of Phase Encoding Steps",
    "Number of Temporal Positions",
    "Percent Phase Field of View",
    "Percent Sampling",
    "Performed Procedure Step Description",
    "Performed Procedure Step ID",
    "Performed Procedure Step Status",
    "Performed Procedure Type Description",
    "Performed Protocol Code Sequence",
    "Photometric Interpretation",
    "Pixel Bandwidth",
    "Pixel Representation",
    "Pixel Spacing",
    "Planar Configuration",
    "Position Reference Indicator",
    "Presentation LUT Shape",
    "Protocol Name",
    "Real World Value Mapping Sequence",
    "Receive Coil Name",
    "Reconstruction Diameter",
    "Referenced Image Sequence",
    "Referenced Performed Procedure Step Sequence",
    "Referenced Study Sequence",
    "Relative Opacity",
    "Repetition Time",
    "Request Attributes Sequence",
    "Requested Contrast Agent",
    "Requested Procedure Comments",
    "Requested Procedure ID",
    "Requested Procedure Location",
    "Requested Procedure Priority",
    "Requesting Service",
    "Rows",
    "SAR",
    "SOP Class UID",
    "Samples per Pixel",
    "Scan Options",
    "Scanning Sequence",
    "Scheduled Protocol Code Sequence",
    "Secondary Capture Device ID",
    "Secondary Capture Device Manufacturer",
    "Secondary Capture Device Manufacturer's Model Name",
    "Secondary Capture Device Software Versions",
    "Sequence Name",
    "Sequence Variant",
    "Series Description",
    "Slice Thickness",
    "Smallest Image Pixel Value",
    "Software Versions",
    "Spacing Between Slices",
    "Special Needs",
    "Specific Character Set",
    "Stack ID",
    "Study Comments",
    "Study Description",
    "Study Priority ID",
    "Study Status ID",
    "Transmit Coil Name",
    "Trigger Window",
    "Variable Flip Angle Flag",
    "Window Center",
    "Window Width",
    "dB/dt",
)

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


hj_guess_importance_values = (
    "0:Accession Number",
    "0:Acquisition Number",
    "0:Angio Flag",
    "0:Bits Allocated",
    "0:Bits Stored",
    "0:Cardiac Number of Images",
    "0:Code Value",
    "0:Coding Scheme Designator",
    "0:Content Time",
    "0:Contrast/Bolus Agent",
    "0:Conversion Type",
    "0:De-identification Method Code Sequence",
    "0:Digital Image Format Acquired",
    "0:Echo Number(s)",
    "0:Frame of Reference UID",
    "0:High Bit",
    "0:High R-R Value",
    "0:Imaged Nucleus",
    "0:Instance Creation Date",
    "0:Intervals Acquired",
    "0:Intervals Rejected",
    "0:Largest Image Pixel Value",
    "0:Longitudinal Temporal Information Modified",
    "0:Low R-R Value",
    "0:MR Acquisition Type",
    "0:Number of Temporal Positions",
    "0:Performed Procedure Step Description",
    "0:Performed Procedure Step ID",
    "0:Performed Procedure Step Status",
    "0:Performed Procedure Type Description",
    "0:Performed Protocol Code Sequence",
    "0:Photometric Interpretation",
    "0:Pixel Representation",
    "0:Planar Configuration",
    "0:Position Reference Indicator",
    "0:Presentation LUT Shape",
    "0:Real World Value Mapping Sequence",
    "0:Receive Coil Name",
    "0:Referenced Image Sequence",
    "0:Referenced Performed Procedure Step Sequence",
    "0:Referenced Study Sequence",
    "0:Relative Opacity",
    "0:Request Attributes Sequence",
    "0:Requested Contrast Agent",
    "0:Requested Procedure Comments",
    "0:Requested Procedure ID",
    "0:Requested Procedure Location",
    "0:Requested Procedure Priority",
    "0:Requesting Service",
    "0:SOP Class UID",
    "0:Samples per Pixel",
    "0:Scheduled Protocol Code Sequence",
    "0:Secondary Capture Device ID",
    "0:Secondary Capture Device Manufacturer",
    "0:Secondary Capture Device Manufacturer's Model Name",
    "0:Secondary Capture Device Software Versions",
    "0:Smallest Image Pixel Value",
    "0:Software Versions",
    "0:Special Needs",
    "0:Specific Character Set",
    "0:Stack ID",
    "0:Study Comments",
    "0:Study Description",
    "0:Study Priority ID",
    "0:Study Status ID",
    "0:Transmit Coil Name",
    "0:Trigger Window",
    "0:Variable Flip Angle Flag",
    "0:Window Center",
    "0:Window Width",
    "1:Body Part Examined",
    "2:Manufacturer",
    "2:Manufacturer's Model Name",
    "2:Protocol Name",
    "2:Sequence Name",
    "3:Series Description",
    "4:Image Type",
    "0:Image Position (Patient)",
    "0:B1rms",
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

#
# def identify_single_volumes(all_series_dicom_files: List[Path]):
#     slice_location_dictionary = collections.defaultdict(list)
#     # build a dictionary of lists for each location.
#     for dcm_file in all_series_dicom_files:
#         if not Path(dcm_file).exists():
#             continue
#         dataset = pydicom.dcmread(dcm_file, stop_before_pixels=True)
#         # print(f"{dataset.ImagePositionPatient} {dist_from_origin}")
#         slice_location_dictionary[dist_from_origin].append(dataset)
#         del dataset
#         del dcm_file
#     # oder each slice based on the order they were acquired
#     for dist_from_origin, slice_list in slice_location_dictionary.items():
#         new_slice_list = sorted(slice_list, key=lambda s: s.InstanceNumber)
#         slice_location_dictionary[dist_from_origin] = new_slice_list
#         del new_slice_list
#         del dist_from_origin
#         del slice_list
#     # extract each volume from the lists in the dictionaries
#     volumes_dictionary = collections.defaultdict(list)
#     for _slice_loc, slice_list in slice_location_dictionary.items():
#         for index in range(0, len(slice_list)):
#             volumes_dictionary[index].append(slice_list[index])
#     return volumes_dictionary
#

keep_columns_prostatid_selected_images = [
    # "FileName",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesDescription",
    # "ImageType",
    # "ImageTypeADC",
    # "ImageTypeFA",
    # "ImageTypeTrace",
    # "SeriesVolumeCount",
    "Diffusionb-valueCount",
    "Diffusionb-valueMax",
    # "HasDiffusionGradientOrientation",
    "Diffusionb-valueSet",
    # "EchoTime",
    # "PixelBandwidth",
    # "EchoTrainLength",
    # "SAR",
    # "PixelSpacingMin",
    # "SliceThickness",
    # "RepetitionTime",
    # "FlipAngle",
    # "PercentSampling",
    # "ScanOptions",
    "SeriesNumber",
    # "Manufacturer'sModelName",
    "ManufacturerCode",
    # "AxialIndicator",
    # "CoronalIndicator",
    # "SaggitalIndicator",
    # "ImageOrientationPatient_0",
    # "ImageOrientationPatient_1",
    # "ImageOrientationPatient_2",
    # "ImageOrientationPatient_3",
    # "ImageOrientationPatient_4",
    # "ImageOrientationPatient_5",
    "all_dicom_files",
    # "IsDerivedImageType",
    # "SpecificCharacterSet",
    # "InstanceCreationDate",
    # "InstanceCreationTime",
    # "SOPClassUID",
    # "SOPInstanceUID",
    # "StudyDate",
    # "ContentDate",
    # "StudyTime",
    # "ContentTime",
    # "AccessionNumber",
    # "ReferringPhysician'sName",
    # "StudyDescription",
    # "ReferencedImageSequence",
    # "Patient'sName",
    # "PatientID",
    # "Patient'sBirthDate",
    # "Patient'sSex",
    # "PatientIdentityRemoved",
    # "ScanningSequence",
    # "SequenceVariant",
    # "MRAcquisitionType",
    # "ImagingFrequency",
    # "EchoNumbers",
    "MagneticFieldStrength",
    # "SpacingBetweenSlices",
    # "SoftwareVersions",
    # "TriggerTime",
    # "CardiacNumberofImages",
    # "ReconstructionDiameter",
    # "ReceiveCoilName",
    # "AcquisitionMatrix",
    # "In-planePhaseEncodingDirection",
    # "PatientPosition",
    # "StudyID",
    # "InstanceNumber",
    # "FrameofReferenceUID",
    # "ImagesinAcquisition",
    # "PositionReferenceIndicator",
    # "SliceLocation",
    # "SamplesperPixel",
    # "PhotometricInterpretation",
    # "PlanarConfiguration",
    # "Rows",
    # "Columns",
    # "PixelSpacingMax",
    # "BitsAllocated",
    # "BitsStored",
    # "HighBit",
    # "PixelRepresentation",
    # "SmallestImagePixelValue",
    # "LargestImagePixelValue",
    # "WindowCenter",
    # "WindowWidth",
    # "ScheduledProtocolCodeSequence",
    # "RealWorldValueMappingSequence",
    # "RelativeOpacity",
    # "AngioFlag",
    # "NumberofAverages",
    # "ImagedNucleus",
    # "PercentPhaseFieldofView",
    # "HeartRate",
    # "TriggerWindow",
    # "VariableFlipAngleFlag",
    # "Diffusionb-value",
    # "AcquisitionNumber",
    # "StackID",
    # "In-StackPositionNumber",
    # "InversionTime",
    # "GUESS_ONNX_CODE",
    "GUESS_ONNX",
]


# PROSTAT_TYPE is a simple application of the series description classifier rule set.
result_columns = ["PROSTAT_TYPE"]

# candidate_tags = [x.split(":")[1] for x in importance_values]
#
# # Common dicom numeric type codes
# numeric_vr_types = ["DS", "FL", "FD", "IS", "SL", "SS", "UL", "US"]


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
        except Exception as _:
            pass
        if value is not None or value == "":
            information[key] = value
    return information


def generate_dicom_dataframe(
    session_dirs: list, output_file: str, inferer: ImageTypeClassifierBase
):
    dfs = [pd.DataFrame.from_dict({})]
    for ses_dir in session_dirs:
        study = ProcessOneDicomStudyToVolumesMappingBase(
            study_directory=ses_dir, inferer=inferer
        )
        study.run_inference()
        print(f"Processing {ses_dir}: {study.series_dictionary}")
        for series_number, series in study.series_dictionary.items():
            modality = series.get_series_modality()
            plane = series.get_acquisition_plane()
            print(f"         {series_number} {modality} {plane}")
            img_dict = {}
            for index, series_vol in enumerate(series.volume_info_list):
                ds = pydicom.dcmread(
                    series_vol.one_volume_dcm_filenames[0], stop_before_pixels=True
                )
                img_dict = data_set_to_dict(ds)
                img_dict["_vol_index"] = index
                img_dict["_dcm_image_type"] = modality
                img_dict["_dcm_image_orientation_patient"] = plane
                img_dict["FileName"] = series_vol.one_volume_dcm_filenames[0]
                for k, v in series.get_series_info_dict().items():
                    img_dict[k] = v
            series_df = pd.DataFrame.from_dict(data=img_dict, orient="index").T
            dfs.append(series_df)

    if len(dfs) > 1:
        df = pd.concat(dfs, axis=0, ignore_index=True)

        all_columns = list(df.columns)
        ordered_columns = [
            "FileName",
            "_vol_index",
            "_dcm_image_type",
            "_dcm_image_orientation_patient",
        ] + [x for x in output_additional_flags if x in all_columns]

        prefered_odering = make_unique_ordered_list(ordered_columns + all_columns)

        df = df[prefered_odering]

        df.to_excel(output_file, index=False)
    else:
        print(f"NO MR DICOM DATA FOUND IN {session_dirs}")


if __name__ == "__main__":
    # path to prostate data DICOM dir: /localscratch/Users/mbrzus/Botimageai/homerun/DATA/150_Test_Data

    # Note: Prostate data not in strict BIDS format therefore require small change in do_training_extraction function
    # to find the data
    # dicom_path = "/localscratch/Users/mbrzus/Botimageai/homerun/DATA/150_Test_Data"
    # out = "../../data/prostate_all_dicom_raw.xlsx"
    # out_p = "../../data/prostate_df.pkl"
    # generate_dicom_dataframe(dicom_dir=dicom_path, output_file=out, out_pkl=out_p)

    # dicom_path = "/localscratch/Users/mbrzus/Stroke_Data/IOWA_STROKE_RETRO_DICOM"
    # out = "../../data/iowaStroke_all_dicom_raw.xlsx"
    # out_p = "../../data/iowaStroke_df.pkl"
    # generate_dicom_dataframe(dicom_dir=dicom_path, output_file=out, out_pkl=out_p)

    # dicom_path = "/localscratch/Users/mbrzus/TrackOn/HDNI_003"
    # out = "../../data/track32_all_dicom_raw.xlsx"
    # out_p = "../../data/track32_df.pkl"
    # generate_dicom_dataframe(dicom_dir=dicom_path, output_file=out, out_pkl=out_p)
    #
    # dicom_path = Path(
    #     "/Shared/sinapse/MiniPigScratch/CLN2_data/noah_batten_porcine_model/dicom_SIEREN_CLN2"
    # )
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--dicom_path",
        type=str,
        default="/Users/johnsonhj/Botimageai/homerun/DATA/150_Test_Data",
        help="Path to DICOM directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/Users/johnsonhj/Downloads/testfile.xlsx",
        help="Path to output excel file",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=Path(__file__).parent.parent / "models" / "rf_classifier.onnx",
        help="Path to the model used for image type inference",
    )

    args = parser.parse_args()
    dicom_path = args.dicom_path
    model: Path = Path(args.model)
    out = args.out

    path_dirs = sorted(list(glob(f"{dicom_path}/*/*")))
    ses_dirs = [x for x in path_dirs if Path(x).is_dir()]
    if not model.exists():
        print(f"Model {model} does not exist")
        sys.exit(255)
    inferer = ImageTypeClassifierBase(classification_model_filename=model)
    generate_dicom_dataframe(session_dirs=ses_dirs, output_file=out, inferer=inferer)
    print(out)
