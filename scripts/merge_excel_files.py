import argparse
import re

# from dcm_classifier.
import pandas as pd
from pathlib import Path
import numpy as np
import xlrd
from dcm_classifier.namic_dicom_typing import convert_array_to_index_value

max_header_length: int = 34


def get_session_id_from_filename(filename: str):
    file_contents = filename.split("/")
    # possibly can add method to add paths not containing session ids
    id_string: str = file_contents[6]
    if "THPBALL" in id_string:
        return None
    elif "THP" in id_string:
        return id_string[3 : id_string.index("_")]
    else:
        return file_contents[7]


def merge_excel_files(bids_data: str, all_data: str, output_file: str):
    bids_orig_sheet: pd.DataFrame = pd.read_excel(bids_data)  # bids, original data
    all_data_sheet: pd.DataFrame = pd.read_excel(all_data)  # all data, new spreadsheet

    # add session id column to all_data_sheet
    all_data_sheet["session_id"] = ""

    # iterate through all_data_sheet and add session id to each row
    for index, row in all_data_sheet.iterrows():
        file_str = str(row["FileName"])
        all_data_sheet.at[index, "session_id"] = get_session_id_from_filename(file_str)

    # group all_data_sheet by session id
    grouped_all_data = all_data_sheet.groupby("session_id")

    # group bids_orig_sheet by session id
    grouped_bids_data = bids_orig_sheet.groupby(by="session_id")

    # find all unique session ids to iterate through
    unique_session_ids = all_data_sheet["session_id"].unique()
    res = []
    # append to res all the merged dataframes, dataframes are merged by session id
    for session_id in unique_session_ids:
        res.append(
            pd.merge(
                grouped_all_data.get_group(session_id),
                grouped_bids_data.get_group("ses-" + session_id),
                left_on="SeriesNumber",
                right_on="series_number",
                how="outer",
            )
        )

    merged_frame = pd.concat(res)

    merged_frame.to_excel(output_file, sheet_name="results", engine="openpyxl")


def combine_directory_excel_files(directory: str, output_file: str):
    combined_frame = pd.DataFrame()
    for file in Path(directory).glob("*.xlsx"):
        print(file)
        current_frame = pd.read_excel(file)
        combined_frame = pd.concat([combined_frame, current_frame])

    print("Excel Files Read")

    combined_frame.to_excel(output_file, sheet_name="results", engine="openpyxl")


# def check_length_three_pixel_spacing(frame: pd.DataFrame) -> pd.DataFrame:
#     df = frame
#     pixel_spacing_col = df[["PixelSpacing"]]
#     for row in pixel_spacing_col.loc[1:]:
#         if len(row) == 3:
#             print("here")
#     print("done")
dropped_cols = []


def one_hot_encoding_from_array(
    frame: pd.DataFrame, col_name: str, index_field: str
) -> pd.DataFrame:
    df = frame
    output_frame = frame[[index_field]]
    image_type_col = df[[col_name]]
    header_list = []
    # get all the unique column
    unique_image_types = image_type_col[col_name].unique()
    for image_type in unique_image_types:
        if type(image_type) is not str:
            continue
        row_contents = re.findall(r"\b\w+\b", image_type)
        if len(row_contents) > 0:
            for type_element in row_contents:
                if type_element not in header_list:
                    header_list.append(type_element)
    # print(len(header_list))
    # print(header_list)
    for header in header_list:
        series = pd.Series(
            (df[col_name].str.contains(header)).fillna(0).astype(int),
            name=f"{col_name}_{header}",
        )
        output_frame = output_frame.merge(series, left_index=True, right_index=True)

    # drop the image type column from the dataframe
    return output_frame

    merged_frame.concat(res)
    # merged_frame = pd.concat(res)


def one_hot_encoding_from_str_col(frame: pd.DataFrame, col_name: str) -> pd.DataFrame:
    df = frame
    encoding = pd.get_dummies(df[col_name], dtype=int)
    output_frame = pd.concat([df, encoding], axis=1)
    return output_frame


def one_hot_encoding_from_array_floats(
    frame: pd.DataFrame, col_name: str
) -> pd.DataFrame:
    convert_array_to_index_value()


# drops columns that have less than 5% of values NaN and less than 2 unique values
def identify_unusable_cols(frame: pd.DataFrame) -> list:
    df = frame
    droppable_cols = []
    length = len(df)
    for col in df.columns:
        if (df[col].count() / length) <= 0.05 or df[col].nunique() < 2:
            droppable_cols.append(col)
    # droppable_cols.append("list_of_ordered_volume_files")
    # droppable_cols.append("Image Type")
    print(f"dropping {len(droppable_cols)} cols")
    [print(x) for x in sorted(droppable_cols)]
    return df.drop(columns=droppable_cols)


def truncate_column_names(df: pd.DataFrame, max_length: int) -> pd.DataFrame:
    for col in df.columns:
        df[col].name = col.replace(col, col[:max_length])
    return df


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # # Add directory argument
    # parser.add_argument(
    #     "--dir",
    #     "-d",
    #     type=str,
    #     required=True,
    #     help="",
    # )
    # # Add output argument
    # parser.add_argument(
    #     "--output",
    #     "-o",
    #     type=str,
    #     required=True,
    #     help="",
    # )
    # args = parser.parse_args()
    # merge_excel_files(args.dir, args.output)
    # merge_excel_files("~/programs/dcm_files/fmri_14340_bids_tests.xls", "~/programs/dcm_files/fmri_14340_all_test.xlsx",
    #                   "~/programs/dcm_files/fmri_14340_test_merged.xls")
    # combine_directory_excel_files(
    #     "/tmp/dicom_data_tables", "~/programs/dcm_files/combined.xls"
    # )
    #
    # merge_excel_files(
    #     "~/programs/dcm_files/bids_image_information.xlsx",
    #     "~/programs/dcm_files/combined.xls",
    #     "~/programs/dcm_files/combined_merged.xls",
    # )

    # combine_directory_excel_files(
    #     "/home/cavriley/files/source_excel_files",
    #     "/home/cavriley/files/output_excel_files/combined_excel_file.xlsx",
    # )

    # input_data_frame = pd.read_excel(
    #     "/home/cavriley/files/output_excel_files/combined_excel_file_raw.xlsx"
    # )
    # input_data_frame = truncate_column_names(input_data_frame, max_header_length)
    # droppable_cols = identify_unusable_cols(input_data_frame)
    # input_data_frame = input_data_frame.drop(droppable_cols, axis=1)
    # input_data_frame = remove_duplicate_inferer_pydicom_cols(input_data_frame)
    # input_data_frame.to_excel(
    #     "/home/cavriley/files/output_excel_files/combined_excel_file_raw_no_brackets.xlsx",
    #     sheet_name="results",
    #     engine="openpyxl",
    # )
    # new_frame = remove_duplicate_inferer_pydicom_cols(input_data_frame)
    # if "Contrast/Bolus Agent" in droppable_cols:
    #     droppable_cols.remove("Contrast/Bolus Agent")
    # droppable_cols = [
    #     "Collimator Type",
    #     "Actual Frame Duration",
    #     "Position Reference Indicator",
    #     "Corrected Image",
    #     "Number of Slices",
    #     "Number of Time Slices",
    #     "Series Type",
    #     "Units",
    #     "Counts Source",
    #     "Decay Correction",
    #     "Frame Reference Time",
    #     "Image Index",
    #     "[Product id]",
    #     "[Table delta]",
    #     "[Cardiac repetition time]",
    #     "[Images per cardiac cycle]",
    #     "[Delay after trigger]",
    #     "[Pause Interval]",
    #     "[Pause Time]",
    #     "[Slice offset on freq axis]",
    #     "[Surface Coil Type]",
    #     "[Extremity Coil flag]",
    #     "[Calibrated Field strength]",
    #     "[User data 9]",
    #     "[User data 10]",
    #     "[User data 13]",
    #     "[User data 14]",
    #     "[User data 16]",
    #     "[User data 19]",
    #     "[Projection angle]",
    #     "[Phase Contrast flow axis]",
    #     "[Velocity encoding]",
    #     "[Thickness disclaimer]",
    #     "[Projection Algorithm]",
    #     "[Projection Algorithm Name]",
    #     "[Cardiac phase number]",
    #     "[Concatenated SAT {# DTI Diffusion Dir., release 9.0 & below}]",
    #     "[Velocity Encode Scale]",
    #     "[Fast phases]",
    #     "Laterality",
    #     "Stack ID",
    #     "[Rotation from source x rot]",
    #     "[Rotation from source y rot]",
    #     "[Rotation from source z rot]",
    #     "[Image Filtering 0.5/0.2T]",
    #     "[RTIA_timer]",
    #     "[Fps]",
    #     "[Auto window/level alpha]",
    #     "[Auto window/level beta]",
    #     "[Auto window/level window]",
    #     "[Auto window/level level]",
    #     "[Start time(secs) in first axial]",
    #     "[Indicates study has complete info (DICOM/genesis)]",
    #     "[Landmark Counter]",
    #     "[Series Complete Flag]",
    #     "[Number of images archived]",
    #     "[Image archive flag]",
    #     "[Scout Type]",
    #     "[Foreign Image Revision]",
    #     "[Lower range of Pixels1]",
    #     "[Upper range of Pixels1]",
    #     "[Lower range of Pixels2]",
    #     "[Upper range of Pixels2]",
    #     "[Version of the hdr struct]",
    #     "[Views per segment]",
    #     "[Respiratory rate, bpm]",
    #     "[Respiratory trigger point]",
    #     "[GE image integrity]",
    #     "[Filter Mode (String slop field 1 in legacy GE MR]",
    #     "[String slop field 2]",
    #     "[Image Type (real, imaginary, phase, magnitude)]",
    #     "[Neg_scanspacing]",
    #     "[User_fill_map_MSW]",
    #     "[User_fill_map_LSW]",
    #     "[Coil ID Data]",
    #     "[Prescan Reuse String]",
    #     "[ASSET Acquisition Calibration Series UID]",
    #     "[Rx Stack Identification]",
    #     "Pixel Padding Value",
    #     "Group Length",
    #     "[NEW/SEEN Status]",
    #     "[Receive Origin]",
    #     "[Receive Date]",
    #     "[Receive Time]",
    #     "[Receive Origin Description]",
    #     "[Full fidelity]",
    #     "Temporal Resolution",
    #     "[Series from which Prescribed]",
    #     "[PURE Acquisition Calibration Series UID]",
    #     "Date of Last Calibration",
    #     "Time of Last Calibration",
    #     "[CSA Series Header Type]",
    #     "[PMTF Information 1]",
    #     "[PMTF Information 2]",
    #     "[PMTF Information 3]",
    #     "[PMTF Information 4]",
    #     "[Series Workflow Status]",
    #     "Synchronization Frame of Reference UID",
    #     "Concatenation UID",
    #     "[CSA Image Header Version ??]",
    #     "[PositivePCSDirections]",
    #     "Study Status ID",
    #     "[Mitra Rotation]",
    #     "[Mitra Window Width]",
    #     "[Mitra Window Centre]",
    #     "[Mitra Invert]",
    #     "Derivation Description",
    #     "Requesting Service",
    #     "Length to End",
    #     "Nominal Interval",
    #     "[FMRIStimulInfo]",
    #     "Code Value",
    #     "Coding Scheme Designator",
    #     "Intervals Acquired",
    #     "Intervals Rejected",
    #     "Study Arrival Date",
    #     "Study Arrival Time",
    #     "Study Completion Date",
    #     "Study Completion Time",
    #     "[Chemical Shift]",
    #     "[Chemical Shift Number MR]",
    #     "[Diffusion B-Factor]",
    #     "[Diffusion Direction]",
    #     "[Image Enhanced]",
    #     "[Image Type ED ES]",
    #     "[Phase Number]",
    #     "[Slice Number MR]",
    #     "[Slice Orientation]",
    #     "[Cardiac Sync]",
    #     "[Diffusion Echo Time]",
    #     "[Dynamic Series]",
    #     "[EPI Factor]",
    #     "[Number of Echoes]",
    #     "[Number of Locations]",
    #     "[Number of PC Directions]",
    #     "[Number of Phases MR]",
    #     "[Number of Slices MR]",
    #     "[Partial Matrix Scanned]",
    #     "[PC Velocity]",
    #     "[Prepulse Delay]",
    #     "[Prepulse Type]",
    #     "[Reconstruction Number MR]",
    #     "[Respiration Sync]",
    #     "[Scanning Technique Description MR]",
    #     "[SPIR]",
    #     "[Water Fat Shift]",
    #     "[Flip Angle Philips]",
    #     "[Interactive]",
    #     "[Echo Time Display MR]",
    #     "[Number of Stacks]",
    #     "[Examination Source]",
    #     "[Acquisition Number]",
    #     "[Number of Dynamic Scans]",
    #     "[Echo Train Length]",
    #     "[Imaging Frequency]",
    #     "[Inversion Time]",
    #     "[Magnetic Field Strength]",
    #     "[Imaged Nucleus]",
    #     "[Number of Averages]",
    #     "[Phase FOV Percent]",
    #     "[Sampling Percent]",
    #     "[Prospective Motion Correction]",
    #     "[Window Center]",
    #     "[Window Width]",
    #     "[Number of Chemical Shift]",
    #     "[Repetition Time]",
    #     "[Syncra Scan Type]",
    #     "[Diffusion Direction RL]",
    #     "[Diffusion Direction AP]",
    #     "[Diffusion Direction FH]",
    #     "Presentation LUT Shape",
    #     "Frame Type",
    #     "Creator-Version UID",
    #     "Pixel Presentation",
    #     "Volumetric Properties",
    #     "Volume Based Calculation Technique",
    #     "Complex Image Component",
    #     "Acquisition Contrast",
    #     "Contrast/Bolus Volume",
    #     "Contrast/Bolus Ingredient Concentration",
    #     "Pulse Sequence Name",
    #     "Echo Pulse Sequence",
    #     "Inversion Recovery",
    #     "Multiple Spin Echo",
    #     "Multi-planar Excitation",
    #     "Phase Contrast",
    #     "Time of Flight Contrast",
    #     "Spoiling",
    #     "Steady State Pulse Sequence",
    #     "Echo Planar Pulse Sequence",
    #     "Magnetization Transfer",
    #     "T2 Preparation",
    #     "Blood Signal Nulling",
    #     "Saturation Recovery",
    #     "Spectrally Selected Suppression",
    #     "Spectrally Selected Excitation",
    #     "Spatial Pre-saturation",
    #     "Tagging",
    #     "Oversampling Phase",
    #     "Geometry of k-Space Traversal",
    #     "Segmented k-Space Traversal",
    #     "Rectilinear Phase Encode Reordering",
    #     "Tag Thickness",
    #     "Partial Fourier Direction",
    #     "Cardiac Synchronization Technique",
    #     "Receive Coil Manufacturer Name",
    #     "Receive Coil Type",
    #     "Quadrature Receive Coil",
    #     "Multi-Coil Element Name",
    #     "Multi-Coil Element Used",
    #     "Transmit Coil Type",
    #     "Chemical Shift Reference",
    #     "MR Acquisition Frequency Encoding Steps",
    #     "De-coupling",
    #     "k-space Filtering",
    #     "Parallel Reduction Factor In-plane",
    #     "Diffusion Directionality",
    #     "Parallel Acquisition",
    #     "Parallel Acquisition Technique",
    #     "Inversion Times",
    #     "Metabolite Map Description",
    #     "Partial Fourier",
    #     "Velocity Encoding Direction",
    #     "Velocity Encoding Minimum Value",
    #     "Number of k-Space Trajectories",
    #     "Frequency Correction",
    #     "Diffusion Anisotropy Type",
    #     "Parallel Reduction Factor out-of-plane",
    #     "Parallel Reduction Factor Second In-plane",
    #     "Respiratory Motion Compensation Technique",
    #     "Respiratory Signal Source",
    #     "Bulk Motion Compensation Technique",
    #     "Applicable Safety Standard Agency",
    #     "Specific Absorption Rate Definition",
    #     "Gradient Output Type",
    #     "Specific Absorption Rate Value",
    #     "Gradient Output",
    #     "Water Referenced Phase Correction",
    #     "MR Spectroscopy Acquisition Type",
    #     "MR Acquisition Phase Encoding Steps in-plane",
    #     "MR Acquisition Phase Encoding Steps out-of-plane",
    #     "RF Echo Train Length",
    #     "Gradient Echo Train Length",
    #     "Frame Laterality",
    #     "Respiratory Interval Time",
    #     "Nominal Respiratory Trigger Delay Time",
    #     "LUT Explanation",
    #     "Data Point Rows",
    #     "Data Point Columns",
    #     "LUT Label",
    #     "Coverage of k-Space",
    #     "Flow Compensation Direction",
    #     "Conversion Type",
    #     "Patient Orientation",
    #     "Planar Configuration",
    #     "Retrieve AE Title",
    #     "Study Priority ID",
    #     "Data Set Type",
    #     "Data Set Subtype",
    #     "[Unique Identifier]",
    #     "[Organ]",
    #     "[Patient Name]",
    #     "[Key Images]",
    #     "[Image Number]",
    #     "[Tamar Study Body Part]",
    #     "[Tamar Site Id]",
    #     "[Tamar Translate Flags]",
    #     "[Tamar Exe Software Version]",
    #     "[Tamar Study Has Sticky Note]",
    #     "Date of Secondary Capture",
    #     "Time of Secondary Capture",
    #     "Confidentiality Code",
    #     "Contrast/Bolus Total Dose",
    #     "Contrast/Bolus Ingredient",
    #     "Filler Order Number / Imaging Service Request",
    #     "Image Position",
    #     "Image Orientation",
    #     "Location",
    #     "Image Geometry Type",
    #     "Masking Image",
    #     "Acquisitions in Series",
    #     "Image Dimensions",
    #     "Image Format",
    #     "Image Location",
    #     "Patient's Birth Name",
    #     "Patient's Mother's Birth Name",
    #     "Country of Residence",
    #     "Smoking Status",
    #     "Patient's Religious Preference",
    #     "[number of images in series]",
    #     "[Tamar Study Status]",
    #     "[Tamar Study Age]",
    #     "[Institution Code]",
    #     "[Source AE]",
    #     "[Deferred Validation]",
    #     "[Series Owner]",
    #     "Pixel Aspect Ratio",
    #     "Requested Contrast Agent",
    #     "Performed Procedure Step Status",
    #     "Performed Procedure Type Description",
    #     "Requested Procedure ID",
    #     "Requested Procedure Priority",
    #     "Requested Procedure Location",
    #     "Names of Intended Recipients of Results",
    #     "Requested Procedure Comments",
    #     "Placer Order Number / Imaging Service Request (Retired)",
    #     "Filler Order Number / Imaging Service Request (Retired)",
    #     "Order Entered By",
    #     "[Series Type]",
    #     "Special Needs",
    #     "[Tamar Compression Type]",
    #     "[MedCom OOG Type]",
    #     "[MedCom OOG Version]",
    #     "Overlay Rows",
    #     "Overlay Columns",
    #     "Number of Frames in Overlay",
    #     "Overlay Description",
    #     "Overlay Type",
    #     "Overlay Origin",
    #     "Image Frame Origin",
    #     "Overlay Bits Allocated",
    #     "Overlay Bit Position",
    #     "Convolution Kernel",
    #     "Randoms Correction Method",
    #     "Scatter Correction Method",
    #     "Decay Factor",
    #     "Number of Frames",
    #     "Red Palette Color Lookup Table Descriptor",
    #     "Green Palette Color Lookup Table Descriptor",
    #     "Blue Palette Color Lookup Table Descriptor",
    #     "[Reject Image Flag]",
    #     "[Significant Flag]",
    #     "[Confidential Flag]",
    #     "[Assigning Authority For Patient ID]",
    #     "[GE IIS Compression ID]",
    #     "[Original Study Instance UID]",
    #     "[Patient's Name]",
    #     "[Study Description]",
    #     "[Referring Physician's Name]",
    #     "[Requesting Physician's Name]",
    #     "[Reason for Study]",
    #     "[Study Comments]",
    #     "[Performing Physician's Name]",
    #     "Smallest Pixel Value in Series",
    #     "Largest Pixel Value in Series",
    #     "Contrast/Bolus Route",
    #     "Acquisition DateTime",
    #     "Beat Rejection Flag",
    #     "Content Qualification",
    #     "Burned In Annotation",
    #     "Secondary Capture Device ID",
    #     "Secondary Capture Device Manufacturer",
    #     "Secondary Capture Device Manufacturer's Model Name",
    #     "Secondary Capture Device Software Versions",
    #     "Digital Image Format Acquired",
    #     "B1rms",
    #     "Relative Opacity",
    #     "Last Menstrual Date",
    #     "Temporal Position Index",
    #     "Lossy Image Compression Ratio",
    #     "Lossy Image Compression Method",
    #     "Scheduled Study Start Date",
    #     "Scheduled Study Start Time",
    #     "Query/Retrieve Level",
    #     "Instance Availability",
    #     "Skip Beats",
    #     "[Breast Box x0]",
    #     "Tag Angle First Axis",
    #     "Tag Spacing First Dimension",
    #     "Transmit Coil Manufacturer Name",
    #     "De-coupled Nucleus",
    #     "De-coupling Method",
    #     "Time Domain Filtering",
    #     "Cardiac Signal Source",
    #     "Cardiac Beat Rejection Technique",
    #     "Diffusion b-value XX",
    #     "Diffusion b-value XY",
    #     "Diffusion b-value XZ",
    #     "Diffusion b-value YY",
    #     "Diffusion b-value YZ",
    #     "Diffusion b-value ZZ",
    #     "Respiratory Trigger Delay Threshold",
    #     "Signal Domain Columns",
    #     "Data Representation",
    #     "Resonant Nucleus",
    #     "Referring Physician's Telephone Numbers",
    #     "Study Read Date",
    #     "Study Read Time",
    #     "[Private Creator]",
    #     "Number of Series Related Instances",
    #     "[Original Series Instance UID]",
    #     "[Original SOP Instance UID]",
    #     "Storage Media File-set UID",
    #     "[Breast Box y0]",
    #     "[Breast Box x1]",
    #     "[Breast Box y1]",
    #     "Issuer of Patient ID",
    #     "[Stentor Remote AETitle Element]",
    #     "[Stentor Local AETitle Element]",
    #     "[Stentor Transfer Syntax Value]",
    #     "[Patient's Name Single Byte]",
    #     "[Requesting Physician's Name Single Byte]",
    #     "[Window Smoothing Taste]",
    #     "[GL TrafoType]",
    #     "[Retrospective Motion Correction]",
    #     "[Functional Proc Group Name]",
    #     "[Functional Processing Name]",
    #     "[Bias of Functional Image]",
    #     "[Scale of Functional Image]",
    #     "[Length of Parameters String]",
    #     "[Store Parameters string, delimited by character ESC=0x1B (27)]",
    #     "[Functional Image Version]",
    #     "[Store Color Ramp]",
    #     "[Store Width of Functional Image]",
    #     "[Store level of Functional Image]",
    #     "[Analysis Package]",
    #     "[Original Study UID]",
    #     "[NPW factor]",
    #     "[Referring Physician's Name Single Byte]",
    #     "Patient's Birth Time",
    #     "Palette Color Lookup Table UID",
    # ]
    #
    # combine_directory_excel_files(
    #     "/home/cavriley/files/source_excel_files_raw",
    #     "/home/cavriley/files/output_excel_files/combined_excel_file_raw.xlsx",
    #     droppable_cols,
    # )

    input_data_frame = pd.read_excel(
        "/home/cavriley/files/output_excel_files/combined_excel_file_raw.xlsx"
    )
    data_dictionary_frame = pd.read_excel(
        "~/files/dcm_files/header_data_dictionary.xlsx"
    )
    data_dictionary_frame = truncate_column_names(
        data_dictionary_frame, max_header_length
    )
    index_field = "FileName"
    output_data_frame = input_data_frame[[index_field]]

    for index, row in data_dictionary_frame.iterrows():
        current_header = row["header_name"][:max_header_length]
        used_in_training_flag = row["used_for_training"]

        if current_header == index_field:
            pass
        elif "drop" in row["action"]:
            print(current_header)
            pass
        elif "keep" in row["action"] and used_in_training_flag == 1:
            output_data_frame[current_header] = pd.Series(
                input_data_frame[current_header]
            )
        elif (
            row["action"] == "one_hot_encoding_from_array"
            and used_in_training_flag == 1
        ):
            encoded_frame = one_hot_encoding_from_array(
                input_data_frame, current_header, index_field
            )
            output_data_frame = pd.merge(
                left=output_data_frame, right=encoded_frame, on=index_field
            )
        elif (
            row["action"] == "one_hot_encoding_from_str_col"
            and used_in_training_flag == 1
        ):
            encoded_frame = one_hot_encoding_from_str_col(
                input_data_frame, current_header
            )
            encoded_frame[index_field] = input_data_frame[[index_field]]
            output_data_frame = pd.merge(
                left=output_data_frame, right=encoded_frame, on=index_field
            )
            # pass
        elif row["action"] == "convert_array_to_index_value":
            pass
        encoded_frame = None
    corr_matrix = output_data_frame[output_data_frame.columns[1:]].corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print(len(to_drop))
    # Drop features
    output_data_frame.drop(to_drop, axis=1, inplace=True)
    print(output_data_frame)
    output_data_frame.to_excel(
        "~/files/output_excel_files/testing_data2.xlsx", engine="openpyxl"
    )
