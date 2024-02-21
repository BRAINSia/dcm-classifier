#!/usr/bin/env python3

import re

import pandas as pd
from pathlib import Path

max_heder_length: int = 24


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
    frames = []
    for file in Path(directory).rglob("*.xlsx"):
        print(file)
        frames.append(pd.read_excel(file))

    combined_frame = pd.concat(frames)
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
    out_put_frame = frame[[index_field]]
    image_type_col = df[[col_name]]
    header_list = []
    # get all the unique image types
    unique_image_types = image_type_col[col_name].unique()
    for image_type in unique_image_types:
        if not isinstance(image_type, str):
            continue
        row_contents = re.findall(r"\b\w+\b", image_type)
        if len(row_contents) > 0:
            for type_element in row_contents:
                if type_element not in header_list:
                    header_list.append(type_element)
    print(len(header_list))
    print(header_list)
    for header in header_list:
        series = pd.Series(
            (df[col_name].str.contains(header)).fillna(0).astype(int),
            name=f"{col_name}_{header}",
        )
        out_put_frame = out_put_frame.merge(series, left_index=True, right_index=True)

    # drop the image type column from the dataframe
    return out_put_frame


def one_hot_encoding_from_col_str(frame: pd.DataFrame, col_name: str) -> pd.DataFrame:
    pass


def parse_columns(file: str):
    frame = pd.read_excel(file)

    # headers: pd.DataFrame = pd.DataFrame(
    #     {
    #         "Header Name": drop_cols,
    #         "Action": ["Dropped"] * len(drop_cols),
    #         "Reasons": ["Less than 5% of values unique"] * len(drop_cols),
    #     }
    # )
    frame = identify_and_drop_unusable_cols(frame)
    frame = one_hot_encoding_from_array(frame, "ImageType")
    frame.to_excel(
        "~/programs/dcm_files/combined_encoded_data.xls",
        sheet_name="results",
        engine="openpyxl",
    )
    # new_frame = pd.DataFrame()
    # for i, col in enumerate(frame.columns):
    #     frame[col].name != "FileName":
    #         if frame[col].name == "ImageType":
    #             col_dict = get_col_dict_from_Image_Type(frame[col])
    #             new_frame = pd.concat([new_frame, pd.DataFrame(col_dict)], axis=1)
    #         first_val = frame[col].iloc[1]
    #         header_info: dict = {
    #             "Header Name": frame[col].name,
    #             "Action": "Kept",
    #             "Reasons": "",
    #         }
    # if the first value of the column is a string then break it up into multiple columns
    # if type(first_val) is str:
    #     encoding = pd.get_dummies(frame[col], dtype=int)
    #     new_frame = pd.concat([new_frame, encoding], axis=1)

    # new_frame.drop_duplicates(inplace=True)
    # print(new_frame)
    # new_frame.to_excel(
    #     "~/programs/dcm_files/combined_encoded_data.xls",
    #     sheet_name="results",
    #     engine="openpyxl",
    # )


# drops columns that have less than 5% of values NaN and less than 4 unique values
def identify_and_drop_unusable_cols(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame
    droppable_cols = []
    length = len(df)
    for col in df.columns:
        if (df[col].count() / length) <= 0.05 or df[col].nunique() < 2:
            droppable_cols.append(col)
            dropped_cols.append(col)
    # droppable_cols.append("list_of_ordered_volume_files")
    # droppable_cols.append("Image Type")
    print(f"dropping {len(droppable_cols)} cols")
    [print(x) for x in sorted(droppable_cols)]
    return df.drop(columns=droppable_cols)


def truncate_column_names(df: pd.DataFrame, max_length: int) -> pd.DataFrame:
    for col in df.columns:
        df[col].name = col.replace(col, col[:24])
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
    # parse_columns("~/programs/dcm_files/all_dicom_combined_data.xlsx")
    input_data_frame = pd.read_excel(
        "~/programs/dcm_files/all_dicom_combined_data.xlsx"
    )
    input_data_frame = truncate_column_names(input_data_frame, max_heder_length)
    identify_and_drop_unusable_cols(input_data_frame)
    data_dictionary_frame = pd.read_excel(
        "~/programs/dcm_files/header_data_dictionary.xlsx"
    )
    data_dictionary_frame = truncate_column_names(
        data_dictionary_frame, max_heder_length
    )
    index_field = "FileName"
    output_data_frame = input_data_frame[[index_field]]

    for index, row in data_dictionary_frame.iterrows():
        current_header = row["header_name"][:max_heder_length]
        if current_header == index_field:
            pass
        elif row["action"] == "keep":
            output_data_frame[current_header] = pd.Series(
                input_data_frame[current_header]
            )
        elif row["action"] == "drop":
            pass
        elif row["action"] == "one_hot_encoding_from_array":
            encoded_frame = one_hot_encoding_from_array(
                input_data_frame, current_header, index_field
            )
            output_data_frame = pd.merge(
                left=output_data_frame, right=encoded_frame, on=index_field
            )
        elif row["action"] == "one_hot_encoding_from_col_str":
            pass

    output_data_frame.to_excel(
        "~/programs/dcm_files/merged_output_file.xlsx", engine="openpyxl"
    )
