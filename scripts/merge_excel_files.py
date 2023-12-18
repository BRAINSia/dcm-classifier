import argparse
import re

# from dcm_classifier.
import pandas as pd
from pathlib import Path
import numpy as np
import xlrd

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

    #
    merge_excel_files(
        "~/programs/dcm_files/bids_image_information.xlsx",
        "~/programs/dcm_files/combined.xls",
        "~/programs/dcm_files/combined_merged.xls",
    )
