#!/usr/bin/env python3

from pathlib import Path
import pandas


import pandas as pd


def get_all_column_names(excel_files: list[Path]) -> set[str]:
    # Get all column names from all excel files

    # available_columns = set()
    all_dataframes = []
    for excel_file in excel_files:
        if "~$" in excel_file.as_posix():
            # Skip temp files
            continue
        print(f"Processing {excel_file}")

        with open(excel_file, "rb") as file:
            file_extension = excel_file.suffix
            if file_extension == ".xlsx":
                df = pd.read_excel(file.read(), engine="openpyxl")
            elif file_extension == ".xls":
                df = pd.read_excel(file.read())
            # elif file_extension == ".csv":
            #     df = pd.read_csv(file.read())
            else:
                raise ValueError(f"Unknown file extension {file_extension}")

            all_dataframes.append(df)

            # curr_columns = set(df.columns)
            # new_columns = curr_columns - available_columns
            # print(new_columns)
            # for col in new_columns:
            #     available_columns.add(col)

    big_dataframe: pandas.DataFrame = pd.concat(
        all_dataframes, axis=0, ignore_index=True
    )
    return big_dataframe


if __name__ == "__main__":

    home_dir = Path.home()
    excel_files: list[Path] = [
        x for x in (home_dir / "dicom_data_tables").glob("*.xls*")
    ]

    all_column_names: pd.DataFrame = get_all_column_names(excel_files)

    all_column_names.to_excel(home_dir / "all_dicom_combined_data.xlsx")
