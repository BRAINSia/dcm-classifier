import pandas as pd
from pathlib import Path


def combine_directory_excel_files(directory: str, output_file: str):
    combined_frame = pd.DataFrame()
    for file in Path(directory).glob("*.xlsx"):
        print(file)
        current_frame = pd.read_excel(file)
        combined_frame = pd.concat([combined_frame, current_frame])

    print("Excel Files Read")

    combined_frame.to_excel(output_file, sheet_name="results", engine="openpyxl")


if __name__ == "__main__":
    excel_files_directory: str = ""
    output_file: str = ""
    combine_directory_excel_files(excel_files_directory, output_file)
