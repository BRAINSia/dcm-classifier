from pathlib import Path
import pandas as pd
from parse_useful_column_headers import *


def label_150(data: pd.DataFrame | str) -> pd.DataFrame:
    if isinstance(data, str):
        data = pd.read_excel(data)

    for index, row in data.iterrows():
        if "ADC" in row["FileName"]:
            data.at[index, "label"] = "adc"
        elif "DWI" in row["FileName"]:
            data.at[index, "label"] = "tracew"
        elif "T1" in row["FileName"]:
            data.at[index, "label"] = "t1w"
        elif "T2" in row["FileName"]:
            data.at[index, "label"] = "t2w"
        else:
            data.at[index, "label"] = "other"

    return data



if __name__ == "__main__":
    # path_150 = "/tmp/dicom_data/150_Test_DATA.xlsx"
    #
    # header_dict_path = "/home/cavriley/files/dcm_files/header_data_dictionary_Dec18.xlsx"
    # header_frame = pd.read_excel(header_dict_path)
    # training_150_sheet = parse_column_headers(header_dictionary_df=header_frame, input_file=path_150, output_path="/tmp/dicom_data/150_training.xlsx", header=True)

    labeled_150_frame: pd.DataFrame = label_150("/tmp/dicom_data/150_training.xlsx")

    print(labeled_150_frame)
    # labeled_150_frame.to_excel("/tmp/dicom_data/150_training_labeled.xlsx", index=False)
    combined_labeled_sheet_path = "/tmp/dicom_data/combined_all_training_data_Mar13_labeledNewField_augmented.xlsx"
    #
    combined_frame = pd.read_excel(combined_labeled_sheet_path)
    combined_frame = pd.concat([combined_frame, labeled_150_frame], axis=0, ignore_index=True)
    #
    combined_frame.to_excel("/tmp/dicom_data/combined_all_training_data_Mar13_labeledNewField_augmented_with_150.xlsx", index=False)
