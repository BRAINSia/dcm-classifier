import pandas as pd


def add_volume_label(input_frame: str, output_frame: str):
    df = pd.read_excel(input_frame)
    # Removed non MR modality rows
    df = df[df["Modality"] == "MR"]

    # Add volume label column
    df["volume_label"] = ""

    # Set dwig and tracew volume labels to bval_img
    df.loc[
        df["_dcm_series_type"] == "dwig",
        "volume_label",
    ] = "bval_img"

    df.loc[
        df["_dcm_series_type"] == "tracew",
        "volume_label",
    ] = "bval_img"

    modalities = ["t1w", "t2w", "adc", "fa", "flair"]
    for modality in modalities:
        # Set volume label for each modality
        df.loc[df["_dcm_series_type"] == modality, "volume_label"] = modality

    df.to_excel(output_frame, sheet_name="results", engine="openpyxl")


if __name__ == "__main__":
    input_frame = "/tmp/dicom_data_tables_Michal/combined_data_without_iowa_Dec19.xlsx"
    output_frame = (
        "/home/cavriley/files/output_excel_files/combined_labeled_file_Dec20.xlsx"
    )
    add_volume_label(input_frame, output_frame)
