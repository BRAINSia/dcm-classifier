import pandas as pd
from pathlib import Path

features = [
    "Diffusionb-valueCount",
    "Diffusionb-valueMax",
    "Echo Number(s)",
    "Echo Time",
    "Echo Train Length",
    "Flip Angle",
    "Image Type",
    "Imaging Frequency",
    "In-plane Phase Encoding Direction",
    "Inversion Time",
    "MR Acquisition Type",
    "Manufacturer",
    "Number of Averages",
    "Pixel Bandwidth",
    "Repetition Time",
    "SAR",
    "Scanning Sequence",
    "Sequence Variant",
    "Variable Flip Angle Flag",
    "dB/dt",
]


def remove_rows_with_duplicate_values(input_path: str, outpath: str) -> None:
    """
    Remove rows with duplicate values from a pandas dataframe
    """
    df = pd.read_excel(input_path)

    # duplicates = df[df.duplicated(features, keep="first")]
    # df = df[~df["Unnamed: 0"].isin(duplicates["Unnamed: 0"])]
    df = df[~df.duplicated(features, keep="first")]
    print(df.shape)
    df = df.rename(columns={"Diffusionb-valueCount": "Diffusionb-valueBool"})

    df.to_excel(outpath, index=False)


if __name__ == "__main__":
    input_path = "/tmp/dicom_data/combined_predicthd_data_Jan9.xlsx"
    output_path = "/tmp/dicom_data/no_duplicates/combined_predicthd_data_Feb26.xlsx"
    remove_rows_with_duplicate_values(input_path, output_path)
