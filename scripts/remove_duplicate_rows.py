import pandas as pd
from pathlib import Path

features = [
    # "Diffusionb-valueCount",
    # "Diffusionb-valueMax",
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


def remove_rows_with_duplicate_values(input_frame: str | pd.DataFrame, outpath: str = None, save_to_excel: bool = True) -> None | pd.DataFrame:
    """
    Remove rows with duplicate values from a pandas dataframe
    """
    if type(input_frame) is str:
        df = pd.read_excel(input_frame)
    else:
        df = input_frame

    try:
        df = df[~df.duplicated(features, keep="first")]
    except KeyError:
        print("KeyError in removing duplicate rows")

    if save_to_excel:
        df.to_excel(outpath, index=False)
    else:
        return df


if __name__ == "__main__":
    input_path = ""
    output_path = ""
    remove_rows_with_duplicate_values(input_path, output_path)
