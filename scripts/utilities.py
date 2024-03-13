from pathlib import Path
import pandas as pd


def combine_all_excel_files(excel_files: list[Path] | list[pd.DataFrame]) -> pd.DataFrame:
    """
    combine all excel files into one dataframe

    Args:
        excel_files: list[Path] | list[pd.DataFrame] either a list of dataframe or a list of paths to excel files

    Returns:
        pd.DataFrame with all dataframes combined
    """

    # available_columns = set()
    all_dataframes = []
    if not type(excel_files[0]) is pd.DataFrame:
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
    else:
        all_dataframes = excel_files

    big_dataframe: pd.DataFrame = pd.concat(
        all_dataframes, axis=0, ignore_index=True
    )
    return big_dataframe


removal_features = [
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

    Args:
        input_frame: str | pd.DataFrame: either a path to an excel file or a pandas dataframe
        outpath: str: path to save the dataframe to, if not provided, the dataframe is returned
        save_to_excel: bool: whether to save the dataframe to an excel file

    Returns:
        None | pd.DataFrame: if save_to_excel is True, None is returned, otherwise the dataframe is returned
    """
    if type(input_frame) is str:
        df = pd.read_excel(input_frame)
    else:
        df = input_frame

    try:
        df = df[~df.duplicated(removal_features, keep="first")]
    except KeyError:
        print("KeyError in removing duplicate rows")

    if save_to_excel and outpath is not None:
        df.to_excel(outpath, index=False)
    else:
        return df


def identify_unusable_cols(frame: pd.DataFrame) -> list[str]:
    """
    Identify columns that have minimal values and can likely be removed

    Args:
        frame: pd.DataFrame: dataframe to identify columns to remove

    Returns:
        list[str]: list of columns that can be removed
    """
    df = frame
    droppable_cols = []
    length = len(df)
    for col in df.columns:
        if (df[col].count() / length) <= 0.05 or df[col].nunique() < 2:
            droppable_cols.append(col)

    return droppable_cols


def merge_labels_and_training_data(labels: (str | Path) | pd.DataFrame, training_data: (str | Path) | pd.DataFrame, save_to_excel: bool = True) -> pd.DataFrame | None:
    """
    Merge labels and training data into one dataframe

    Args:
        labels: (str | Path) | pd.DataFrame: either a path to an excel file or a pandas dataframe
        training_data: (str | Path) | pd.DataFrame: either a path to an excel file or a pandas dataframe
        save_to_excel: bool: whether to save the dataframe to an excel file

    Returns:
        pd.DataFrame | None: if save_to_excel is True and the training_data is a path to an excel file, None is returned, otherwise the dataframe is returned

    """
    final_df_name = None

    # check if the input is a dataframe or a path to an excel file
    if type(labels) is pd.DataFrame:
        label_df = labels
    else:
        df = pd.read_excel(labels, index_col=0)
        label_df = df[
            ["FileName", "label"]
        ]

    # check if the input is a dataframe or a path to an excel file
    if type(training_data) is pd.DataFrame:
        trainings_df = training_data
    else:
        trainings_df = pd.read_excel(training_data, index_col=0)
        final_df_name = training_data.replace(".xlsx", "_labeled.xlsx")

    # merge on "FileName"
    merged_df = pd.merge(
        left=trainings_df,
        right=label_df,
        how="left",
        left_on="FileName",
        right_on="FileName",
    )

    # save to excel if save_to_excel is True
    if save_to_excel and final_df_name is not None:
        merged_df.to_excel(final_df_name, index=False)
    else:
        return merged_df
