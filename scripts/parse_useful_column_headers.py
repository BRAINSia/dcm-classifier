import re
import pandas as pd
from pathlib import Path


max_header_length: int = 34


def identify_unusable_cols(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame
    droppable_cols = []
    length = len(df)
    for col in df.columns:
        if (df[col].count() / length) <= 0.05 or df[col].nunique() < 2:
            droppable_cols.append(col)

    print(f"dropping {len(droppable_cols)} cols")
    [print(x) for x in sorted(droppable_cols)]
    return df.drop(columns=droppable_cols)


def one_hot_encoding_from_array(
    frame: pd.DataFrame, col_name: str, index_field: str
) -> pd.DataFrame:
    df = frame
    output_frame = frame[[index_field]]
    column = df[[col_name]]
    header_list = []
    # get all the unique column
    unique_values = column[col_name].unique()
    for value in unique_values:
        if type(value) is not str:
            continue
        row_contents = re.findall(r"\b\w+\b", value)
        if len(row_contents) > 0:
            for type_element in row_contents:
                if type_element not in header_list and "_" not in type_element:
                    header_list.append(type_element)
                else:
                    elements = type_element.split("_")
                    for element in elements:
                        if element not in header_list:
                            header_list.append(element)

    for header in header_list:
        series = pd.Series(
            (df[col_name].str.contains(header)).fillna(0).astype(int),
            name=f"{col_name}_{header}",
        )
        output_frame = output_frame.merge(series, left_index=True, right_index=True)

    return output_frame


def one_hot_encoding_from_str_col(
    frame: pd.DataFrame, col_name: str, index_field: str
) -> pd.DataFrame:
    output_frame = frame[[index_field]]
    possible_values = frame[col_name].unique()
    if col_name == "Manufacturer":
        manufacturers = ["siemens", "ge", "philips", "toshiba"]
        replace_dict = {}
        for value in possible_values:
            if type(value) is str:
                for manufacturer in manufacturers:
                    if manufacturer in value.lower():
                        replace_dict[value] = manufacturer
                        break
                    else:
                        replace_dict[value] = "other"

        frame[col_name].replace(replace_dict, inplace=True)

        for manufacturer in manufacturers + ["other"]:
            series = pd.Series(
                (frame[col_name].str.contains(manufacturer)).fillna(0).astype("int16"),
                name=f"{col_name}_{manufacturer}",
            )
            output_frame = output_frame.merge(series, left_index=True, right_index=True)
    else:
        for value in possible_values:
            if type(value) is str:
                series = pd.Series(
                    (frame[col_name].str.contains(value)).fillna(0).astype("int16"),
                    name=f"{col_name}_{value.replace(' ', '_')}",
                )
                output_frame = output_frame.merge(
                    series, left_index=True, right_index=True
                )

    return output_frame


def truncate_column_names(df: pd.DataFrame, max_length: int) -> pd.DataFrame | None:
    for col in df.columns:
        df[col].name = col.replace(col, col[:max_length])
    return df


def parse_column_headers(
    header_dictionary_df: pd.DataFrame,
    input_file: (str | Path) | pd.DataFrame,
    output_path: str | Path = None,
    save_to_excel: bool = True,
) -> pd.DataFrame | None:
    index_field = "FileName"

    if isinstance(input_file, pd.DataFrame):
        input_df = input_file
    else:
        input_df: pd.DataFrame = pd.read_excel(input_file)

    # remove the rows that have no file name
    input_df = input_df[input_df["FileName"].notna()]
    new_data_frame = input_df[[index_field]]

    for index, row in header_dictionary_df.iterrows():
        current_header = row["header_name"][:max_header_length]

        if "keep" in row["action"]:
            try:
                new_column_series = pd.Series(input_df[current_header]).reset_index(drop=True)
                new_data_frame[current_header] = new_column_series
            except KeyError:
                print(f"KeyError: {current_header}")
        elif row["action"] == "one_hot_encoding_from_array":
            try:
                encoded_frame = one_hot_encoding_from_array(
                    input_df, current_header, index_field
                )
                new_data_frame = pd.merge(
                    left=new_data_frame, right=encoded_frame, on=index_field
                )
            except KeyError:
                print(f"KeyError: {current_header}")

        elif row["action"] == "one_hot_encoding_from_str_col":
            try:
                encoded_frame = one_hot_encoding_from_str_col(
                    input_df, current_header, index_field
                )
                new_data_frame = pd.merge(
                    left=new_data_frame, right=encoded_frame, on=index_field
                )
            except KeyError:
                print(f"KeyError: {current_header}")
    if save_to_excel:
        new_data_frame.to_excel(output_path, index=False)
    else:
        return new_data_frame


if __name__ == "__main__":
    base_dir = ""
    header_data_dictionary_frame = f"{base_dir}/path_to_header_dictionary"
    header_df = pd.read_excel(header_data_dictionary_frame)

    # process header df
    clean_header_df = header_df[header_df["used_for_training"] == 1]
    clean_header_df = clean_header_df[clean_header_df["action"] != "drop"]
    clean_header_df = truncate_column_names(clean_header_df, max_header_length)

    # delete orig df
    del header_df

    output_file = f"{base_dir}/"
    input_file = f"{base_dir}/"

    parse_column_headers(
        header_dictionary_df=clean_header_df,
        input_file=input_file,
        output_path=output_file,
    )
