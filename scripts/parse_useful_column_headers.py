import re
import pandas as pd
from dcm_classifier.namic_dicom_typing import convert_array_to_index_value


max_header_length: int = 34


def identify_unusable_cols(frame: pd.DataFrame) -> list:
    df = frame
    droppable_cols = []
    length = len(df)
    for col in df.columns:
        if (df[col].count() / length) <= 0.05 or df[col].nunique() < 2:
            droppable_cols.append(col)
    # droppable_cols.append("list_of_ordered_volume_files")
    # droppable_cols.append("Image Type")
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
                (frame[col_name].str.contains(manufacturer)).fillna(0).astype(int),
                name=f"{col_name}_{manufacturer}",
            )
            output_frame = output_frame.merge(series, left_index=True, right_index=True)
    else:
        for value in possible_values:
            if type(value) is str:
                series = pd.Series(
                    (frame[col_name].str.contains(value)).fillna(0).astype(int),
                    name=f"{col_name}_{value.replace(' ', '_')}",
                )
                # output_frame[f"{col_name}_{value.replace(' ', '_')}"] = frame[col_name].str.contains(value).astype(int)
                output_frame = output_frame.merge(
                    series, left_index=True, right_index=True
                )
    # encoding = pd.get_dummies(frame[col_name], dtype=int)
    # for name in encoding.columns:
    #     encoding.rename(columns={name: f"{col_name}_{name}"}, inplace=True)
    return output_frame


def one_hot_encoding_from_array_floats(
    frame: pd.DataFrame, col_name: str
) -> pd.DataFrame:
    convert_array_to_index_value()


def truncate_column_names(df: pd.DataFrame, max_length: int) -> pd.DataFrame:
    for col in df.columns:
        df[col].name = col.replace(col, col[:max_length])
    return df


def parse_column_headers(
    header_dictionary_path: str, input_path: str, output_path: str
):
    data_dictionary_frame = pd.read_excel(header_dictionary_path)
    data_dictionary_frame = truncate_column_names(
        data_dictionary_frame, max_header_length
    )
    index_field = "FileName"

    input_data_frame = pd.read_excel(input_path)
    output_data_frame = input_data_frame[[index_field]]

    for index, row in data_dictionary_frame.iterrows():
        current_header = row["header_name"][:max_header_length]
        used_in_training_flag = row["used_for_training"]

        if current_header == index_field:
            pass
        elif "drop" in row["action"]:
            pass
        elif "keep" in row["action"] and used_in_training_flag == 1:
            try:
                output_data_frame[current_header] = pd.Series(
                    input_data_frame[current_header]
                )
            except KeyError:
                pass
                # print(f"KeyError: {current_header}, in {file}")
        elif (
            row["action"] == "one_hot_encoding_from_array"
            and used_in_training_flag == 1
        ):
            try:
                encoded_frame = one_hot_encoding_from_array(
                    input_data_frame, current_header, index_field
                )
                output_data_frame = pd.merge(
                    left=output_data_frame, right=encoded_frame, on=index_field
                )
            except KeyError:
                # print(f"KeyError: {current_header}, in {file}")
                pass

        elif (
            row["action"] == "one_hot_encoding_from_str_col"
            and used_in_training_flag == 1
        ):
            try:
                encoded_frame = one_hot_encoding_from_str_col(
                    input_data_frame, current_header, index_field
                )
                output_data_frame = pd.merge(
                    left=output_data_frame, right=encoded_frame, on=index_field
                )
            except KeyError:
                # print(f"KeyError: {current_header}, in {file}")
                pass

        # elif row["action"] == "one_hot_encoding_from_array_floats":
        #     encoded_frame = one_hot_encoding_from_array_floats(
        #         input_data_frame, current_header, index_field
        #     )
        #     # encoded_frame[index_field] = input_data_frame[[index_field]]
        #     output_data_frame = pd.merge(
        #         left=output_data_frame, right=encoded_frame, on=index_field
        #     )
        # combined_frame = pd.concat([combined_frame, output_data_frame])

    output_data_frame.to_excel(output_path, sheet_name="results", engine="openpyxl")


if __name__ == "__main__":
    header_data_dictionary_frame = "~/files/dcm_files/header_data_dictionary.xlsx"
    input_file = (
        "/home/cavriley/files/all_excel_files_combined/combined_excel_file_raw.xlsx"
    )
    output_file = (
        "/home/cavriley/files/all_excel_files_combined/testing_data_Dec18.xlsx"
    )
    parse_column_headers(header_data_dictionary_frame, input_file, output_file)
