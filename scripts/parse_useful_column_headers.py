import re
import pandas as pd
from pathlib import Path
from src.dcm_classifier.namic_dicom_typing import convert_array_to_index_value


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

    # def parse_column_headers(
    #     header_dictionary_path: str, input_path: str, output_path: str
    # ):
    #     data_dictionary_frame = pd.read_excel(header_dictionary_path)
    #     data_dictionary_frame = truncate_column_names(
    #         data_dictionary_frame, max_header_length
    #     )
    #     index_field = "FileName"
    #
    #     input_data_frame = pd.read_csv(input_path)
    #     output_data_frame = input_data_frame[[index_field]]
    #
    #     for index, row in data_dictionary_frame.iterrows():
    #         current_header = row["header_name"][:max_header_length]
    #         used_in_training_flag = row["used_for_training"]


def parse_column_headers(
    header_dictionary_df: pd.DataFrame,
    input_file,
    output_path: str,
    header: bool,
):
    # training_data_frame_path = "/tmp/dcm_classifier_training_data/dcm_train_data/training/training_iowa_stroke_data_Jan14.xlsx"
    # try:
    #     training_data_frame = pd.read_excel(training_data_frame_path)
    # except FileNotFoundError:
    #     training_data_frame = pd.DataFrame()

    index_field = "FileName"
    print(f"Input: {input_file}")
    input_df = pd.read_excel(input_file)
    # input_df = input_df.iloc[:2]
    input_df = input_df[input_df["FileName"].notna()]
    new_data_frame = input_df[[index_field]]

    for index, row in header_dictionary_df.iterrows():
        current_header = row["header_name"][:max_header_length]
        print(current_header)
        # if current_header == index_field:
        #     pass
        # elif "drop" in row["action"]:
        #     pass
        if "keep" in row["action"]:
            try:
                new_data_frame[current_header] = pd.Series(input_df[current_header])
            except KeyError:
                pass
                # print(f"KeyError: {current_header}, in {file}")
        elif row["action"] == "one_hot_encoding_from_array":
            try:
                encoded_frame = one_hot_encoding_from_array(
                    input_df, current_header, index_field
                )
                new_data_frame = pd.merge(
                    left=new_data_frame, right=encoded_frame, on=index_field
                )
            except KeyError:
                # print(f"KeyError: {current_header}, in {file}")
                pass

        elif row["action"] == "one_hot_encoding_from_str_col":
            try:
                encoded_frame = one_hot_encoding_from_str_col(
                    input_df, current_header, index_field
                )
                new_data_frame = pd.merge(
                    left=new_data_frame, right=encoded_frame, on=index_field
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
    print(new_data_frame.shape)
    new_data_frame.to_excel(output_path, index=False)
    print(f"Output: {output_path}")
    # training_data_frame = pd.concat([training_data_frame, new_data_frame])
    # del new_data_frame
    # training_data_frame.to_excel(training_data_frame_path, index=False)

# rename and normalize diffusion bval col
def set_diffusion_bval_bool(frame: pd.DataFrame) -> pd.DataFrame:
    diffusion_bval_col = "Diffusionb-valueMax"
    diffusion_bval_bool = "Diffusionb-valueCount"

    frame[diffusion_bval_bool] = 1 if (frame[diffusion_bval_col].notna()) and (frame[diffusion_bval_bool] != 0) else 0
    frame.rename(columns={diffusion_bval_bool: "Diffusionb-valueBool"}, inplace=True)
    return frame

if __name__ == "__main__":
    # base_data_dir = "/tmp/dcm_classifier_training_data/dcm_train_data"
    # processed_data_dir = f"{base_data_dir}/processed_site_data"
    # raw_data_dir = f"{base_data_dir}/raw_site_data"
    # header_data_dictionary_frame = f"{base_data_dir}/header_data_dictionary_Dec18.xlsx"

    base_data_dir = "/tmp/dcm_classifier_training_data/dcm_train_data"
    processed_data_dir = f"{base_data_dir}/processed_site_data"
    raw_data_dir = f"{base_data_dir}/raw_site_data"
    header_data_dictionary_frame = f"/home/cavriley/files/dcm_files/header_data_dictionary_Dec18.xlsx"
    header_df = pd.read_excel(header_data_dictionary_frame)

    # process header df
    clean_header_df = header_df[header_df["used_for_training"] == 1]
    clean_header_df = clean_header_df[clean_header_df["action"] != "drop"]
    clean_header_df = truncate_column_names(clean_header_df, max_header_length)

    # delete orig df
    del header_df

    output_dir = f"/tmp/dicom_data/training/"

    # input_dir = f"{base_data_dir}/combined/split_combined_file"
    # for file in Path(input_dir).glob("*.xlsx"):
    #     file_name = file.name
    #     print(f"Input: {file}")
    #     parse_column_headers(clean_header_df, file, output_dir + file_name, header=True)
    input_file = "/tmp/dicom_data/no_duplicates/combined_predicthd_data_Feb26.xlsx"
    input_frame = pd.read_excel(input_file)
    # replace values over 5000 with -12345 (null)
    input_frame[input_frame >= 5000] = -12345

    parse_column_headers(
        clean_header_df,
        input_file,
        output_dir + "combined_training_predicthd_data_Feb26.xlsx",
        header=True,
    )

    # size = input_frame.shape[0]
    # # create 30 chunks
    # chunk_size = size // 30
    # for chunk in range(0, size, chunk_size):
    #     df = input_frame.iloc[chunk : chunk + chunk_size]
    #     parse_column_headers(clean_header_df, df, output_file, header=True)
    #     del df

    #
    # site_file_names = []
    # for site_data in Path(raw_data_dir).glob("*.xlsx"):
    #     site_file_names.append(site_data.name)
    #
    # sites_to_process = []
    # for site_file_name in site_file_names:
    #     if Path(f"{processed_data_dir}/{site_file_name}").exists():
    #         continue
    #     else:
    #         sites_to_process.append(site_file_name)
    #
    # for site_file_name in sites_to_process:
    #     print(site_file_name)
    #     input_file = f"{raw_data_dir}/{site_file_name}"
    #     output_file = f"{processed_data_dir}/{site_file_name}"
    #     parse_column_headers(header_data_dictionary_frame, input_file, output_file)
    #
    # df = pd.read_excel(f"{raw_data_dir}/PHD_024.xlsx")
    # print(df.shape)
    # # print(df.head())

    # input_file = f"/home/mbrzus/programming/dcm_train_data/processed/processed_iowa_stroke_data_Jan12.csv"
    # output_file = f"/home/mbrzus/programming/dcm_train_data/training/training_iowa_stroke_data_Jan12.csv"
    # parse_column_headers(header_data_dictionary_frame, input_file, output_file)
