import pandas as pd

# # make a copy of the dataframe containing only fields needed for heuristic based labeling
# df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/combined/combined_iowa_stroke_data_Jan9.xlsx"
# )
#
# columns_to_keep = [
#     "FileName",
#     "_vol_index",
#     "_dcm_volume_type",
#     "_dcm_volume_orientation_patient",
#     "_dcm_series_number",
#     "_dcm_series_type",
#     "HasDiffusionGradientOrientation",
#     "Diffusionb-valueCount",
#     "Diffusionb-valueMax",
#     "Image Type",
#     "Modality",
#     "Series Description",
# ]
#
# new_df = df[columns_to_keep]
# new_df.to_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_raw.xlsx"
# )

# # get all unique series descriptions
# df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_raw.xlsx"
# )
# series_descriptions = df["Series Description"].unique()
# new_df = pd.DataFrame(series_descriptions, columns=["Series Description"])
# new_df.to_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/unique_series_descriptions_raw.xlsx"
# )


# investigate series descriptions
def update_labels_and_series_desc(series_desc_list: list, label: str):
    # update labels in label final file
    final_df = pd.read_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_final.xlsx"
    )
    for series_desc in series_desc_list:
        final_df.loc[final_df["Series Description"] == series_desc, "label"] = label
    final_df.to_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_final.xlsx",
        index=False,
    )

    # remove rows from remaining file that have label set in the final file
    remaining_df = pd.read_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_remaining.xlsx"
    )
    for series_desc in series_desc_list:
        remaining_df = remaining_df[remaining_df["Series Description"] != series_desc]
    remaining_df.to_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_remaining.xlsx",
        index=False,
    )

    # remove series descriptions from remaining series descriptions file
    series_desc_df = pd.read_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/unique_series_descriptions_remaining.xlsx"
    )
    for series_desc in series_desc_list:
        series_desc_df = series_desc_df[
            series_desc_df["Series Description"] != series_desc
        ]
    series_desc_df.to_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/unique_series_descriptions_remaining.xlsx",
        index=False,
    )


# df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/unique_series_descriptions_remaining.xlsx"
# )
# series_descriptions_list = df["Series Description"].tolist()
#
# # dwig and tracew
# # Notes: dwig series label was computed using a heuristic based on the following:
# # - all volumes corresponding to the series has multiple gradient directions
# # - all vollumes corresponding to the series have b-values
# # - more details about implementation are visible in the src code for the package
#
# # Similarly we label TRACEW components as bval_img
# final_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_final.xlsx"
# )
# final_df.loc[final_df["_dcm_series_type"] == "dwig", "label"] = "bval_vol"
# final_df.loc[final_df["_dcm_series_type"] == "tracew", "label"] = "bval_vol"
# final_df.to_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_final.xlsx",
#     index=False,
# )
#
# # remove rows from remaining file that have label set in the final file
# remaining_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_remaining.xlsx"
# )
# remaining_df = remaining_df[remaining_df["_dcm_series_type"] != "dwig"]
# remaining_df = remaining_df[remaining_df["_dcm_series_type"] != "tracew"]
# remaining_df.to_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_remaining.xlsx",
#     index=False,
# )

# # get all unique series descriptions for dwig _dcm_series_type
# final_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_final.xlsx"
# )
# dwig_series_descriptions = (
#     final_df[final_df["_dcm_series_type"] == "dwig"]["Series Description"]
#     .unique()
#     .tolist()
# )
# tracew_series_descriptions = (
#     final_df[final_df["_dcm_series_type"] == "tracew"]["Series Description"]
#     .unique()
#     .tolist()
# )
# series_descriptions_dwitracew = dwig_series_descriptions + tracew_series_descriptions
# [print(i) for i in series_descriptions_dwitracew]
# print(series_descriptions_dwitracew)
# print(len(series_descriptions_dwitracew))
#
# # check which of the dwig series descriptions are still in the remaining file
# remaining_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_remaining.xlsx"
# )
# list_to_keep = []
# for series_description in series_descriptions_dwitracew:
#     df = remaining_df[remaining_df["Series Description"] == series_description]
#     # print rows if not empty
#     if not df.empty:
#         list_to_keep.append(series_description)
# print("list_to_keep")
# [print(i) for i in list_to_keep]
# # remove fields to keep from dwig series descriptions list
# for item in list_to_keep:
#     series_descriptions_dwitracew.remove(item)
#
# # remove series descriptions from remaining series descriptions file
# series_desc_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/unique_series_descriptions_remaining.xlsx"
# )
# for series_desc in series_descriptions_dwitracew:
#     series_desc_df = series_desc_df[series_desc_df["Series Description"] != series_desc]
# series_desc_df.to_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/unique_series_descriptions_remaining.xlsx",
#     index=False,
# )


# Old model performance on old data is very strong as that what it was trained on. Therefore we can trust most of the predictions.
# Load the original data
final_file_path = "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_final.xlsx"
final_df = pd.read_excel(final_file_path)

# Load the mirror data
remaining_file_path = "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_remaining.xlsx"
remaining_df = pd.read_excel(remaining_file_path)

#  Ensure that the series type and description columns are strings
final_df["_dcm_series_type"] = final_df["_dcm_series_type"].astype(str)
final_df["Series Description"] = final_df["Series Description"].astype(str)
remaining_df["_dcm_series_type"] = remaining_df["_dcm_series_type"].astype(str)
remaining_df["Series Description"] = remaining_df["Series Description"].astype(str)

# Conditions for filtering
condition = (
    (final_df["_dcm_series_type"] != "INVALID")
    & (~final_df["_dcm_series_type"].str.contains("LOW_PROBABILITY", na=False))
    & (~final_df["Series Description"].str.contains("PERFUSION", na=False))
)

# Update 'label' column in the original dataframe
final_df.loc[condition, "label"] = final_df.loc[condition, "_dcm_series_type"]

# Recompute the condition for remaining_df
condition_remaining = (
    (remaining_df["_dcm_series_type"] != "INVALID")
    & (~remaining_df["_dcm_series_type"].str.contains("LOW_PROBABILITY", na=False))
    & (~remaining_df["Series Description"].str.contains("PERFUSION", na=False))
)
# Remove these rows from mirror_df
remaining_df = remaining_df.drop(remaining_df[condition_remaining].index)

# Save the updated dataframes
final_df.to_excel(final_file_path, index=False)
remaining_df.to_excel(remaining_file_path, index=False)
