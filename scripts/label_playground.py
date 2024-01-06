import pandas as pd

# # make a copy of the dataframe containing only fields needed for heuristic based labeling
# df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/combined/combined_data_without_iowa_Dec19.xlsx"
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
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_raw.xlsx"
# )

# # get all unique series descriptions
# df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_raw.xlsx"
# )
# series_descriptions = df["Series Description"].unique()
# new_df = pd.DataFrame(series_descriptions, columns=["Series Description"])
# new_df.to_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/unique_series_descriptions_raw.xlsx"
# )


# investigate series descriptions
def update_labels_and_series_desc(series_desc_list: list, label: str):
    # update labels in label final file
    final_df = pd.read_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_final.xlsx"
    )
    for series_desc in series_desc_list:
        final_df.loc[final_df["Series Description"] == series_desc, "label"] = label
    final_df.to_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_final.xlsx",
        index=False,
    )

    # remove rows from remaining file that have label set in the final file
    remaining_df = pd.read_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_remaining.xlsx"
    )
    for series_desc in series_desc_list:
        remaining_df = remaining_df[remaining_df["Series Description"] != series_desc]
    remaining_df.to_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_remaining.xlsx",
        index=False,
    )

    # remove series descriptions from remaining series descriptions file
    series_desc_df = pd.read_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/unique_series_descriptions_remaining.xlsx"
    )
    for series_desc in series_desc_list:
        series_desc_df = series_desc_df[
            series_desc_df["Series Description"] != series_desc
        ]
    series_desc_df.to_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/unique_series_descriptions_remaining.xlsx",
        index=False,
    )


df = pd.read_excel(
    "/home/mbrzus/programming/dcm_train_data/labeling/unique_series_descriptions_raw.xlsx"
)
series_descriptions_list = df["Series Description"].tolist()

# fmri
# fmri_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "fmri" in series_description:
#         fmri_series_descriptions.append(series_description_raw)
#
# [print(i) for i in fmri_series_descriptions]
# print(fmri_series_descriptions)
# print(len(fmri_series_descriptions))
# update_labels_and_series_desc(fmri_series_descriptions, "fmri")

# # adc
# adc_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "adc" in series_description:
#         adc_series_descriptions.append(series_description_raw)
#
# [print(i) for i in adc_series_descriptions]
# print(adc_series_descriptions)
# print(len(adc_series_descriptions))
# # # check if all adc series descriptions in data file have voluume index 0 - they should
# # df = pd.read_excel(
# #     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_remaining.xlsx"
# # )
# # for series_description in adc_series_descriptions:
# #     df = df[df["Series Description"] == series_description]
# #     df = df[df["_vol_index"] != 0]  # should be empty
# #     # print rows if not empty
# #     if not df.empty:
# #         print(df)
# #         # print number of rows
# #         print(len(df))
#
# ####### SUCCESS - all adc series descriptions have volume index 0 ########
# update_labels_and_series_desc(adc_series_descriptions, "adc")

# # fa
# fa_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "fa" in series_description:
#         fa_series_descriptions.append(series_description_raw)
#
# [print(i) for i in fa_series_descriptions]
# print(fa_series_descriptions)
# print(len(fa_series_descriptions))
# # Upon visual inspection those fields need to be removed
# list_to_remove = [
#     "FAST IR T1 SAG",
#     "FASTBRN,3-P,2D,LOCALIZER",
#     "dWIP SSh_DWI FAST SENSE",
#     "SSh_DWI FAST",
#     "TRACK 3DT1 default",
#     "faReg - DTI_32_directions SENSE",  # not sure about this one
#     "facReg - DTI_32_directions SENSE",  # not sure about this one
# ]
# for item in list_to_remove:
#     fa_series_descriptions.remove(item)
#
# # check if all adc series descriptions in data file have voluume index 0 - they should
# df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_remaining.xlsx"
# )
# for series_description in fa_series_descriptions:
#     df = df[df["Series Description"] == series_description]
#     df = df[df["_vol_index"] != 0]  # should be empty
#     # print rows if not empty
#     if not df.empty:
#         print(df)
#         # print number of rows
#         print(len(df))
#
# ####### SUCCESS - all fa series descriptions have volume index 0 ########
# update_labels_and_series_desc(fa_series_descriptions, "fa")

# # dwig
# # Notes: dwig series label was computed using a heuristic based on the following:
# # - all volumes corresponding to the series has multiple gradient directions
# # - all vollumes corresponding to the series have b-values
# # - more details about implementation are visible in the src code for the package
# final_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_final.xlsx"
# )
# final_df.loc[final_df["_dcm_series_type"] == "dwig", "label"] = "bval_vol"
# final_df.to_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_final.xlsx",
#     index=False,
# )
#
# # remove rows from remaining file that have label set in the final file
# remaining_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_remaining.xlsx"
# )
# remaining_df = remaining_df[remaining_df["_dcm_series_type"] != "dwig"]
# remaining_df.to_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_remaining.xlsx",
#     index=False,
# )

# get all unique series descriptions for dwig _dcm_series_type
final_df = pd.read_excel(
    "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_final.xlsx"
)
dwig_series_descriptions = (
    final_df[final_df["_dcm_series_type"] == "dwig"]["Series Description"]
    .unique()
    .tolist()
)
[print(i) for i in dwig_series_descriptions]
print(dwig_series_descriptions)
print(len(dwig_series_descriptions))

# check which of the dwig series descriptions are still in the remaining file
remaining_df = pd.read_excel(
    "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_remaining.xlsx"
)
list_to_keep = []
for series_description in dwig_series_descriptions:
    df = remaining_df[remaining_df["Series Description"] == series_description]
    # print rows if not empty
    if not df.empty:
        list_to_keep.append(series_description)
print("list_to_keep")
[print(i) for i in list_to_keep]
# remove fields to keep from dwig series descriptions list
for item in list_to_keep:
    dwig_series_descriptions.remove(item)

# remove series descriptions from remaining series descriptions file
series_desc_df = pd.read_excel(
    "/home/mbrzus/programming/dcm_train_data/labeling/unique_series_descriptions_remaining.xlsx"
)
for series_desc in dwig_series_descriptions:
    series_desc_df = series_desc_df[series_desc_df["Series Description"] != series_desc]
series_desc_df.to_excel(
    "/home/mbrzus/programming/dcm_train_data/labeling/unique_series_descriptions_remaining.xlsx",
    index=False,
)
