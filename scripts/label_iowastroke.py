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


df = pd.read_excel(
    "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/unique_series_descriptions_remaining.xlsx"
)
series_descriptions_list = df["Series Description"].tolist()


# # loc
# loc_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "loc" in series_description or "scout" in series_description:
#         loc_series_descriptions.append(series_description_raw)
#
# [print(i) for i in loc_series_descriptions]
# print(loc_series_descriptions)
# print(len(loc_series_descriptions))
# update_labels_and_series_desc(loc_series_descriptions, "loc")

# # flair
# flair_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "flair" in series_description:
#         flair_series_descriptions.append(series_description_raw)
#
# [print(i) for i in flair_series_descriptions]
# print(flair_series_descriptions)
# print(len(flair_series_descriptions))
# update_labels_and_series_desc(flair_series_descriptions, "flair")

# # fa
# fa_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "fa" in series_description or "fractional" in series_description:
#         fa_series_descriptions.append(series_description_raw)
#
# [print(f'"{i}",') for i in fa_series_descriptions]
# print(fa_series_descriptions)
# print(len(fa_series_descriptions))
# list_to_remove = [
#     "COR_T2_FAST",
#     "T2_COR_fast",
#     "FL2D_TRA_HEMO_fast",
#     "T2_AX_fast",
# ]
# for item in list_to_remove:
#     fa_series_descriptions.remove(item)
#
# update_labels_and_series_desc(fa_series_descriptions, "fa")

# # t2w
# t2w_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "t2" in series_description:
#         t2w_series_descriptions.append(series_description_raw)
#
# [print(f'"{i}",') for i in t2w_series_descriptions]
# print(t2w_series_descriptions)
# print(len(t2w_series_descriptions))
# list_to_remove = [
#     "PD_T2_AX",
#     "AX_T2_STAR__HEMO_",
#     "t2_tirm_tra_dark_fluid",
#     "T2_FL2D_AX_HEMO",
#     "t2_fl2d_tra_hemo",
# ]
# for item in list_to_remove:
#     t2w_series_descriptions.remove(item)
#
# update_labels_and_series_desc(t2w_series_descriptions, "t2w")

# t1w
t1w_series_descriptions = []
for series_description_raw in series_descriptions_list:
    series_description = str(series_description_raw).lower()
    if "t1" in series_description or "mprage" in series_description:
        t1w_series_descriptions.append(series_description_raw)

[print(f'"{i}",') for i in t1w_series_descriptions]
print(t1w_series_descriptions)
print(len(t1w_series_descriptions))
# list_to_remove = [
#     "PD_T2_AX",
#     "AX_T2_STAR__HEMO_",
#     "t2_tirm_tra_dark_fluid",
#     "T2_FL2D_AX_HEMO",
#     "t2_fl2d_tra_hemo",
# ]
# for item in list_to_remove:
#     t1w_series_descriptions.remove(item)

update_labels_and_series_desc(t1w_series_descriptions, "t1w")

# # adc
# adc_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "adc" in series_description or "apparent" in series_description:
#         adc_series_descriptions.append(series_description_raw)
#
# [print(i) for i in adc_series_descriptions]
# print(adc_series_descriptions)
# print(len(adc_series_descriptions))
# # check if all adc series descriptions in data file have voluume index 0 - they should
# remain_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_remaining.xlsx"
# )
# for series_description in adc_series_descriptions:
#     sub_df = remain_df[remain_df["Series Description"] == series_description]
#     sub_df = sub_df[sub_df["_vol_index"] != 0]  # should be empty
#     # print rows if not empty
#     if not sub_df.empty:
#         print(sub_df)
#         # print number of rows
#         print(len(sub_df))
#
# ###### SUCCESS - all adc series descriptions have volume index 0 ########
# update_labels_and_series_desc(adc_series_descriptions, "adc")
# # ROUND 2: Find if there are remaining datasets with ADC in ImageType
# remain_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_remaining.xlsx"
# )
# adc_df = remain_df[remain_df["Image Type"].str.contains("ADC")]
# print(adc_df)


# # dwi THIS ALSO NEEDED MANUAL ADJUSTMENT in xlsx
# remaining_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_remaining.xlsx"
# )
# remaining_df = remaining_df[remaining_df["Series Description"].notna()]
# dwi_series_descriptions = []
# bval_df = remaining_df[remaining_df["Diffusionb-valueMax"] > 0]
# bval_df = bval_df[bval_df["Diffusionb-valueMax"] != 7.23e75]
# unique_bvals = bval_df["Diffusionb-valueMax"].unique().tolist()
# print(unique_bvals)
# series_descriptions_list_bval = bval_df["Series Description"].unique().tolist()
# for series_description_raw in series_descriptions_list_bval:
#     print(series_description_raw)
# #
# # list_to_remove = [
# #     "SAG_T1_POST",
# #     "AXIAL_T1_FLAIR",
# #     "COR_T1_POST_OPTIONAL",
# #     "Survey_MST",
# #     "SAG_T1",
# #     "T2_FLAIR_NEW",
# #     "AXIAL_T2_MV",
# #     "AXIAL_T1_FS_POST",
# #     "AXIAL_T2_FFE",
# # ]
# # for item in list_to_remove:
# #     series_descriptions_list_bval.remove(item)
# # get sub df for series descriptions in series_descriptions_list_bval
# check_bval_df = bval_df[
#     bval_df["Series Description"].isin(series_descriptions_list_bval)
# ]
# # check if there are bvalues <= 0
# check_bval_df = check_bval_df[check_bval_df["Diffusionb-valueMax"] <= 0]
# print(check_bval_df)
# update_labels_and_series_desc(series_descriptions_list_bval, "bval_vol")
# # Round 2: bvals = 0
# remaining_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_remaining.xlsx"
# )
# remaining_df = remaining_df[remaining_df["Series Description"].notna()]
# dwi_series_descriptions = []
# bval_df = remaining_df[remaining_df["Diffusionb-valueMax"] == 0]
# bval_df = bval_df[bval_df["Diffusionb-valueMax"] != 7.23e75]
# unique_bvals = bval_df["Diffusionb-valueMax"].unique().tolist()
# print(unique_bvals)
# series_descriptions_list_bval = bval_df["Series Description"].unique().tolist()
# for series_description_raw in series_descriptions_list_bval:
#     print(f'"{series_description_raw}",')
#
# list_to_remove = [
#     "ASSET_calibration_8ch",
#     "3D_TOF_1SLAB_8ch",
#     "COL_3D_TOF_1SLAB_8ch",
#     "PJN_3D_TOF_1SLAB_8ch",
#     "Sag_CUBE_T1_8ch_PRE",
#     "Sag_CUBE_T1_8ch_POST",
#     "Survey",
#     "SAG_T1",
#     "AX_T2",
#     "T2W_FFE",
#     "T2_COR",
#     "SAG_3D_FLAIR",
#     "AX_3D_FLAIR",
#     "AXIAL_3D_T1",
#     "SAG_T1_FSE",
#     "Ax_T2_FLAIR_FS",
#     "Ax_T2_FSE",
#     "AX_T1_FSE",
#     "AX_T2__GRE_EPI",
#     "POST_AX_T1",
#     "POST_COR_T1",
#     "Ax_T2_FLAIR",
#     "AX_T2_PROPELLER",
#     "AX_T2__GRE",
#     "AX_T1",
#     "SAG_T1_FSE_POST",
#     "Ax_T1_PROPELLER_POST",
#     "COR_T1_POST",
#     "AX_T2_FLAIR_PROPELLER",
#     "Ax_T2_PROPELLER",
#     "Ax_T1_PROPELLER",
#     "T2_AX",
#     "T1_SAG",
#     "T2_FLAIR_AX",
#     "T2__GRE_AX",
#     "SWAN_SWI",
#     "SWAN_RECON",
#     "Ax_T2_FLAIR_PROPELLER",
#     "Survey_SHC",
#     "FE_AX",
#     "FLAIR_AX",
#     "T1_FSPGR_3D_AX",
#     "SAG_T1_REFORMATS",
#     "COR_T1_REFORMATS",
#     "AX_T_T1_FS_POST",
#     "AX_FLAIR",
#     "3D_T1_AX_POST",
#     "V3D_T1_COR",
#     "V3D_T1_SAG",
#     "Sag_T1",
#     "Ax_T1",
#     "Ax_T2_FS",
#     "Cor_T2__GRE_EPI",
#     "ax_gre",
#     "T2_AXL",
#     "T2_FFE__AX",
#     "Cor_T2_FSE",
#     "Sag_CUBE_T1",
#     "AX_CUBE_T1",
#     "COR_SWAN",
#     "AX_SWAN_REFORMAT",
#     "Ax_T2_PROPELLER_FS_Post",
#     "Sag_CUBE_T1_FS_POST",
#     "AX_CUBE_POST",
#     "COR_CUBE_POST",
# ]
# for item in list_to_remove:
#     series_descriptions_list_bval.remove(item)
# # get sub df for series descriptions in series_descriptions_list_bval
# check_bval_df = bval_df[
#     bval_df["Series Description"].isin(series_descriptions_list_bval)
# ]
# # check if there are bvalues <= 0
# check_bval_df = check_bval_df[check_bval_df["Diffusionb-valueMax"] <= 0]
# print(check_bval_df)
# update_labels_and_series_desc(series_descriptions_list_bval, "bval_vol")


# # Old model performance on old data is very strong as that what it was trained on. Therefore we can trust most of the predictions.
# # Load the original data
# final_file_path = "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_final.xlsx"
# final_df = pd.read_excel(final_file_path)
#
# # Load the mirror data
# remaining_file_path = "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_remaining.xlsx"
# remaining_df = pd.read_excel(remaining_file_path)
#
# #  Ensure that the series type and description columns are strings
# final_df["_dcm_series_type"] = final_df["_dcm_series_type"].astype(str)
# final_df["Series Description"] = final_df["Series Description"].astype(str)
# remaining_df["_dcm_series_type"] = remaining_df["_dcm_series_type"].astype(str)
# remaining_df["Series Description"] = remaining_df["Series Description"].astype(str)
#
# # Conditions for filtering
# condition = (
#     (final_df["_dcm_series_type"] != "INVALID")
#     & (~final_df["_dcm_series_type"].str.contains("LOW_PROBABILITY", na=False))
#     & (~final_df["Series Description"].str.contains("PERFUSION", na=False))
# )
#
# # Update 'label' column in the original dataframe
# final_df.loc[condition, "label"] = final_df.loc[condition, "_dcm_series_type"]
#
# # Recompute the condition for remaining_df
# condition_remaining = (
#     (remaining_df["_dcm_series_type"] != "INVALID")
#     & (~remaining_df["_dcm_series_type"].str.contains("LOW_PROBABILITY", na=False))
#     & (~remaining_df["Series Description"].str.contains("PERFUSION", na=False))
# )
# # Remove these rows from mirror_df
# remaining_df = remaining_df.drop(remaining_df[condition_remaining].index)
#
# # Save the updated dataframes
# final_df.to_excel(final_file_path, index=False)
# remaining_df.to_excel(remaining_file_path, index=False)
