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
        "/home/mbrzus/programming/dcm_train_data/labeling/predicthd/combined_data_without_iowa_labeling_final.xlsx"
    )
    for series_desc in series_desc_list:
        final_df.loc[final_df["Series Description"] == series_desc, "label"] = label
    final_df.to_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/predicthd/combined_data_without_iowa_labeling_final.xlsx",
        index=False,
    )

    # remove rows from remaining file that have label set in the final file
    remaining_df = pd.read_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/predicthd/combined_data_without_iowa_labeling_remaining.xlsx"
    )
    for series_desc in series_desc_list:
        remaining_df = remaining_df[remaining_df["Series Description"] != series_desc]
    remaining_df.to_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/predicthd/combined_data_without_iowa_labeling_remaining.xlsx",
        index=False,
    )

    # remove series descriptions from remaining series descriptions file
    series_desc_df = pd.read_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/predicthd/unique_series_descriptions_remaining.xlsx"
    )
    for series_desc in series_desc_list:
        series_desc_df = series_desc_df[
            series_desc_df["Series Description"] != series_desc
        ]
    series_desc_df.to_excel(
        "/home/mbrzus/programming/dcm_train_data/labeling/predicthd/unique_series_descriptions_remaining.xlsx",
        index=False,
    )


df = pd.read_excel(
    "/home/mbrzus/programming/dcm_train_data/labeling/predicthd/unique_series_descriptions_remaining.xlsx"
)
series_descriptions_list = df["Series Description"].tolist()


# # localizer
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


# # fmri
# fmri_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "fmri" in series_description or "func" in series_description:
#         fmri_series_descriptions.append(series_description_raw)
#
# [print(i) for i in fmri_series_descriptions]
# print(fmri_series_descriptions)
# print(len(fmri_series_descriptions))
# update_labels_and_series_desc(fmri_series_descriptions, "fmri")

# adc
# adc_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "adc" in series_description:
#         adc_series_descriptions.append(series_description_raw)
#
# [print(i) for i in adc_series_descriptions]
# print(adc_series_descriptions)
# print(len(adc_series_descriptions))
# # check if all adc series descriptions in data file have voluume index 0 - they should
# df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_remaining.xlsx"
# )
# for series_description in adc_series_descriptions:
#     df = df[df["Series Description"] == series_description]
#     df = df[df["_vol_index"] != 0]  # should be empty
#     # print rows if not empty
#     if not df.empty:
#         print(df)
#         # print number of rows
#         print(len(df))

####### SUCCESS - all adc series descriptions have volume index 0 ########
# update_labels_and_series_desc(adc_series_descriptions, "adc")
# # ROUND 2: I found additional ADC images where seires description uses full name instead of abbreviation
# adc_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "apparent" in series_description:
#         adc_series_descriptions.append(series_description_raw)
# [print(i) for i in adc_series_descriptions]
# print(adc_series_descriptions)
# print(len(adc_series_descriptions))
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
# # ROUND 2: I found additional FA images where seires description uses full name instead of abbreviation
# fa_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "fractional" in series_description:
#         fa_series_descriptions.append(series_description_raw)
#
# [print(i) for i in fa_series_descriptions]
# print(fa_series_descriptions)
# print(len(fa_series_descriptions))
# update_labels_and_series_desc(fa_series_descriptions, "fa")
# # ROUND 3: Seems that faReg and facRef DTI images are also FA.
# # some have bvalue, some have 'FA' in the image type
# fa_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "fa" in series_description and "reg - dti" in series_description:
#         fa_series_descriptions.append(series_description_raw)
#
# [print(i) for i in fa_series_descriptions]
# print(fa_series_descriptions)
# print(len(fa_series_descriptions))
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
#
# # get all unique series descriptions for dwig _dcm_series_type
# final_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_final.xlsx"
# )
# dwig_series_descriptions = (
#     final_df[final_df["_dcm_series_type"] == "dwig"]["Series Description"]
#     .unique()
#     .tolist()
# )
# [print(i) for i in dwig_series_descriptions]
# print(dwig_series_descriptions)
# print(len(dwig_series_descriptions))
#
# # check which of the dwig series descriptions are still in the remaining file
# remaining_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_remaining.xlsx"
# )
# list_to_keep = []
# for series_description in dwig_series_descriptions:
#     df = remaining_df[remaining_df["Series Description"] == series_description]
#     # print rows if not empty
#     if not df.empty:
#         list_to_keep.append(series_description)
# print("list_to_keep")
# [print(i) for i in list_to_keep]
# # remove fields to keep from dwig series descriptions list
# for item in list_to_keep:
#     dwig_series_descriptions.remove(item)
#
# # remove series descriptions from remaining series descriptions file
# series_desc_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/unique_series_descriptions_remaining.xlsx"
# )
# for series_desc in dwig_series_descriptions:
#     series_desc_df = series_desc_df[series_desc_df["Series Description"] != series_desc]
# series_desc_df.to_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/unique_series_descriptions_remaining.xlsx",
#     index=False,
# )


# # t1w
# t1w_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "t1" in series_description or "mprage" in series_description:
#         t1w_series_descriptions.append(series_description_raw)
#
# [print(i) for i in t1w_series_descriptions]
# print(t1w_series_descriptions)
# print(len(t1w_series_descriptions))
# # Upon visual inspection those fields need to be removed
# # todo: NOTE: there are IR (inverse recovery) images that I am not sure if should be classified as t1w or flair, for now I keep them as t1w
# list_to_remove = [
#     "T1RHO MAP - 10ms",
#     "T1RHO MAP - 60ms",  # seem to be some specific cardiovascular scans https://jcmr-online.biomedcentral.com/articles/10.1186/s12968-023-00940-1
#     "Ax T1 FLAIR PROPELLER",
# ]
# for item in list_to_remove:
#     t1w_series_descriptions.remove(item)
#
# update_labels_and_series_desc(t1w_series_descriptions, "t1w")


# # PDT2
# """
# Problems:
# - not all PDT2 serieses have mutliple volumes. For data with only one volume I will label it as pd or t2w depending on
# previous model classification.
#
# For serieses with multiple volumes:
# - in most cases (probably all) previous model classified one of them as t2w and other as lowe probability. Therefore,
# I will label t2w volumes as t2w and other ones as pd.
# """
# pdt2_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "pd" in series_description and "t2" in series_description:
#         pdt2_series_descriptions.append(series_description_raw)
# [print(i) for i in pdt2_series_descriptions]
# print(pdt2_series_descriptions)
# print(len(pdt2_series_descriptions))
#
# final_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_final.xlsx"
# )
# for series_desc in pdt2_series_descriptions:
#     final_df.loc[
#         final_df["Series Description"] == series_desc, "label"
#     ] = final_df.apply(
#         lambda row: "t2w" if row["_dcm_volume_type"] == "t2w" else "pd", axis=1
#     )
#
# final_df.to_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_final.xlsx",
#     index=False,
# )
#
# # remove rows from remaining file that have label set in the final file
# remaining_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_remaining.xlsx"
# )
# for series_desc in pdt2_series_descriptions:
#     remaining_df = remaining_df[remaining_df["Series Description"] != series_desc]
# remaining_df.to_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_remaining.xlsx",
#     index=False,
# )
#
# # remove series descriptions from remaining series descriptions file
# series_desc_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/unique_series_descriptions_remaining.xlsx"
# )
# for series_desc in pdt2_series_descriptions:
#     series_desc_df = series_desc_df[series_desc_df["Series Description"] != series_desc]
# series_desc_df.to_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/unique_series_descriptions_remaining.xlsx",
#     index=False,
# )

# # pd
# pd_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "pd" in series_description:
#         pd_series_descriptions.append(series_description_raw)
#
# [print(i) for i in pd_series_descriptions]
# print(pd_series_descriptions)
# print(len(pd_series_descriptions))
# update_labels_and_series_desc(pd_series_descriptions, "pd")


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


# # t2w star
# t2w_star_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "star" in series_description or "*" in series_description:
#         t2w_star_series_descriptions.append(series_description_raw)
#
# [print(i) for i in t2w_star_series_descriptions]
# print(t2w_star_series_descriptions)
# print(len(t2w_star_series_descriptions))
# # Upon visual inspection those fields need to be removed
# # todo: NOTE: there are images named T2STAR MAP. I am not sure if they should be a different class, for now I include them with T2*
# list_to_remove = ["179PredictHD*0783200810203DSPG"]
# for item in list_to_remove:
#     t2w_star_series_descriptions.remove(item)
#
# update_labels_and_series_desc(t2w_star_series_descriptions, "t2star")


# # t2w
# t2w_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "t2" in series_description:
#         t2w_series_descriptions.append(series_description_raw)
#
# [print(i) for i in t2w_series_descriptions]
# print(t2w_series_descriptions)
# print(len(t2w_series_descriptions))
# # # Upon visual inspection those fields need to be removed
# # # todo: NOTE: similar to t1 there are images with IR (inverse recovery) that I am not sure if should be classified as t2w or flair, for now I keep them as t2w
# list_to_remove = [
#     "SCOUT2",
#     "fix ep2d_dti6_192_pat2f_12 directions",
#     "fix ep2d_dti6_192_pat2f_12 directions_TRACEW",
#     "T2 MAP",  # not sure about this one
# ]
# for item in list_to_remove:
#     t2w_series_descriptions.remove(item)
#
# update_labels_and_series_desc(t2w_series_descriptions, "t2w")


# # field map
# fieldmap_series_descriptions = []
# for series_description_raw in series_descriptions_list:
#     series_description = str(series_description_raw).lower()
#     if "field" in series_description:
#         fieldmap_series_descriptions.append(series_description_raw)
#
# [print(i) for i in fieldmap_series_descriptions]
# print(fieldmap_series_descriptions)
# print(len(fieldmap_series_descriptions))
# update_labels_and_series_desc(fieldmap_series_descriptions, "field_map")


# # tracew
# """
# The tracew serieses also comprise of bvalue images.
# We investigate all images with non zero bvalue that were not classified as part of the DWI labeling above.
# """
# remaining_df = pd.read_excel(
#     "/home/mbrzus/programming/dcm_train_data/labeling/combined_data_without_iowa_labeling_remaining.xlsx"
# )
# bval_series_desc_raw = (
#     remaining_df[remaining_df["Diffusionb-valueMax"] >= 0]["Series Description"]
#     .unique()
#     .tolist()
# )
# bval_series_desc = []
# for series_desc in bval_series_desc_raw:
#     for i in ["dti", "dwi", "diff", "b0", "b500", "b600", "b1000", "b1200", "b1500"]:
#         if i in str(series_desc).lower():
#             bval_series_desc.append(series_desc)
#             break
#
# bval_series_desc_failed = [i for i in bval_series_desc_raw if i not in bval_series_desc]
#
# [print(i) for i in bval_series_desc]
# print("\n\n=====================================\n\n")
# [print(i) for i in bval_series_desc_failed]
# print(len(bval_series_desc))
# print(len(bval_series_desc_failed))
# update_labels_and_series_desc(bval_series_desc, "bval_vol")

# bval_series_desc_dti = [i for i in bval_series_desc if "dti" in str(i).lower()]
# [print(i) for i in bval_series_desc_no_dti]
# print(bval_series_desc)
# [print(i) for i in bval_series_desc_dti]
# print(len(bval_series_desc))
