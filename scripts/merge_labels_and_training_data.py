import pandas as pd


# label_file = (
#     "/home/mbrzus/programming/dcm_train_data/training/labeled/combined_all_Jan29.xlsx"
# )
# df = pd.read_excel(label_file)
# # replace label bval_img with bval_vol
# df.loc[df["label"] == "dwig", "label"] = "bval_vol"
# df.loc[df["label"] == "t2starw", "label"] = "t2star"
# df.to_excel(label_file, index=False)

label_file = "/home/mbrzus/programming/dcm_train_data/labeling/minipig/combined_minipig_labeling_final.xlsx"
df = pd.read_excel(label_file)
trainings_file = (
    "/home/mbrzus/programming/dcm_train_data/training/minipig/training_minipig.xlsx"
)
trainings_df = pd.read_excel(trainings_file)

label_df = df[["FileName", "label"]]
print(label_df.shape)
print(label_df.head())

# merge on "FileName"
merged_df = pd.merge(
    left=trainings_df,
    right=label_df,
    how="left",
    left_on="FileName",
    right_on="FileName",
)
final_df_name = trainings_file.replace(".xlsx", "_labeled.xlsx")
merged_df.to_excel(final_df_name, index=False)


# cols = trainings_df.columns.tolist()
# [print(cols.count(col)) for col in cols]
