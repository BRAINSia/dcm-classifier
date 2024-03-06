import pandas as pd


label_file = "/home/mbrzus/programming/dcm_train_data/V2-20240229/no_duplicates/labeling/combined_all_data_labeling_Feb29_final.xlsx"
df = pd.read_excel(label_file, index_col=0)
trainings_file = "/home/mbrzus/programming/dcm_train_data/V2-20240229/training/combined_all_training_data_Mar5.xlsx"
trainings_df = pd.read_excel(trainings_file, index_col=0)

label_df = df[
    ["FileName", "Diffusionb-valueBool", "HasDiffusionGradientOrientation", "label"]
]
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
