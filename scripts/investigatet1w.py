import pandas as pd

df = pd.read_excel(
    "/home/mbrzus/programming/dcm_train_data/training/labeled/no_IR/t1w_bad.xlsx"
)

label_file = "/home/mbrzus/programming/dcm_train_data/labeling/iowa_stroke/combined_stroke_data_labeling_final_NO_IR.xlsx"
label_df = pd.read_excel(label_file)

# # get data that contains stroke in the File Name
# filenames = df[df["File Name"].str.contains("Stroke")]["File Name"]
#
# # get data from label where FileName is in filenames
# label_df = label_df[label_df["FileName"].isin(filenames)]
# print(label_df.shape)
# labels = label_df["label"].tolist()
# series_desc = label_df["Series Description"].tolist()
#
# [print(f"{label} - {desc}") for label, desc in zip(labels, series_desc)]


# FLAIR
# get data that contains stroke in the File Name
filenames = df["flair"]
# remove nan and empty
filenames = filenames.dropna()

# get data from label where FileName is in filenames
label_df = label_df[label_df["FileName"].isin(filenames)]
print(label_df.shape)
labels = label_df["label"].tolist()
series_desc = label_df["Series Description"].tolist()

[print(f"{label} - {desc}") for label, desc in zip(labels, series_desc)]
