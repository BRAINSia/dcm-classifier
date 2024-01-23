import pandas as pd

df = pd.read_excel(
    "/home/mbrzus/programming/dcm_train_data/training/labeled/combined_all_Jan22.xlsx",
)
print(df.shape)

# TODO some labels are incorrectly set, make sure to adjust them
# TODO there is t2star and t2starw that are the same, make sure to adjust them
# drop rows with no label
df = df.dropna(subset=["label"])

# print how many of each label
print(df["label"].value_counts())

# dataframe with the same number of each label, randomly selected
# the number is equal to the number of the smallest label
out_df = (
    df.groupby("label")
    .apply(lambda x: x.sample(df["label"].value_counts().min(), random_state=42))
    .reset_index(drop=True)
)
print(out_df.shape)
out_df.to_excel(
    "/home/mbrzus/programming/dcm_train_data/training/labeled/combined_all_Jan22_small_balanced.xlsx",
    index=False,
)
