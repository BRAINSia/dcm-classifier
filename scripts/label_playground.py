import pandas as pd

df = pd.read_excel(
    "/home/mbrzus/programming/dcm_train_data/combined/combined_data_without_iowa_Dec19.xlsx"
)

# list of unique values in the _dcm_series_type column that require further analysis
label_list = [
    "INVALID",
    "LOW_PROBABILITY_t1w",
    "t2starw",
    "LOW_PROBABILITY_t2w",
    "LOW_PROBABILITY_fmri",
    "LOW_PROBABILITY_t2starw",
    "LOW_PROBABILITY_fa",
    "LOW_PROBABILITY_tracew",
    "LOW_PROBABILITY_flair",
    "field_map",
    "nan",
    "LOW_PROBABILITY_field_map",
]

# series_labels = df["_dcm_series_type"].unique()
# [print(i) for i in series_labels]
# print(len(series_labels))


# Filter the DataFrame
sub_df = df[df["_dcm_series_type"].isin(label_list)]

# Save the filtered DataFrame to an Excel file
# sub_df.to_excel("/home/mbrzus/programming/dcm_train_data/combined/filtered_data.xlsx", index=False)
options = {"strings_to_formulas": False, "strings_to_urls": False}
writer = pd.ExcelWriter(
    "/home/mbrzus/programming/dcm_train_data/combined/filtered_data.xlsx",
    engine="xlsxwriter",
    engine_kwargs={"options": options},
)
sub_df.to_excel(writer, sheet_name="results", index=False)
writer._save()


# Get the counts of each unique value in the _dcm_series_type column
row_counts = sub_df["_dcm_series_type"].value_counts()
# Output the counts
print(row_counts)

# all unique values in the _dcm_series_type column - 19
# INVALID
# t2w
# t1w
# LOW_PROBABILITY_t1w
# t2starw
# LOW_PROBABILITY_t2w
# LOW_PROBABILITY_fmri
# dwig
# LOW_PROBABILITY_t2starw
# fa
# adc
# LOW_PROBABILITY_fa
# tracew
# LOW_PROBABILITY_tracew
# LOW_PROBABILITY_flair
# flair
# field_map
# nan
# LOW_PROBABILITY_field_map

# unsure types and counts
# LOW_PROBABILITY_fmri         5149
# INVALID                      4395
# LOW_PROBABILITY_t1w          1088
# LOW_PROBABILITY_t2w          1038
# t2starw                       995
# LOW_PROBABILITY_tracew        206
# LOW_PROBABILITY_fa            185
# LOW_PROBABILITY_t2starw       171
# field_map                      77
# LOW_PROBABILITY_flair          27
# LOW_PROBABILITY_field_map       1
