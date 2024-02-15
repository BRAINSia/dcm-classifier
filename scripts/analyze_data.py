import pandas as pd

features_to_remove = [
    "Image Type_ASL",
    "Image Type_MAP",
    "Image Type_PCA",
    "Image Type_RELCBF",
    "Image Type_VELOCITY",
    "Image Type_TTP",
    "Image Type_PBP",
    "Image Type_COLLAPSE",
    "Image Type_RD",
    "Image Type_FILTERED",
    "Image Type_MSUM",
    "Image Type_SAVE",
    "Image Type_SCREEN",
    "Image Type_MEAN",
    "Image Type_MOCO",
    "Image Type_TTEST",
    "Image Type_MOTIONCORRECTION",
    "Image Type_DIS3D",
    "Image Type_OUT",
    "Image Type_BVALUE",
    "Image Type_VASCULAR",
    "Image Type_AVERAGE",
    "Image Type_GSP",
    "Image Type_CALC",
    "Image Type_IMAGE",
    "MR Acquisition Type_UNKNOWN",
    "Image Type_FAT",
    "Image Type_DRG",
    "Image Type_CDWI",
    "Image Type_COMPRESSED",
    "Image Type_FM4",
    "In-plane Phase Encoding Direction_OTHER",
    "Image Type_COMPOSED",
    "Image Type_DIF",
    "Image Type_COMP",
    "Image Type_REG",
    "Image Type_B0",
    "Image Type_TENSOR",
    "Image Type_4",
    "Image Type_THICK",
    "Image Type_COR",
    "Image Type_PERFUSION",
    "Image Type_PROC",
    # "Image Type_SECODNARY",
    "Image Type_MIXED",
    "Image Type_MIN",
    "Image Type_CSAPARALLEL",
    "Image Type_MIP",
    "Image Type_RRISDOC",
    "Image Type_COMBINED",
    "Image Type_SUBTRACT",
    "In-plane Phase Encoding Direction_COLUMN",
    "Image Type_SH4",
    "Image Type_PROJECTION",
    "Image Type_WATER",
    "Image Type_DIXON",
    "Image Type_SAG",
    "Image Type_1",
    "Image Type_6",
    "Image Type_SH",
    "Image Type_LOSSY",
    "Image Type_JP2K",
    "Image Type_BOUND",
    "Manufacturer_other",
    "Image Type_UNKNOWN",
    "Image Type_DECOMPRESSED",
    "Image Type_FS",
    "Image Type_BS",
    "Image Type_SH2",
    # "Image Type_Secondary",
    # "Image Type_Derived",
    "Image Type_AXIAL",
    # "Image Type_DWI",
    "Image Type_W",
    "Image Type_PHASE",
    "Image Type_FM2",
    "Image Type_FM1",
    "Image Type_0040",
    "Image Type_ENDORECTAL",
    "Image Type_DYNACAD2",
    "Image Type_RX",
    "Image Type_11",
    "Image Type_DRB",
    "Image Type_DRS",
    "Image Type_RESAMPLED",
    "Image Type_20159358",
    "Image Type_12",
    "Image Type_9",
    "Image Type_13",
    "Image Type_SH5",
    "Sequence Variant_TOF",
    "Sequence Variant_MTC",
    "Image Type_ISODWI",
    # 88 features
    "Image Type_REFORMATTED",
    "Image Type_EXP",
    "Image Type_FM",
    "Image Type_IP",
    "Image Type_3",
    "Image Type_SWI",
    "Image Type_MNIP",
    "Image Type_Derived",
    "Image Type_Secondary",
    "Image Type_ENHANCED",
    "Image Type_DWI",
    "Image Type_SUB",
    "Image Type_SECODNARY",
    # 74 features
    "Image Type_POSDISP",
    "Image Type_IN",
    "Image Type_MPR",
    "Image Type_T1",
    "Image Type_TRA",
    "Image Type_DFC",
    "Image Type_T2",
    "Image Type_FIL",
    "Image Type_MFSPLIT",
    "Image Type_FM3",
    "Image Type_GDC",
    "Image Type_MIX",
    "Image Type_CSA",
    # 61 features
    "Image Type_UNSPECIFIE",
    "Samples per Pixel",
    "Image Type_TRACEW",
    "Image Type_PRIMARY",
    "Image Type_IR",
    "SeriesVolumeCount",
    "Image Type_SECONDARY",
    "Image Type_PROCESSED",
    "Image Type_PROPELLER",
    # "Manufacturer_toshiba",
    # 51 features without toshiba
    # "Image Type_P",
    # "Scanning Sequence_RM",
    # "Echo Number(s)",
    # "MR Acquisition Type_3D",
    # "Image Type_DIFFUSION",
    # "Variable Flip Angle Flag_Y",
    # "Image Type_SE",
    # "Sequence Variant_NONE",
    # "Variable Flip Angle Flag_N",
    # "Image Type_NORM",
    # "Manufacturer_ge",
    # "Manufacturer_siemens",
    # "Sequence Variant_SS",
    # "Sequence Variant_OSP",
    # "Image Type_DIS2D",
    # "Image Type_OTHER",
    # "Manufacturer_philips",
    # "Image Type_M",
    # "Image Type_NONE",
    # "Image Type_FFE",
    # "Image Type_2",
    # 34 features without toshiba, ge, siemens, philips
]
imagetype_to_integer_mapping = {
    "bval_vol": 0,
    "t1w": 1,
    "t2star": 2,
    "t2w": 3,
    "field_map": 4,
    "pd": 5,
    "flair": 6,
    "adc": 7,
    "fa": 8,
    "fmri": 9,
    "stir": 10,
    "eadc": 11,
}

# label_file = "/home/mbrzus/programming/dcm_train_data/training/labeled/no_IR/combined_predicthd_Jan29_labeled_NO_IR.xlsx"
# label_df = pd.read_excel(label_file)
# # Consolidate DataFrame operations
# df = label_df.loc[:, ~label_df.columns.str.contains("^Unnamed")]
# df.drop(
#     columns=["FileName"] + [i for i in features_to_remove if i in df.columns],
#     inplace=True,
# )
# print(df.columns)
# # Keep only rows where 'label' matches keys in imagetype_to_integer_mapping
# df = df[df["label"].isin(imagetype_to_integer_mapping.keys())]
# # print number of unique labels and their count
# print(df.shape)
# # get only unique rows
# df = df.drop_duplicates()
# print(df.shape)
# print(df["label"].value_counts())


label_file = "/home/mbrzus/programming/dcm_train_data/training/labeled/no_IR/combined_all_Jan30_NO_IR_LOC.xlsx"
label_df = pd.read_excel(label_file)

# Pre-processing steps
df = label_df.loc[:, ~label_df.columns.str.contains("^Unnamed")]
df.drop(
    columns=features_to_remove, inplace=True
)  # [i for i in features_to_remove if i in df.columns], inplace=True)

# Filter based on imagetype_to_integer_mapping
df = df[df["label"].isin(imagetype_to_integer_mapping.keys())]

# Preserve FileName column in a separate DataFrame
file_names = df[["File Name"]].reset_index(drop=True)

# Remove duplicates including FileName
df = df.drop_duplicates()

# Re-attach FileName using the index
df = df.reset_index(drop=True)
df["File Name"] = file_names.loc[df.index, "File Name"]

# Organize into bundles
bundle_h = df[df["File Name"].str.startswith("/h")]
bundle_l = df[df["File Name"].str.startswith("/l")]
bundle_other = df[
    ~df["File Name"].str.startswith("/h") & ~df["File Name"].str.startswith("/l")
]

# Print the count of each label for each bundle
print("Bundle /h:")
print(bundle_h["label"].value_counts())
print("\nBundle /l:")
print(bundle_l["label"].value_counts())
print("\nOther Bundle:")
print(bundle_other["label"].value_counts())
