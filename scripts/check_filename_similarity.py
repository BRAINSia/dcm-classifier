import pandas as pd



def check_similarity(frame1_path, frame2_path):
    cols = ["FileName", "Repetition Time", "Imaging Frequency", "Pixel Bandwidth", "Flip Angle", "Echo Time"]

    frame1 = pd.read_excel(frame1_path, index_col=False)
    frame2 = pd.read_excel(frame2_path, index_col=False)

    common_rows_df1 = frame1[frame1['FileName'].isin(frame2['FileName'])]
    common_rows_df2 = frame2[frame2['FileName'].isin(frame1['FileName'])]


    subset_frame1 = common_rows_df1[cols]
    subset_frame2 = common_rows_df2[cols]
    subset_frame1 = subset_frame1.head(5000)
    subset_frame2 = subset_frame2.head(5000)
    subset_frame1.set_index('FileName',)
    subset_frame2.set_index('FileName',)


    same = subset_frame1 == subset_frame2
    print(same)
    print(subset_frame2.shape)
    print(subset_frame1.shape)
    print(subset_frame2.columns)
    print(subset_frame2.columns)
    # are_equal = subset_frame1.equals(subset_frame2)
    # print(are_equal)
    differences = subset_frame1 != subset_frame2

    print(differences)


if __name__ == "__main__":
    base_dir = "/tmp/dicom_data/V2-20240229"
    frame1_path = f"{base_dir}/raw_data/combined_iowa_stroke_data_Jan9.xlsx"
    frame2_path = f"/tmp/dicom_data/no_duplicates/combined_iowa_stroke_data_Feb26.xlsx"
    check_similarity(frame1_path, frame2_path)
