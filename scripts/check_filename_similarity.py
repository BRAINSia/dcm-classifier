import pandas as pd



def check_similarity(frame1_path, frame2_path):
    cols = ["FileName", "Repetition Time", "Imaging Frequency", "Pixel Bandwidth", "Flip Angle", "Echo Time"]

    frame1 = pd.read_excel(frame1_path)
    frame2 = pd.read_excel(frame2_path)
    print(frame1.shape)
    print(frame2.shape)

    # for comparing iowa raw and no duplicates
    frame1 = frame1[frame1["FileName"].isin(frame2["FileName"])]
    frame1 = frame1.dropna(subset=["FileName"])
    frame2 = frame2.dropna(subset=["FileName"])

    frame1.sort_values(by=['FileName'], inplace=True)
    frame2.sort_values(by=['FileName'], inplace=True)

    subset_frame1 = frame1[cols]
    subset_frame2 = frame2[cols]
    # subset_frame1 = subset_frame1.head(5000)
    # subset_frame2 = subset_frame2.head(5000)
    for training_row, do_duplicates in zip(subset_frame1.iterrows(), subset_frame2.iterrows()):
        print(f"Training: {training_row[1]['FileName']} | Duplicates: {do_duplicates[1]['FileName']}")

        for header in cols:
            if training_row[1][header] != do_duplicates[1][header]:
                print(f"Header: {header}")

                # print(f"Training: {training_row[1]['FileName']} | Duplicates: {do_duplicates[1]['FileName']}")

                print(f"Training: {training_row[1][header]} | Duplicates: {do_duplicates[1][header]}\n")

    # same = subset_frame1 == subset_frame2
    # print(same)
    # print(subset_frame2.shape)
    # print(subset_frame1.shape)
    # print(subset_frame2.columns)
    # print(subset_frame2.columns)
    # # are_equal = subset_frame1.equals(subset_frame2)
    # # print(are_equal)
    # differences = subset_frame1 != subset_frame2

    # print(differences)


if __name__ == "__main__":
    base_dir = "/tmp/dicom_data/V2-20240229"
    all_no_duplicates = "/tmp/dicom_data/V2-20240229/no_duplicates/combined_all_data_Feb29.xlsx"
    new_all_training_no_duplicates = "/tmp/dicom_data/V2-20240229/training/combined_all_training_data_Mar5.xlsx"
    # iowa_training = "/tmp/dicom_data/V2-20240229/training/combined_training_iowa_stroke_data_Feb26.xlsx"
    # iowa_new_training = "/home/cavriley/files/dcm_files/no_duplicates_iowa_data_file.xlsx"
    # iowa_raw = "/tmp/dicom_data/V2-20240229/raw_data/combined_iowa_stroke_data_Jan9.xlsx"
    # iowa_no_duplicates = "/tmp/dicom_data/V2-20240229/no_duplicates/combined_iowa_stroke_data_Feb26.xlsx"
    check_similarity(all_no_duplicates, new_all_training_no_duplicates)

