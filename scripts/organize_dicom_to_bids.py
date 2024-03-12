#!/usr/bin/env python3

from create_dicom_fields_sheet import generate_dicom_dataframe
from glob import glob
import random
import os
import pandas as pd
from pathlib import Path


def organize_data_to_bids(df_path: str, output_dir: str):
    # read dataframe and construct new dicom dir
    df = pd.read_excel(df_path)
    subs = []
    all_sess = []
    for i, row in df.iterrows():
        series_num = row["SeriesNumber"]
        print(Path(row["FileName"]))
        sub = Path(row["FileName"]).parent.parent.parent.name.split("_")[0]
        # pad subject with 0s at the begining to make it 5 digits and create subject dir
        # sub = sub.zfill(5)
        if sub not in subs:
            subs.append(sub)
            os.system(f"mkdir -p {output_dir}/sub-{sub}")

        # get all unique sessions (dates) corresponding to SeriesInstanceUID
        sess = list(
            set(
                list(
                    df[df["StudyInstanceUID"] == row["StudyInstanceUID"]][
                        "InstanceCreationDate"
                    ]
                    .dropna()
                    .values
                )
            )
        )
        # if no session number, generate a random one
        if len(sess) == 0:
            sess = [random.randint(30000000, 99999999)]
        # for each session, create a session dir
        for i in sess:
            if i not in all_sess:
                all_sess.append(i)
                os.system(f"mkdir -p {output_dir}/sub-{sub}/ses-{int(i)}")
                print(f"mkdir -p {output_dir}/sub-{sub}/ses-{int(i)}")

        # try to get a current session
        try:
            ses = int(row["InstanceCreationDate"])
        # if there is no sessions, there should be only one randomly created one
        except Exception as e:
            if len(sess) == 1:
                ses = int(sess[0])
            else:
                print(f"Impossible to determine session number.: {e}")
                continue

        # this is the final session dir
        ses_dir = f"{output_dir}/sub-{sub}/ses-{ses}"
        # copy the dicom files to the session dir
        os.system(f"cp -r {Path(row['FileName']).parent} {ses_dir}/{series_num}")


if __name__ == "__main__":
    # generate big dicom dataframe
    dicom_path = "/localscratch/Users/mbrzus/Stroke_Data/IOWA_STROKE_RETRO_DICOM_NEW"
    out = "../data/new_IowaStrokeRetro.xlsx"
    ses_dirs = sorted(list(glob(f"{dicom_path}/XY*")))
    generate_dicom_dataframe(session_dirs=ses_dirs, output_file=out)

    # make new dir for organized dicom data
    bids_dicom_path = (
        "/localscratch/Users/mbrzus/Stroke_Data/IOWA_STROKE_RETRO_DICOM_NEW_BIDS"
    )
    if not os.path.exists(bids_dicom_path):
        os.system(f"mkdir -p {bids_dicom_path}")

    # organize data to BIDS
    organize_data_to_bids(df_path=out, output_dir=bids_dicom_path)

    # TODO: This script failed on the Iowa Stroke Retro dataset because every subject has the same InstanceCreationDate
    # which was set to the date of the XNAT upload
