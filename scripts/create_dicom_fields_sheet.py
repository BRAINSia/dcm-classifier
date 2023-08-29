import pandas as pd
from glob import glob
from dcm_classify.study_processing import ProcessOneDicomStudyToVolumesMappingBase


def generate_dicom_dataframe(session_dirs: list, output_file: str):
    dfs = []
    for ses_dir in session_dirs:
        study = ProcessOneDicomStudyToVolumesMappingBase(study_directory=ses_dir)
        for series_number, series in study.series_dictionary.items():
            d = series.get_series_info_dict()
            series_df = pd.DataFrame.from_dict(data=d, orient="index").T
            dfs.append(series_df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_excel(output_file, index=False)


if __name__ == "__main__":
    # path to prostate data DICOM dir: /localscratch/Users/mbrzus/Botimageai/homerun/DATA/150_Test_Data

    # Note: Prostate data not in strict BIDS format therefore require small change in do_training_extraction function
    # to find the data
    # dicom_path = "/localscratch/Users/mbrzus/Botimageai/homerun/DATA/150_Test_Data"
    # out = "../../data/prostate_all_dicom_raw.xlsx"
    # out_p = "../../data/prostate_df.pkl"
    # generate_dicom_dataframe(dicom_dir=dicom_path, output_file=out, out_pkl=out_p)

    # dicom_path = "/localscratch/Users/mbrzus/Stroke_Data/IOWA_STROKE_RETRO_DICOM"
    # out = "../../data/iowaStroke_all_dicom_raw.xlsx"
    # out_p = "../../data/iowaStroke_df.pkl"
    # generate_dicom_dataframe(dicom_dir=dicom_path, output_file=out, out_pkl=out_p)

    # dicom_path = "/localscratch/Users/mbrzus/TrackOn/HDNI_003"
    # out = "../../data/track32_all_dicom_raw.xlsx"
    # out_p = "../../data/track32_df.pkl"
    # generate_dicom_dataframe(dicom_dir=dicom_path, output_file=out, out_pkl=out_p)
    #
    # dicom_path = Path(
    #     "/Shared/sinapse/MiniPigScratch/CLN2_data/noah_batten_porcine_model/dicom_SIEREN_CLN2"
    # )

    # test new DICOM files from Iowa Retro Stroke for the missing dicom fiels
    dicom_path = "/Shared/boeslab/Data/Lesion/SickKids/raw_sourcedata/VIPS1c/VIPSI_singleinfarcts_20221215"
    out = "../data/testfile.xlsx"
    ses_dirs = sorted(list(glob(f"{dicom_path}/VIPS1*")))[:1]
    generate_dicom_dataframe(session_dirs=ses_dirs, output_file=out)
