from pathlib import Path

import pandas as pd


def get_models(input_path: str) -> list:
    """
    This function takes in a dataframe and returns the models used in the dataframe
    """
    input_frame = pd.read_excel(input_path)
    try:
        models = input_frame["Manufacturer's Model Name"].unique()
        print(models)
    except KeyError:
        print("model name not found in ", input_path)
        return []
    return models


def get_sites(input_frame: pd.DataFrame) -> list:
    """
    This function takes in a dataframe and returns the sites used in the dataframe
    """
    sites = input_frame["site"].unique()
    return sites


def get_num_rows(input_path: str) -> int:
    """
    This function takes in a dataframe and returns the number of rows in the dataframe
    """
    input_frame = pd.read_excel(input_path)
    return input_frame.shape[0]


if __name__ == "__main__":
    input_dir = "/tmp/dcm_classifier_training_data/dcm_train_data/combined"

    # pred_hd_path = f"{input_dir}/combined_predicthd_data_Jan9.xlsx"
    # pred_hd = pd.read_excel(f"{input_dir}/combined_predicthd_data_Jan9.xlsx")
    # print("Predict HD:")
    # pred_hd_models = get_models(pred_hd_path)
    # print(f"Models: {pred_hd_models}")
    # print(f"Sites: {get_sites(input_frame)}")

    iowa_path = f"{input_dir}/combined_iowa_stroke_data_Jan9.xlsx"
    iowa_rows = get_num_rows(iowa_path)
    # # iowa = pd.read_excel(f"{input_dir}/combined_iowa_stroke_data_Jan9.xlsx")
    # print("Iowa Stroke:")
    # iowa_models = get_models(iowa_path)

    predict_rows = 0
    for file in Path(
        "/tmp/dcm_classifier_training_data/dcm_train_data/raw_site_data"
    ).glob("*.xlsx"):
        print(file.as_posix())
        predict_rows += get_num_rows(file.as_posix())

    nebraska_rows = 0
    for file in Path("/tmp/nebraska_dicom_data").glob("*.xlsx"):
        print(file.as_posix())
        nebraska_rows += get_num_rows(file.as_posix())
        # print(nebraska_models)
    # print("Predict HD Models: ", set(PredictHD_Models), len(set(PredictHD_Models)))
    # print("Nebraska Models: ", set(nebraska_models), len(set(nebraska_models)))
    # print("Overall Sites: ", set(get_sites(pred_hd) + get_sites(iowa)))
    # print("Overall Models: ", set(iowa_models), len(set(iowa_models)))
    Iowa_Models = {
        "Ingenia",
        "MRT200SP3",
        "Avanto",
        "Aera",
        "Symphony",
        "TrioTim",
        "Gyroscan NT",
        "Espree",
        "Verio",
        "Achieva",
        "DISCOVERY MR950",
        "Skyra",
        "Avanto_DOT",
        "MAGNETOM Sola",
        "MAGNETOM VISION Plus",
        "Optima MR450w",
        "Skyra_fit",
        "SIGNA Voyager",
        "Avanto_fit",
        "Orian",
        "MRT200PP3",
        "ECHELON",
    }

    Nebraska_Models = {
        "SIGNA Hero",
        "Multiva",
        "Ingenia Elition S",
        "SIGNA Pioneer",
        "TrioTim",
        "MAGNETOM Skyra",
        "Aera",
        "https://github.com/fedorov/dcmqi.git",
        "Symphony",
        "Optima MR360",
        "Ingenia Elition X",
        "MAGNETOM Aera",
        "DISCOVERY MR750w",
        "SIGNA EXCITE",
        "CLASSIC CR",
        "MAGNETOM Vida Fit",
        "Titan3T",
        "Certegra",
        "MAGNETOM_ESSENZA",
        "SymphonyTim",
        "SIGNA Explorer",
        "Prodiva CS",
        "Espree",
        "SIGNA Artist",
        "Skyra (iQMR)",
        "Galan 3T",
        "MAGNETOM Vida fit",
        "Prodiva CX",
        "Intera",
        "MAGNETOM Vida",
        "MAGNETOM Sempra",
        "Achieva",
        "Prisma_fit",
        "Verio",
        "Titan",
        "Optima MR450w",
        "MAGNETOM Lumina",
        "Prisma",
        "ProstatID",
        "MAGNETOM Altea",
        "SIGNA Creator",
        "Skyra_fit",
        "Avanto_fit",
        "SIGNA Premier",
        "DISCOVERY MR750",
        "Signa HDxt",
        "QT Prostate MRI",
        "SIGNA Architect",
        "ECHELON_OVAL",
        "MAGNETOM Skyra Fit (BioMatrix)",
        "Avanto",
        "MAGNETOM Sola",
        "Spectra",
        "Artemis",
        "Skyra",
        "Ingenia",
        "Biograph_mMR",
        "SIGNA Voyager",
        "Achieva dStream",
        "Verio_DOT",
        "OsiriX",
        "MAGNETOM Prisma",
        "SIGNA PET/MR",
    }

    Predict_HD_Models = {
        "GENESIS_SIGNA",
        "Sonata",
        "PMOD",
        "Espree",
        "Ingenia",
        "Symphony",
        "TrioTim",
        "SIGNA EXCITE",
        "SymphonyTim",
        "Verio",
        "Skyra",
        "MfctModNm",
        "DISCOVERY MR750w",
        "Achieva",
        "MAGNETOM Vida",
        "Prisma_fit",
        "Signa HDxt",
        "SIGNA HDx",
        "SIGNA",
        "Avanto",
        "Allegra",
        "Optima MR450w",
        "Intera",
    }

    # total_models = set(Nebraska_Models) | set(Iowa_Models) | set(Predict_HD_Models)
    # print("Total Models: ", total_models, len(total_models))

    print(f"Predict HD Rows: {predict_rows}")
    print(f"Nebraska Rows: {nebraska_rows}")
    print(f"Iowa Rows: {iowa_rows}")
    print(f"Total Rows: {predict_rows + nebraska_rows + iowa_rows}")
