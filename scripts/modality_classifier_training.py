import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold

import time
from six import StringIO

# from IPython.display import Image
import pydotplus
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import argparse

from dcm_classify.dicom_config import (
    # make_unique_ordered_list,
    drop_columns_with_no_series_specific_information,
)

rcParams.update({"figure.autolayout": True})

# overwrite objects for new data modalities
imagetype_to_integer_mapping = {
    "adc": 0,
    "fa": 1,
    "tracew": 2,
    "t2w": 3,
    "t2starw": 4,
    "t1w": 5,
    "flair": 6,
    "field_map": 7,
    "dwig": 8,
    "dwi_multishell": 9,
    "fmri": 10,
}

modality_columns = [
    "ImageTypeADC",
    "ImageTypeFA",
    "ImageTypeTrace",
    "SeriesVolumeCount",
    "EchoTime",
    "RepetitionTime",
    "FlipAngle",
    "PixelBandwidth",
    "SAR",
    "Diffusionb-valueCount",
    "Diffusionb-valueMax",
]


def generate_training_data(label_dataframe: str, dicom_dataframe: str, name: str):
    """
    Generate data for model training. Function modifies the "everything" DICOM dataframe by injecting labels from
    label_dataframe.
    :param label_dataframe: filepath to the label dataframe
    :param dicom_dataframe: filepath of "everything" dicom dataframe
    :return:
        x - training data
        y - labels
    """
    input_output_training_columns = modality_columns + ["SeriesInstanceUID", "Modality"]

    label_df = pd.read_excel(label_dataframe)
    dicom_df = pd.read_excel(dicom_dataframe)

    df = pd.merge(dicom_df, label_df, on="SeriesInstanceUID")
    pima = pd.DataFrame(df[input_output_training_columns])

    pima = pima[
        (pima["Modality"] == "adc")
        | (pima["Modality"] == "fa")
        | (pima["Modality"] == "tracew")
        | (pima["Modality"] == "t2w")
        | (pima["Modality"] == "t2starw")
        | (pima["Modality"] == "t1w")
        | (pima["Modality"] == "flair")
        | (pima["Modality"] == "field_map")
        | (pima["Modality"] == "dwig")
        | (pima["Modality"] == "dwi_multishell")
        | (pima["Modality"] == "fmri")
    ]

    pima["Modality"].replace(imagetype_to_integer_mapping, inplace=True)
    pima["Modality"] = pima["Modality"].astype(int)
    pima["Modality"] = pima["Modality"].astype("category")

    model_generation_df = pd.DataFrame(pima[input_output_training_columns])
    del pima

    model_generation_df = model_generation_df.dropna(axis=0)
    model_generation_df.to_excel(f"../training_outputs/{name}_model_generation_df.xlsx")

    X = model_generation_df[modality_columns]
    y = model_generation_df["Modality"]
    return X, y


def generate_training_data2(dicom_dataframe: str, name: str):
    """
    Generate data for model training. Function modifies the "everything" DICOM dataframe by injecting labels from
    label_dataframe.
    :param label_dataframe: filepath to the label dataframe
    :param dicom_dataframe: filepath of "everything" dicom dataframe
    :return:
        x - training data
        y - labels
    """
    input_output_training_columns = modality_columns + ["SeriesInstanceUID", "Modality"]
    input_output_training_columns = make_unique_ordered_list(
        input_output_training_columns
    )

    df = pd.read_excel(dicom_dataframe)
    pima = pd.DataFrame(df[input_output_training_columns])

    pima = pima[
        (pima["Modality"] == "adc")
        | (pima["Modality"] == "fa")
        | (pima["Modality"] == "tracew")
        | (pima["Modality"] == "t2w")
        | (pima["Modality"] == "t2starw")
        | (pima["Modality"] == "t1w")
        | (pima["Modality"] == "flair")
        | (pima["Modality"] == "field_map")
        | (pima["Modality"] == "dwig")
        | (pima["Modality"] == "dwi_multishell")
        | (pima["Modality"] == "fmri")
    ]

    pima["Modality"].replace(imagetype_to_integer_mapping, inplace=True)
    pima["Modality"] = pima["Modality"].astype(int)
    pima["Modality"] = pima["Modality"].astype("category")

    model_generation_df = pd.DataFrame(pima[input_output_training_columns])
    del pima

    model_generation_df = model_generation_df.dropna(axis=0)
    model_generation_df.to_excel(f"../training_outputs/{name}_model_generation_df.xlsx")

    X = model_generation_df[modality_columns]
    y = model_generation_df["Modality"]
    return X, y


# def train_model(x1, x2, x3, y1, y2, y3, use_dt):
def train_model(use_dt):
    """
    Function to train modality classification model. By default uses Random Forrest classifier unless user specifies
    the use of Decision Tree classifier. This function also produces figures to analyze trained model and saves
    the model using skl2onnx library.
    :param x: train data
    :param y: labels
    :param use_dt: if true use Decision Tree over Random Forrest
    """

    # # split data into 70% training and 30% test
    # x1_train, x1_test, y1_train, y1_test = train_test_split(
    #     x1, y1, test_size=0.3, random_state=1
    # )
    # x2_train, x2_test, y2_train, y2_test = train_test_split(
    #     x2, y2, test_size=0.3, random_state=1
    # )
    # x3_train, x3_test, y3_train, y3_test = train_test_split(
    #     x3, y3, test_size=0.3, random_state=1
    # )
    # x_train = pd.concat([x1_train, x2_train, x3_train], axis=0)
    # y_train = pd.concat([y1_train, y2_train, y3_train], axis=0)
    # x_test = pd.concat([x1_test, x2_test, x3_test], axis=0)
    # y_test = pd.concat([y1_test, y2_test, y3_test], axis=0)
    # print(f"SHAPE train: {y_train.shape}")
    # print(f"SHAPE test: {y_test.shape}")

    unique_df = pd.read_excel(
        "../training_outputs/all_model_generation_df_normalized_unique.xlsx"
    )
    norm_df = pd.read_excel(
        "../training_outputs/all_model_generation_df_normalized.xlsx"
    )
    all_df = pd.read_excel("../training_outputs/all_model_generation_df.xlsx")

    uX = unique_df[modality_columns].values
    uy = unique_df["Modality"].values

    allX = all_df[modality_columns].values
    ally = all_df["Modality"].values

    # kf = KFold(n_splits=4, random_state=20, shuffle=True)
    kf = KFold(n_splits=4, random_state=5, shuffle=True)

    ti = None
    tt = None
    for unique_train_index, unique_test_index in kf.split(uX):
        ti = unique_train_index
        tt = unique_test_index
        break
    uX_train, uX_test = uX[ti], uX[tt]

    train_indexes = []
    test_indexes = []
    for index, row in norm_df.iterrows():
        if any((uX_train[:] == row[modality_columns].values).all(1)):
            # if row[modality_columns].values in uX_train:
            train_indexes.append(index)
        else:
            test_indexes.append(index)

    x_train, x_test = allX[train_indexes], allX[test_indexes]
    y_train, y_test = ally[train_indexes], ally[test_indexes]
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_test, return_counts=True))

    # x_fi = []
    # y_fi = []
    # for i in range(11):
    #     yi = np.where(y_test == i)[0][:8]
    #     x_fi.append(x_test[yi])
    #     y_fi.append(y_test[yi])
    #
    # # print(x_fi)
    # x_test = np.array(x_fi).reshape(11*8, 11)
    # y_test = np.array(y_fi).flatten()

    # TODO: Experiment with test set of same number of examples per modality to get better Feature importance

    if use_dt:
        print("Using Decision Tree Classifier")
        clf = DecisionTreeClassifier(max_depth=5)
    else:
        print("Using Random Forest Classifier")
        clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=99)
        # clf = AdaBoostClassifier(n_estimators=100, random_state=99, learning_rate=0.2)
    # Train Decision Tree Classifer
    clf = clf.fit(x_train, y_train)
    # import graphviz
    # for i in range(3):
    #     tree = clf.estimators_[i]
    #     dot_data = export_graphviz(tree,
    #                                feature_names=modality_columns,
    #                                filled=True,
    #                                max_depth=7,
    #                                impurity=False,
    #                                proportion=True)
    #     # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #     # graph.write_png(f"../training_outputs/rf_estimator_{i}.png")
    #     graph = graphviz.Source(dot_data, format="png")
    #     graph.render(filename=f"../training_outputs/rf_estimator_{i}")

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Generate figures to analyze the trained model
    # https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.xaxis.set_ticklabels(c)
    ax.yaxis.set_ticklabels(c)
    # _ = ax.set_title(
    #     f"Confusion Matrix for {clf.__class__.__name__}\non the test data"
    # )
    ax.tick_params(axis="x", colors="red")
    plt.xticks(rotation=90)
    plt.savefig("../training_outputs/confusion_matrix.png", dpi=400)

    input_vector_size: int = len(modality_columns)

    if use_dt:
        dot_data = StringIO()
        export_graphviz(
            clf,
            out_file=dot_data,
            filled=True,
            rounded=True,
            special_characters=True,
            feature_names=modality_columns,
            class_names=[x for x in imagetype_to_integer_mapping.keys()],
        )
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png("../training_outputs/test.png")
        # Image(graph.create_png())
    else:
        start_time = time.time()
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
        elapsed_time = time.time() - start_time

        print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

        feature_names = modality_columns
        forest_importances = pd.Series(importances, index=feature_names)

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.savefig("../training_outputs/rf_importance_norm.pdf")

    # save the model file
    model_filename: str = "../models/rf_classifier.onnx"
    # Convert into ONNX format
    initial_type = [("float_input", FloatTensorType([None, input_vector_size]))]
    onx = convert_sklearn(
        clf, name="RandomForestImageTypeClassifier", initial_types=initial_type
    )
    with open(model_filename, "wb") as f:
        f.write(onx.SerializeToString())


def inference_on_all_data(model_filename: str, model_df: str, out_file: str):
    """
    This function runs inference on all data and produces figures dataframe containing class specific prediction.
    :param model_filename: filepath to the onnx model file.
    """
    print("\nInference on all.\n")

    df = pd.read_excel(model_df)
    e_inputs = pd.DataFrame(df[modality_columns])
    e_inputs.fillna(-1000000, inplace=True)
    e_inputs.replace(np.nan, -1000000, inplace=True)
    e_inputs.replace("nan", -1000000, inplace=True)

    sess = rt.InferenceSession(model_filename)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    prob_name = sess.get_outputs()[1].name

    tester = e_inputs.astype(np.float32).to_numpy()
    pred_onx_run_output = sess.run([label_name, prob_name], {input_name: tester})
    pred_onx = pred_onx_run_output[0]
    probability_onx = pred_onx_run_output[1]
    prob_df = pd.DataFrame(probability_onx)

    outputs = df
    outputs["GUESS_ONNX_CODE"] = pred_onx
    for col in prob_df.columns:
        outputs[f"GUESS_ONNX_idx{col}"] = prob_df[col]
    outputs.to_excel(out_file)


if __name__ == "__main__":
    # description = "author Hans J. Johnson | Script for attempting to classify image types based on dicom header fields"
    # parser = argparse.ArgumentParser(description=description)
    # parser.add_argument(
    #     "--label_df", metavar="file", required=False, help="Path to dataframe containing labels."
    # )
    # parser.add_argument(
    #     "--dicom_df", metavar="file", required=False, help="Output path for Dicom Dataframe."
    # )
    # parser.add_argument(
    #     "--use_DT", action="store_true", required=False, help="Ruleset defining config file."
    # )
    #
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     exit(1)
    # args = parser.parse_args()

    # data
    # i_dcmf_df = "../data/iowaStroke_all_dicom.xlsx"
    # i_lbl_df = "../data/iowaStroke_labels.xlsx"
    # ix, iy = generate_training_data(i_lbl_df, i_dcmf_df, "iowaStroke")

    # p_dcmf_df = "../data/prostate_all_dicom_raw.xlsx"
    # p_lbl_df = "../data/prostate_labels.xlsx"
    # px, py = generate_training_data(p_lbl_df, p_dcmf_df, "prostate")
    #
    # t_dcmf_df = "../data/trackOn_all_dicom_raw.xlsx"
    # tx, ty = generate_training_data2(t_dcmf_df, "trackOn")

    # train_model(x1=ix, x2=px, x3=tx, y1=iy, y2=py, y3=ty, use_dt=False)
    train_model(use_dt=False)

    inference_on_all_data(
        "../models/rf_classifier.onnx",
        "../training_outputs/iowaStroke_model_generation_df.xlsx",
        "../training_outputs/iowaStroke_guess.xlsx",
    )
    inference_on_all_data(
        "../models/rf_classifier.onnx",
        "../training_outputs/prostate_model_generation_df.xlsx",
        "../training_outputs/prostate_guess.xlsx",
    )
    inference_on_all_data(
        "../models/rf_classifier.onnx",
        "../training_outputs/trackOn_model_generation_df.xlsx",
        "../training_outputs/trackOn_guess.xlsx",
    )
