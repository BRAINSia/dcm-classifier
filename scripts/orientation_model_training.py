#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.tree import export_graphviz

from io import StringIO

# from IPython.display import Image
import pydotplus
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

from .create_dicom_fields_sheet.py import make_unique_ordered_list


# overwrite objects for new data modalities
plane_to_integer_mapping = {
    "ax": 0,
    "cor": 1,
    "sag": 2,
}

# modify columns to exclude the orientation
# the decision tree for acqusition plane (in create_labels.py) takes care of that
plane_cols = [
    "ImageOrientationPatient_0",
    "ImageOrientationPatient_1",
    "ImageOrientationPatient_2",
    "ImageOrientationPatient_3",
    "ImageOrientationPatient_4",
    "ImageOrientationPatient_5",
]


def acquisition_plane_decision_tree2(a, t1, t2):
    if a[5] <= t1:
        return 0
    else:
        if a[0] <= t2:
            return 2
        else:
            return 1


def acquisition_plane_decision_tree(a, t1, t2):
    if a[5] <= t1:
        return 0
    else:
        if a[1] <= t2:
            return 1
        else:
            return 2


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
    input_output_training_columns = plane_cols + ["Orientation", "SeriesInstanceUID"]
    input_output_training_columns = make_unique_ordered_list(
        input_output_training_columns
    )

    label_df = pd.read_excel(label_dataframe)
    dicom_df = pd.read_excel(dicom_dataframe)

    df = pd.merge(dicom_df, label_df, on="SeriesInstanceUID")
    pima = pd.DataFrame(df[input_output_training_columns])

    pima["Orientation"].replace(plane_to_integer_mapping, inplace=True)
    pima["Orientation"] = pima["Orientation"].astype(int)
    pima["Orientation"] = pima["Orientation"].astype("category")

    model_generation_df = pd.DataFrame(pima[input_output_training_columns])
    del pima

    model_generation_df = model_generation_df.dropna(axis=0)
    model_generation_df.to_excel(
        f"../training_outputs/{name}_model_generation_df_plane.xlsx"
    )

    X = model_generation_df[plane_cols]
    y = model_generation_df["Orientation"]
    return X, y


def train_model(x1, x2, y1, y2, use_dt):
    """
    Function to train modality classification model. By default uses Random Forrest classifier unless user specifies
    the use of Decision Tree classifier. This function also produces figures to analyze trained model and saves
    the model using skl2onnx library.
    :param x: train data
    :param y: labels
    :param use_dt: if true use Decision Tree over Random Forrest
    """

    # split data into 70% training and 30% test
    x1_train, x1_test, y1_train, y1_test = train_test_split(
        x1, y1, test_size=0.3, random_state=3
    )
    x2_train, x2_test, y2_train, y2_test = train_test_split(
        x2, y2, test_size=0.3, random_state=3
    )
    x_train = pd.concat([x1_train, x2_train], axis=0)
    y_train = pd.concat([y1_train, y2_train], axis=0)
    x_test = pd.concat([x1_test, x2_test], axis=0)
    y_test = pd.concat([y1_test, y2_test], axis=0)

    if use_dt:
        print("Using Decision Tree Classifier")
        clf = DecisionTreeClassifier(max_depth=2)
    else:
        print("Using Random Forest Classifier")
        clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=99)
    # Train Decision Tree Classifer
    clf = clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)
    y_pred_replicate = np.apply_along_axis(
        acquisition_plane_decision_tree2, 1, x_test.values, t1=0.56, t2=0.85
    )
    y_pred_05_0707 = np.apply_along_axis(
        acquisition_plane_decision_tree2, 1, x_test.values, 0.5, 0.707
    )
    y_pred_05_05 = np.apply_along_axis(
        acquisition_plane_decision_tree2, 1, x_test.values, 0.5, 0.5
    )
    y_pred_05_085 = np.apply_along_axis(
        acquisition_plane_decision_tree2, 1, x_test.values, 0.5, 0.85
    )
    y_pred_056_0707 = np.apply_along_axis(
        acquisition_plane_decision_tree2, 1, x_test.values, 0.56, 0.707
    )
    y_pred_056_044 = np.apply_along_axis(
        acquisition_plane_decision_tree, 1, x_test.values, 0.56, 0.44
    )
    y_pred_056_05 = np.apply_along_axis(
        acquisition_plane_decision_tree, 1, x_test.values, 0.56, 0.5
    )
    y_pred_05_052 = np.apply_along_axis(
        acquisition_plane_decision_tree, 1, x_test.values, 0.5, 0.5
    )
    print("Accuracy Decision Tree:", metrics.accuracy_score(y_test, y_pred))
    print(
        "Accuracy with thresholds of 0.56 and 0.85 (replicating the Decision Tree):",
        metrics.accuracy_score(y_test, y_pred_replicate),
    )
    print(
        "Accuracy with thresholds of 0.5 and 0.707:",
        metrics.accuracy_score(y_test, y_pred_05_0707),
    )
    print(
        "Accuracy with thresholds of 0.5 and 0.5:",
        metrics.accuracy_score(y_test, y_pred_05_05),
    )
    print(
        "Accuracy with thresholds of 0.5 and 0.85:",
        metrics.accuracy_score(y_test, y_pred_05_085),
    )
    print(
        "Accuracy with thresholds of 0.56 and 0.707:",
        metrics.accuracy_score(y_test, y_pred_056_0707),
    )
    print("other tree")
    print(
        "Accuracy with thresholds of 0.56 and 0.44:",
        metrics.accuracy_score(y_test, y_pred_056_044),
    )
    print(
        "Accuracy with thresholds of 0.56 and 0.5:",
        metrics.accuracy_score(y_test, y_pred_056_05),
    )
    print(
        "Accuracy with thresholds of 0.5 and 0.5:",
        metrics.accuracy_score(y_test, y_pred_05_052),
    )

    # get class report
    rf_d = classification_report(
        y_test,
        y_pred,
        target_names=list(plane_to_integer_mapping.keys()),
        output_dict=True,
        zero_division=True,
    )
    rf_df = pd.DataFrame(rf_d).transpose()
    rf_dfs = [rf_df]

    # Genereta Classification Reports and save them as CSV files
    rf_df = pd.concat(rf_dfs, axis=0)
    rf_df.to_excel("../training_outputs/orientation_class_report.xlsx")

    # Generate figures to analyze the trained model
    # https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.xaxis.set_ticklabels(plane_to_integer_mapping.keys())
    ax.yaxis.set_ticklabels(plane_to_integer_mapping.keys())
    _ = ax.set_title(
        f"Confusion Matrix for {clf.__class__.__name__}\non the original data"
    )
    plt.savefig("../training_outputs/confusion_matrix_plane.png")

    input_vector_size: int = len(plane_cols)

    if use_dt:
        dot_data = StringIO()
        export_graphviz(
            clf,
            out_file=dot_data,
            filled=True,
            rounded=True,
            special_characters=True,
            feature_names=plane_cols,
            class_names=[x for x in plane_to_integer_mapping.keys()],
        )
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png("../training_outputs/test_plane.png")
        # Image(graph.create_png())

    # save the model file
    model_filename: str = "../training_outputs/dt_classifier_plane.onnx"
    # Convert into ONNX format
    initial_type = [("float_input", FloatTensorType([None, input_vector_size]))]
    onx = convert_sklearn(
        clf, name="RandomForestImageTypeClassifier", initial_types=initial_type
    )
    with open(model_filename, "wb") as f:
        f.write(onx.SerializeToString())


def inference_on_all_data(model_filename: str):
    """
    This function runs inference on all data and produces figures dataframe containing class specific prediction.
    :param model_filename: filepath to the onnx model file.
    """
    print("\nInference on all.\n")

    df = pd.read_excel("../training_outputs/model_generation_df_plane.xlsx")
    e_inputs = pd.DataFrame(df[plane_cols])
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
    outputs.to_excel("../training_outputs/guess_plane.xlsx")


if __name__ == "__main__":
    # description = "author Hans J. Johnson | Script for attempting to classify image types based on dicom header fields"
    # parser = argparse.ArgumentParser(description=description)
    # parser.add_argument(
    #     "--label_df",
    #     metavar="file",
    #     required=False,
    #     help="Path to dataframe containing labels.",
    # )
    # parser.add_argument(
    #     "--dicom_df",
    #     metavar="file",
    #     required=False,
    #     help="Output path for Dicom Dataframe.",
    # )
    # parser.add_argument(
    #     "--use_DT",
    #     action="store_true",
    #     required=False,
    #     help="Use decision tree over Random Forest.",
    # )

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     exit(1)
    # args = parser.parse_args()

    # data
    i_dcmf_df = "../data/iowaStroke_all_dicom.xlsx"
    i_lbl_df = "../data/iowaStroke_labels.xlsx"
    ix, iy = generate_training_data(i_lbl_df, i_dcmf_df, "iowaStroke")

    p_dcmf_df = "../data/prostate_all_dicom_raw.xlsx"
    p_lbl_df = "../data/prostate_labels.xlsx"
    px, py = generate_training_data(p_lbl_df, p_dcmf_df, "prostate")

    df1 = pd.read_excel("../training_outputs/iowaStroke_model_generation_df_plane.xlsx")
    df2 = pd.read_excel("../training_outputs/prostate_model_generation_df_plane.xlsx")
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    print(df.shape)
    df.to_excel("../training_outputs/model_generation_df_plane.xlsx")

    train_model(ix, px, iy, py, use_dt=True)
    inference_on_all_data("../training_outputs/dt_classifier_plane.onnx")

    # TODO: add additional inference function to run inference on new data
    # TODO: Add flags to control file creation. Example idea to create a data directory where
    #  all dataframes will be written, similarly figure directory, and specify exact model output path, etc.
