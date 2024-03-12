import argparse
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pydotplus
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
    classification_report,
)


rcParams.update({"figure.autolayout": True})
plt.rcParams.update({"font.size": 7})

# features currently used
feautures = [
    "Diffusionb-valueBool",
    "Echo Time",
    "Echo Train Length",
    "Flip Angle",
    "HasDiffusionGradientOrientation",
    "Image Type_ADC",
    "Image Type_DERIVED",
    "Image Type_DIFFUSION",
    "Image Type_EADC",
    "Image Type_FA",
    "Image Type_ORIGINAL",
    "Image Type_TRACEW",
    "Imaging Frequency",
    "Inversion Time",
    "Manufacturer_siemens",
    "Number of Averages",
    "Pixel Bandwidth",
    "Repetition Time",
    "SAR",
    "Scanning Sequence_GR",
    "Scanning Sequence_IR",
    "Scanning Sequence_SE",
    "Sequence Variant_MP",
    "Sequence Variant_SP",
    "dB/dt",
    "has_b0",
    "has_pos_b0",
    "likely_diffusion",
]

# overwrite objects for new data modalities
imagetype_to_integer_mapping = {
    "t1w": 0,
    "gret2star": 1,
    "t2w": 2,
    "flair": 3,
    "b0": 4,
    "tracew": 5,
    "adc": 6,
    "fa": 7,
    "eadc": 8,
    "dwig": 9,
}

integer_to_imagetype_mapping = {v: k for k, v in imagetype_to_integer_mapping.items()}


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


def generate_training_data(input_dataframe: str):
    """
    Generate data for model training. Function modifies the "everything" DICOM dataframe by injecting labels from
    label_dataframe.
    :param input_dataframe: filepath to the label dataframe
    :return:
        x - training data
        y - labels
        columns - list of feature columns
    """
    # Load and clean the DataFrame
    df = pd.read_excel(input_dataframe)

    # Consolidate DataFrame operations
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    # df.drop(columns=["FileName"] + features_to_remove, inplace=True)
    df = df[feautures + ["label"]]
    # Keep only rows where 'label' matches keys in imagetype_to_integer_mapping
    df = df[df["label"].isin(imagetype_to_integer_mapping.keys())]

    # Normalize the data
    df["label_encoded"] = df["label"].map(imagetype_to_integer_mapping)
    df.fillna({col: 0 if "Type" in col else -12345 for col in df.columns}, inplace=True)

    # # Strip whitespace from string columns
    # str_cols = df.select_dtypes(include=["object"])
    # df[str_cols.columns] = str_cols.apply(lambda x: x.str.strip())

    # Replace infinite values with NaN and drop rows with NaN values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Extract features and labels
    y = df["label_encoded"].astype("int32").to_numpy()
    x = df.drop(columns=["label", "label_encoded"]).astype("float32").to_numpy()

    [print(f'"{col}",') for col in df.columns]

    rows_with_inf = np.any(np.isinf(x), axis=1)
    # Check if there are any such rows
    if np.any(rows_with_inf):
        # Drop these rows
        x_clean = x[~rows_with_inf]
        y_clean = y[~rows_with_inf]
    else:
        x_clean = x
        y_clean = y

    return x_clean, y_clean, df.columns.drop(["label", "label_encoded"]).tolist()


def train(
    x: np.array, y: np.array, columns, samples_per_class=800, use_dt: bool = False
):
    # make x_train and y_train unique
    # Find indices of unique rows
    _, unique_indices = np.unique(x, axis=0, return_index=True)
    # print("X Train shape:", x_train.shape)
    # Select only unique rows in both x and y
    x_unique = x[unique_indices]
    print("X Unique shape:", x_unique.shape)
    y_unique = y[unique_indices]
    print("X Unique shape:", y_unique.shape)

    # split data into 80% training and 20% test
    x_train, x_test, y_train, y_test = train_test_split(
        x_unique, y_unique, test_size=0.2, random_state=99, stratify=y_unique
    )
    print(np.unique(y_test, return_counts=True))

    x_train_balanced, y_train_balanced = [], []
    for class_label in range(10):
        print(f"Class {class_label}")
        x_train_class = x_train[y_train == class_label]
        y_train_class = y_train[y_train == class_label]

        # Determine if sampling should be with or without replacement
        replace = len(x_train_class) < samples_per_class
        # Sample the data

        x_train_balanced.append(
            pd.DataFrame(x_train_class).sample(
                n=samples_per_class, replace=replace, random_state=99
            )
        )
        y_train_balanced.append(
            pd.Series(y_train_class).sample(
                n=samples_per_class, replace=replace, random_state=99
            )
        )

    x_train_balanced = pd.concat(x_train_balanced, ignore_index=True).to_numpy()
    y_train_balanced = pd.concat(y_train_balanced, ignore_index=True).to_numpy()
    # y_train_balanced = y_train
    print(x_train_balanced.shape, y_train_balanced.shape)

    # clf = RandomForestClassifier(
    #     n_estimators=50, max_depth=9, random_state=99, max_features=20
    # )

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=9, random_state=99, max_features=20
    )
    clf = OneVsRestClassifier(rf)

    clf.fit(x_train_balanced, y_train_balanced)

    # if use_dt:
    #     print("Using Decision Tree Classifier")
    #     clf = DecisionTreeClassifier(max_depth=5)
    # else:
    #     print("Using Random Forest Classifier")
    #     clf = RandomForestClassifier(n_estimators=150, max_depth=9, random_state=99)
    #
    # # clf = clf.fit(x_train, y_train)
    # import graphviz
    #
    # for i in range(3):
    #     tree = clf.estimators_[i]
    #     dot_data = export_graphviz(
    #         tree,
    #         feature_names=feautures,
    #         filled=True,
    #         max_depth=7,
    #         impurity=False,
    #         proportion=True,
    #     )
    #     # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #     # graph.write_png(f"../training_outputs/rf_estimator_{i}.png")
    #     graph = graphviz.Source(dot_data, format="png")
    #     graph.render(filename=f"./rf_estimator_{i}")

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    mask_test = y_test != 10
    y_masked = y_test[mask_test]
    y_pred_masked = y_pred[mask_test]
    print("Accuracy without loc:", metrics.accuracy_score(y_masked, y_pred_masked))

    cm = confusion_matrix(y_test, y_pred)

    # Calculating accuracy for each class
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    # Print accuracy for each class
    for i, accuracy in enumerate(class_accuracies):
        print(f"Accuracy for class {i}: {accuracy:.2f}")

    # # Generate figures to analyze the trained model
    # # https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
    # # Generate normalized confusion matrix figure
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, normalize="true")
    # classes = [x for x in imagetype_to_integer_mapping.keys()]
    # ax.xaxis.set_ticklabels(classes)
    # ax.yaxis.set_ticklabels(classes)
    # _ = ax.set_title(
    #     f"Normalized Confusion Matrix for {clf.__class__.__name__}\non the test data"
    # )
    # ax.tick_params(axis="x", colors="red")
    # plt.xticks(rotation=90)
    # plt.savefig("./confusion_matrix_normalized.png", dpi=400)

    # Compute the normalized confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    # Limit the decimal points to 3
    cm = np.around(cm, decimals=2)

    # Display the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=imagetype_to_integer_mapping.keys()
    )
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format=".2f")

    _ = ax.set_title(
        f"Normalized Confusion Matrix for {clf.__class__.__name__}\non the test data"
    )
    ax.tick_params(axis="x", colors="red")
    plt.xticks(rotation=90)
    plt.savefig("./confusion_matrix_normalized.png", dpi=400)
    input_vector_size: int = len(columns)

    # if use_dt:
    #     dot_data = StringIO()
    #     export_graphviz(
    #         clf,
    #         out_file=dot_data,
    #         filled=True,
    #         rounded=True,
    #         special_characters=True,
    #         feature_names=columns,
    #         class_names=[x for x in imagetype_to_integer_mapping.keys()],
    #     )
    #     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #     graph.write_png("./test.png")
    #     # Image(graph.create_png())
    # else:
    #     start_time = time.time()
    #     importances = clf.feature_importances_
    #     std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    #     elapsed_time = time.time() - start_time
    #
    #     print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    #
    #     feature_names = columns
    #     forest_importances = pd.Series(importances, index=feature_names)
    #     forest_importances = forest_importances.sort_values(ascending=False)
    #     # print the feature importances vertically
    #     print("\nFeature importances sorted:")
    #     print(forest_importances)

    #     fig, ax = plt.subplots()
    #     forest_importances.plot.bar(yerr=std, ax=ax)
    #     ax.set_title("Feature importances using MDI")
    #     ax.set_ylabel("Mean decrease in impurity")
    #     fig.tight_layout()
    #     plt.savefig("./rf_importance_norm_all.pdf")

    # save the model file
    model_filename: str = "/home/mbrzus/Desktop/ova_rf_classifier.onnx"
    # Convert into ONNX format
    initial_type = [("float_input", FloatTensorType([None, input_vector_size]))]
    onx = convert_sklearn(
        clf, name="OneVsAllRandomForestImageTypeClassifier", initial_types=initial_type
    )
    with open(model_filename, "wb") as f:
        f.write(onx.SerializeToString())

    # TODO: CHECK ONNX version to create the model MUST be the same version to read the model
    return 0  # forest_importances


def k_fold_cross_validation(
    x, y, input_vector_size, n_splits=2, samples_per_class=3000
):
    print(x.shape, y.shape)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    overall_accuracies = []
    class_accuracies = {i: [] for i in range(12)}
    best_accuracy = 0
    best_model = None
    all_y_true = []
    all_y_pred = []
    cumulative_accuracies = []

    # make x_train and y_train unique
    # Find indices of unique rows
    _, unique_indices = np.unique(x, axis=0, return_index=True)
    # print("X Train shape:", x_train.shape)
    # Select only unique rows in both x and y
    x_unique = x[unique_indices]
    print("X Unique shape:", x_unique.shape)
    y_unique = y[unique_indices]
    print(np.unique(y_unique, return_counts=True))

    for fold, (train_index, test_index) in enumerate(kf.split(x_unique, y_unique)):
        x_train, x_test = x_unique[train_index], x_unique[test_index]
        y_train, y_test = y_unique[train_index], y_unique[test_index]

        x_train_balanced, y_train_balanced = [], []
        for class_label in range(12):
            print(f"Class {class_label}")
            x_train_class = x_train[y_train == class_label]
            y_train_class = y_train[y_train == class_label]

            # Determine if sampling should be with or without replacement
            replace = len(x_train_class) < samples_per_class

            # Sample the data
            x_train_balanced.append(
                pd.DataFrame(x_train_class).sample(
                    n=samples_per_class, replace=replace, random_state=99
                )
            )
            y_train_balanced.append(
                pd.Series(y_train_class).sample(
                    n=samples_per_class, replace=replace, random_state=99
                )
            )
            # Combine the balanced data
        x_train_balanced = pd.concat(x_train_balanced, ignore_index=True).to_numpy()
        y_train_balanced = pd.concat(y_train_balanced, ignore_index=True).to_numpy()

        print(x_train_balanced.shape, y_train_balanced.shape)
        clf = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=99)
        clf.fit(x_train_balanced, y_train_balanced)

        y_pred = clf.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        overall_accuracy = accuracy_score(y_test, y_pred)
        overall_accuracies.append(overall_accuracy)
        cumulative_accuracies.append(overall_accuracies.copy())

        if overall_accuracy > best_accuracy:
            best_accuracy = overall_accuracy
            best_model = clf

        # Append true and predicted labels for this fold
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"Fold {fold + 1} Metrics:")
        print(f"Accuracy: {overall_accuracy:.2f}")
        print(
            classification_report(
                y_test, y_pred, target_names=integer_to_imagetype_mapping.values()
            )
        )

        for i in range(12):
            class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            class_accuracies[i].append(class_accuracy)

    # Print average results
    print(f"Average Overall Accuracy: {np.mean(overall_accuracies):.2f}")
    for i in range(12):
        class_name = integer_to_imagetype_mapping.get(i, "Unknown")
        print(
            f"Average Accuracy for Class {class_name}: {np.mean(class_accuracies[i]):.2f}"
        )

    # Calculate and print classification report for all folds
    print("Classification Report for all folds:")
    print(
        classification_report(
            all_y_true, all_y_pred, target_names=integer_to_imagetype_mapping.values()
        )
    )

    # Calculate and print standard deviation progression
    print("Standard Deviation Progression:")
    for i, accs in enumerate(cumulative_accuracies, start=2):
        std_dev = np.std(accs)
        print(f"Std Dev up to fold {i}: {std_dev:.2f}")

    # Save the best model
    model_filename = "retrain_models/rf_classifier-kfold.onnx"
    initial_type = [("float_input", FloatTensorType([None, input_vector_size]))]
    onx = convert_sklearn(
        best_model, name="RandomForestImageTypeClassifier", initial_types=initial_type
    )
    with open(model_filename, "wb") as f:
        f.write(onx.SerializeToString())

    return best_model
    # return 0


def inference_on_all_data(model_filename: str, model_df: str, out_file: str):
    """
    This function runs inference on all data and produces figures dataframe containing class specific prediction.
    :param model_filename: filepath to the onnx model file.
    """
    print("\nInference on all.\n")

    df = pd.read_excel(model_df)
    # drop columns with 'Unnamed' in the name
    e_inputs = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    # drop the FileName column
    e_inputs = e_inputs.drop(columns=["FileName", "label"])
    existing_columns_to_drop = [
        col for col in features_to_remove if col in e_inputs.columns
    ]
    e_inputs = e_inputs.drop(columns=existing_columns_to_drop)
    print(e_inputs.shape)
    # for columns with 'Type' in the name, replace NaN with 0
    for col in e_inputs.columns:
        if "Type" in col:
            e_inputs[col].fillna(0, inplace=True)
        else:
            e_inputs[col].fillna(-12345, inplace=True)

    x = e_inputs.to_numpy().astype("float32")
    # Find rows with infinite values
    rows_with_inf = np.any(np.isinf(x), axis=1)
    # Check if there are any such rows
    e_inputs = e_inputs[~rows_with_inf].reset_index(drop=True)
    df = df[~rows_with_inf].reset_index(drop=True)

    # e_inputs.fillna(-1000000, inplace=True)
    # e_inputs.replace(np.nan, -1000000, inplace=True)
    # e_inputs.replace("nan", -1000000, inplace=True)

    sess = rt.InferenceSession(model_filename)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    prob_name = sess.get_outputs()[1].name

    tester = e_inputs.astype(np.float32).to_numpy()
    pred_onx_run_output = sess.run([label_name, prob_name], {input_name: tester})
    pred_onx = pred_onx_run_output[0]
    # pred_onnx_str = [list(imagetype_to_integer_mapping.keys())[x] for x in pred_onx]
    probability_onx = pred_onx_run_output[1]
    prob_df = pd.DataFrame(probability_onx)

    outputs = df
    outputs["GUESS_ONNX_CODE"] = pred_onx
    outputs["GUESS_ONNX"] = df["GUESS_ONNX_CODE"].map(integer_to_imagetype_mapping)
    for col in prob_df.columns:
        outputs[f"GUESS_ONNX_idx{col}"] = prob_df[col]
    outputs.to_excel(out_file)


def inference_metrcis_test(model_filename: str, model_df: str, out_file: str, features):
    """
    This function runs inference on all data and produces figures dataframe containing class specific prediction.
    :param model_filename: filepath to the onnx model file.
    """
    print("\nInference on all.\n")

    df = pd.read_excel(model_df)
    df = df[df["label"].isin(imagetype_to_integer_mapping.keys())]
    # Filter rows based on available classes

    # df = df[df["label"].isin(filtered_mapping.keys())]
    print(df.shape)
    available_classes = df["label"].unique()
    filtered_mapping = {
        k: v for k, v in imagetype_to_integer_mapping.items() if k in available_classes
    }
    print(filtered_mapping)
    # Normalize the data
    df["label_encoded"] = df["label"].map(imagetype_to_integer_mapping)
    df.fillna({col: 0 if "Type" in col else -12345 for col in df.columns}, inplace=True)

    # Assuming 'label' column contains the true labels
    true_labels_encoded = df["label_encoded"].values

    # drop columns with 'Unnamed' in the name
    e_inputs = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    print(e_inputs.shape)
    # drop the FileName column
    e_inputs = e_inputs.drop(columns=["FileName", "label", "label_encoded"])
    existing_columns_to_drop = [
        col for col in features_to_remove if col in e_inputs.columns
    ]
    e_inputs = e_inputs.drop(columns=existing_columns_to_drop)

    # Add missing features
    for feature in features:
        if feature not in e_inputs.columns:
            e_inputs[feature] = 0 if "Type" in feature else -12345
    print(e_inputs.columns)
    print(e_inputs.shape)
    # for columns with 'Type' in the name, replace NaN with 0
    for col in e_inputs.columns:
        if "Type" in col:
            e_inputs[col].fillna(0, inplace=True)
        else:
            e_inputs[col].fillna(-12345, inplace=True)

    e_inputs = e_inputs[features]
    x = e_inputs.to_numpy().astype("float32")
    # Find rows with infinite values
    rows_with_inf = np.any(np.isinf(x), axis=1)
    # Check if there are any such rows
    e_inputs = e_inputs[~rows_with_inf].reset_index(drop=True)
    df = df[~rows_with_inf].reset_index(drop=True)

    # e_inputs.fillna(-1000000, inplace=True)
    # e_inputs.replace(np.nan, -1000000, inplace=True)
    # e_inputs.replace("nan", -1000000, inplace=True)

    sess = rt.InferenceSession(model_filename)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    prob_name = sess.get_outputs()[1].name

    # tester = e_inputs.astype(np.float32).to_numpy()
    # pred_onx_run_output = sess.run([label_name, prob_name], {input_name: tester})
    # pred_onx = pred_onx_run_output[0]
    # # pred_onnx_str = [list(imagetype_to_integer_mapping.keys())[x] for x in pred_onx]
    # probability_onx = pred_onx_run_output[1]
    # prob_df = pd.DataFrame(probability_onx)

    tester = e_inputs.astype(np.float32).to_numpy()
    pred_onx_run_output = sess.run([label_name, prob_name], {input_name: tester})
    pred_onx = pred_onx_run_output[0].flatten()
    probability_onx = pred_onx_run_output[1]
    prob_df = pd.DataFrame(probability_onx)

    lbs = list(np.unique(pred_onx))
    # Compute metrics (adjusted for available classes)
    accuracy = accuracy_score(true_labels_encoded, pred_onx)
    print(f"Accuracy: {accuracy}")
    cls_report = classification_report(
        true_labels_encoded,
        pred_onx,
        labels=lbs,
        target_names=[integer_to_imagetype_mapping[i] for i in lbs],
    )

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(cls_report)

    outputs = df
    outputs["GUESS_ONNX_CODE"] = pred_onx
    outputs["GUESS_ONNX"] = df["GUESS_ONNX_CODE"].map(integer_to_imagetype_mapping)
    for col in prob_df.columns:
        outputs[f"GUESS_ONNX_idx{col}"] = prob_df[col]
    outputs.to_excel(out_file)


def perform_grid_search(
    x, y, output_excel_file, samples_per_class=5000, n_splits=5, n_jobs=-1
):
    """
    Perform grid search to find the best parameters for RandomForestClassifier.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training labels.
    output_excel_file (str): File path to save the grid search results as Excel.
    n_splits (int): Number of folds for cross-validation.
    n_jobs (int): Number of jobs to run in parallel (-1 means using all processors).

    Returns:
    best_model (RandomForestClassifier): The best model from grid search.
    """

    # Define the parameter grid
    param_grid = {
        "n_estimators": [1, 10, 25, 50, 75, 100, 150, 200],
        "max_depth": [1, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
        # Add more parameters if needed
    }
    # make x_train and y_train unique
    # Find indices of unique rows
    _, unique_indices = np.unique(x, axis=0, return_index=True)
    # print("X Train shape:", x_train.shape)
    # Select only unique rows in both x and y
    x_unique = x[unique_indices]
    print("X Unique shape:", x_unique.shape)
    y_unique = y[unique_indices]
    print("X Unique shape:", y_unique.shape)

    x_train_balanced, y_train_balanced = [], []
    for class_label in range(12):
        x_train_class = x_unique[y_unique == class_label]
        y_train_class = y_unique[y_unique == class_label]

        # Determine if sampling should be with or without replacement
        replace = len(x_train_class) < samples_per_class

        # Sample the data
        x_train_balanced.append(
            pd.DataFrame(x_train_class).sample(
                n=samples_per_class, replace=replace, random_state=99
            )
        )
        y_train_balanced.append(
            pd.Series(y_train_class).sample(
                n=samples_per_class, replace=replace, random_state=99
            )
        )
        # Combine the balanced data
    x_train_balanced = pd.concat(x_train_balanced, ignore_index=True).to_numpy()
    y_train_balanced = pd.concat(y_train_balanced, ignore_index=True).to_numpy()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=99),
        param_grid=param_grid,
        cv=n_splits,
        verbose=2,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=n_jobs,
    )

    # Fit GridSearchCV
    grid_search.fit(x_train_balanced, y_train_balanced)

    # Extract the best model
    best_model = grid_search.best_estimator_

    # Save the results to a pandas DataFrame and then to an Excel file
    results = pd.DataFrame(grid_search.cv_results_)
    results.to_excel(output_excel_file, index=False)

    return best_model


if __name__ == "__main__":
    # get all feature names needed for training
    # data_file = "/home/mbrzus/programming/dcm_train_data/training/labeled/no_IR/combined_all_Jan30_NO_IR_LOC.xlsx"

    # # K-fold test
    # data_file = "/home/mbrzus/programming/dcm_train_data/training/labeled/no_IR/combined_all_Jan30_NO_IR_LOC.xlsx"
    # # data_file = "/home/mbrzus/programming/dcm_train_data/training/labeled/no_IR/combined_all_Jan30_NO_IR_small_balanced_500.xlsx"
    # x, y, cols = generate_training_data(data_file)
    # best_model = k_fold_cross_validation(
    #     x, y, len(cols), n_splits=10, samples_per_class=5000
    # )

    # Feature importance analysis
    data_file = "/home/mbrzus/programming/dcm_train_data/V2-20240229/training/combined_all_training_data_Mar11_labeledNewField_augmented.xlsx"
    x, y, cols = generate_training_data(data_file)
    importances_all = train(x, y, cols, use_dt=False)

    # zero_values_names = importances_all[importances_all == 0].index
    # print("\nNames with 0 values in importances_all:")
    # for name in zero_values_names:
    #     print(f'"{name}",')
    #
    # small_values_names = importances_all[importances_all <= 1.0e-01].index
    # print("\nNames with small values in importances_all:")
    # for name in small_values_names:
    #     print(f'"{name}",')

    # # INFERENCE
    # data_file = "/home/mbrzus/programming/dcm_train_data/training/labeled/no_IR/combined_all_Jan30_NO_IR_LOC.xlsx"
    # inference_on_all_data(
    #     "retrain_models/rf_classifier.onnx",
    #     data_file,
    #     "/home/mbrzus/programming/dcm_train_data/training/labeled/no_IR/combined_all_Jan30_NO_IR_LOC_INFERENCE.xlsx",
    # )

    # # Minipig inference
    # data_file = "/home/mbrzus/programming/dcm_train_data/training/minipig/training_minipig_labeled.xlsx"
    # inference_metrcis_test(
    #     "retrain_models/rf_classifier-kfold.onnx",
    #     data_file,
    #     "/home/mbrzus/programming/dcm_train_data/training/minipig/training_minipig_INFERENCE.xlsx",
    #     features=feautures,
    # )

    # # GRID SEARCH
    # data_file = "/home/mbrzus/programming/dcm_train_data/training/labeled/no_IR/combined_all_Jan30_NO_IR_LOC.xlsx"
    # x, y, cols = generate_training_data(data_file)
    # best_model = perform_grid_search(
    #     x,
    #     y,
    #     "/home/mbrzus/programming/dcm_train_data/training/labeled/no_IR/grid_search_results.xlsx",
    #     n_splits=5,
    #     n_jobs=-1,
    # )
