import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
    classification_report,
)


rcParams.update({"figure.autolayout": True})
plt.rcParams.update({"font.size": 7})

# features currently used for training
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

# current modality classification classes for training
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
    # Check if there are any rows with infinite values
    if np.any(rows_with_inf):
        # Drop these rows
        x_clean = x[~rows_with_inf]
        y_clean = y[~rows_with_inf]
    else:
        x_clean = x
        y_clean = y

    return x_clean, y_clean, df.columns.drop(["label", "label_encoded"]).tolist()


def train(x: np.array, y: np.array, columns, samples_per_class=800):
    """
    Train the model using the input training data.
    Args:
        x: training data
        y: labels
        columns: training feature names
        samples_per_class: number of samples per class to normalize class imbalance

    Returns:

    """
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
    for class_label in range(len(imagetype_to_integer_mapping.keys())):
        x_train_class = x_train[y_train == class_label]
        y_train_class = y_train[y_train == class_label]

        # Determine if sampling should be with or without replacement
        replace = len(x_train_class) < samples_per_class
        # Sample the standard number of samples per class
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
    print(x_train_balanced.shape, y_train_balanced.shape)

    # Random Forrest Classifier
    clf = RandomForestClassifier(
        n_estimators=50, max_depth=9, random_state=99, max_features=20
    )

    # One vs Rest Classifier
    # rf = RandomForestClassifier(
    #     n_estimators=50, max_depth=9, random_state=99, max_features=20
    # )
    # clf = OneVsRestClassifier(rf)

    # Train the model using the training sets
    clf.fit(x_train_balanced, y_train_balanced)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Save the first three trees of the Random Forrest Classifier for analysis
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

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Calculating accuracy for each class
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    # Print accuracy for each class
    for i, accuracy in enumerate(class_accuracies):
        print(f"Accuracy for class {i}: {accuracy:.2f}")

    # # Generate figures to analyze the trained model
    # # https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
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

    # Compute feature importances
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

    feature_names = columns
    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances = forest_importances.sort_values(ascending=False)
    # print the feature importances vertically
    print("\nFeature importances sorted:")
    print(forest_importances)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig("./rf_importance_norm_all.pdf")

    # save the model file using ONNX
    model_filename: str = "./rf_classifier.onnx"
    # Convert into ONNX format
    initial_type = [("float_input", FloatTensorType([None, input_vector_size]))]
    onx = convert_sklearn(
        clf, name="RandomForestImageTypeClassifier", initial_types=initial_type
    )
    with open(model_filename, "wb") as f:
        f.write(onx.SerializeToString())

    return forest_importances


def k_fold_cross_validation(x, y, input_vector_size, n_splits=5, samples_per_class=800):
    """
    Perform k-fold cross-validation to evaluate the model.
    Args:
        x:
        y:
        input_vector_size:
        n_splits:
        samples_per_class:

    Returns:

    """
    # Define the number of splits for cross-validation
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
        clf = RandomForestClassifier(
            n_estimators=50, max_depth=9, random_state=99, max_features=20
        )
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


def inference_on_all_data(model_filename: str, model_df: str, out_file: str):
    """
    This function runs inference on all data and produces figures dataframe containing class specific prediction.
    :param model_filename: filepath to the onnx model file.
    :param model_df: filepath to the dataframe containing the data to run inference on.
    :param out_file: filepath to save the output dataframe.
    """
    print("\nInference on all.\n")

    df = pd.read_excel(model_df)
    # drop columns with 'Unnamed' in the name
    e_inputs = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    # drop the FileName column
    e_inputs = e_inputs.drop(columns=["FileName", "label"])
    e_inputs = e_inputs[feautures]
    # # ensure there are no empty or Nan values
    # e_inputs.fillna({col: 0 if "Type" in col else -12345 for col in df.columns}, inplace=True)
    x = e_inputs.to_numpy().astype("float32")
    # Find rows with infinite values
    rows_with_inf = np.any(np.isinf(x), axis=1)
    # Check if there are any such rows
    e_inputs = e_inputs[~rows_with_inf].reset_index(drop=True)
    df = df[~rows_with_inf].reset_index(drop=True)

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


def inference_metric_test(model_filename: str, model_df: str, out_file: str, features):
    """
    This function runs inference on all data and produces figures dataframe containing class specific prediction.
    :param model_filename: filepath to the onnx model file.
    """
    print("\nInference on all.\n")

    df = pd.read_excel(model_df)
    df = df[df["label"].isin(imagetype_to_integer_mapping.keys())]
    # Filter rows based on available classes
    # available_classes = df["label"].unique()
    # Normalize the data
    df["label_encoded"] = df["label"].map(imagetype_to_integer_mapping)
    # # ensure there are no empty or Nan values
    # df.fillna({col: 0 if "Type" in col else -12345 for col in df.columns}, inplace=True)

    # Assuming 'label' column contains the true labels
    true_labels_encoded = df["label_encoded"].values

    # drop columns with 'Unnamed' in the name
    e_inputs = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    print(e_inputs.shape)
    # drop the FileName column
    e_inputs = e_inputs.drop(columns=["FileName", "label", "label_encoded"])
    e_inputs = e_inputs[feautures]

    # Add missing features if needed
    for feature in features:
        if feature not in e_inputs.columns:
            e_inputs[feature] = 0 if "Type" in feature else -12345

    e_inputs = e_inputs[features]
    x = e_inputs.to_numpy().astype("float32")
    # Find rows with infinite values
    rows_with_inf = np.any(np.isinf(x), axis=1)
    # Check if there are any such rows
    e_inputs = e_inputs[~rows_with_inf].reset_index(drop=True)
    df = df[~rows_with_inf].reset_index(drop=True)

    sess = rt.InferenceSession(model_filename)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    prob_name = sess.get_outputs()[1].name

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
        "max_depth": [1, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30],
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
    # # K-fold test
    # data_file = "data.xlsx"
    # x, y, cols = generate_training_data(data_file)
    # best_model = k_fold_cross_validation(
    #     x, y, len(cols), n_splits=10, samples_per_class=5000
    # )

    # Feature importance analysis
    data_file = "data.xlsx"
    x, y, cols = generate_training_data(data_file)
    importances_all = train(x, y, cols)

    # # find least important features to remove
    # zero_values_names = importances_all[importances_all == 0].index
    # # small_values_names = importances_all[importances_all <= 1.0e-01].index
    # print("\nNames with 0 values in importances_all:")
    # for name in zero_values_names:
    #     print(f'"{name}",')

    # # INFERENCE
    # data_file = "data.xlsx"
    # inference_on_all_data(
    #     "retrain_models/rf_classifier.onnx",
    #     data_file,
    #     "./combined_all_Jan30_NO_IR_LOC_INFERENCE.xlsx",
    # )

    # # GRID SEARCH
    # data_file = "data.xlsx"
    # x, y, cols = generate_training_data(data_file)
    # best_model = perform_grid_search(
    #     x,
    #     y,
    #     "./grid_search_results.xlsx",
    #     n_splits=5,
    #     n_jobs=-1,
    # )
