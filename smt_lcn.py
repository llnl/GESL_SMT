# Copyright (c) 2025, Lawrence Livermore National Security, LLC. All rights reserved. LLNL-CODE-2003701
# EXPORT CONTROLLED INFORMATION
# This work was produced at the Lawrence Livermore National Laboratory (LLNL) under contract no. DE-AC52-07NA27344 (Contract 44) between the U.S. Department of Energy (DOE) and Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.  See license for disclaimers, notice of U.S. Government Rights and license terms and conditions.

import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# Classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Pickling
import pickle

# Data formats
import json

def intersection(list_a, list_b):
    return [ elem for elem in list_a if elem in list_b ]

def train_for_group( df: pd.DataFrame, group_label: str, n_estimators: int = 1000, max_depth: int = 8, path_to_save: str = 'Classifiers'):
    """
    Automatically trains Random Forest binary classifiers for all Class (Secondary) and Sub-Class (Tertiary) labels under the specified Group label.
    Saves each trained classifier as a pickle file in the given path.
    Parameters:
        df (pd.DataFrame): DataFrame containing features, labels, and identifying columns.
        group_label (str): The Group label to filter and train classifiers for.
        n_estimators (int): Number of trees in each Random Forest.
        max_depth (int): Maximum depth of each tree.
        path_to_save (str): Directory to save pickled classifiers.
    """

    # Filter the DataFrame to only include rows with the specified primary group label.
    # Drop any rows with missing values and reset the index for clean slicing.
    df1 = df.loc[df['primary'] == group_label].copy().dropna().reset_index()

    # Get unique values of Class (secondary) and Sub-Class (tertiary) labels, sorted by their frequency.
    arr_class = np.asanyarray(
        df1.groupby(['secondary']).nunique().sort_values(by=['index'], ascending=False).index
    )
    arr_subclass = np.asanyarray(
        df1.groupby(['tertiary']).nunique().sort_values(by=['index'], ascending=False).index
    )

    # Proceed only if there are rows to train on.
    if df1.shape[0] != 0:
        # Select feature columns (assumes first five columns are identifiers/labels).
        features = df1.iloc[:, -(df1.shape[1] - 5):]
        # Train binary classifiers for each Class (secondary label).
        label = df1['secondary']
        # Split data for training (95%) and discard test split (not used here).
        x_train, _, y_train, _ = train_test_split(features, label, test_size=0.05, random_state=42)
        for evttag in arr_class:
            print("Training classifier for {} Class label...\n".format(evttag))
            # Create binary labels: True for current class, False otherwise.
            y_train_copy = y_train == evttag

            # Initialize and train Random Forest classifier.
            rf = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=42
            )
            rf.fit(x_train, y_train_copy.values)

            # Save the trained classifier as a pickle file.
            with open('{}/RF_{}.pickle'.format(path_to_save, evttag), 'wb') as handle:
                pickle.dump(rf, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Repeat process for Sub-Class (tertiary label) classifiers.
        label = df1['tertiary']
        x_train, _, y_train, _ = train_test_split(features, label, test_size=0.05, random_state=42)
        
        for evttag in arr_subclass:
            print("Training classifier for {} Sub-Class label...\n".format(evttag))
            # Create binary labels for each sub-class.
            y_train_copy = y_train == evttag

            # Initialize and train Random Forest classifier.
            rf = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=42
            )
            rf.fit(x_train, y_train_copy.values)

            # Save the trained sub-class classifier as a pickle file.
            with open('{}/RF_{}.pickle'.format(path_to_save, evttag), 'wb') as handle:
                pickle.dump(rf, handle, protocol=pickle.HIGHEST_PROTOCOL)

def successive_binary_classifier(test_features: pd.DataFrame, label_list: list, path_to_read: str = 'Classifiers') -> str:
    """
    Applies a set of binary classifiers to a single test sample and returns the label with the highest positive probability.
    Parameters:
        test_features (pd.DataFrame): Features for one test sample (single row, formatted as DataFrame).
        label_list (list): List of event tag labels, each with an associated binary classifier.
        path_to_read (str): Directory containing pickled classifiers named 'RF_{evttag}.pickle'.
    Returns:
        pred_label (str): The label with the highest positive probability.
        probStatic (np.array): Array of [label, probability] pairs for all tested labels.
    """
    proba_temp = 0           # Stores the highest probability found so far.
    pred_label = ''          # Stores the label corresponding to the highest probability.
    probStatic = []          # Collects [label, probability] pairs for all labels.

    # Iterate through each label and apply its binary classifier.
    for evttag in label_list:
        # Load the binary classifier for the current label.
        with open('{}/RF_{}.pickle'.format(path_to_read, evttag), 'rb') as handle:
            rf = pickle.load(handle)
            # Predict probabilities for the test sample.
            proba = rf.predict_proba(test_features)
            # If the positive class probability is higher than the current max, update.
            if proba[0][1] > proba_temp:
                probStatic.append([evttag, proba[0][1]])
                proba_temp = proba[0][1]
                pred_label = evttag

    # Convert the list of [label, probability] pairs to a numpy array for sorting.
    probStatic = np.array(probStatic)

    # Sort the probabilities in descending order to find the top three (or fewer) most likely labels.
    sorted_probas_indices = np.argsort(probStatic[:, 1].astype(float))[::-1]
    sorted_probas = probStatic[sorted_probas_indices]
    if sorted_probas.shape[0] > 3:
        sorted_probas = sorted_probas[:3, :]

    # Optionally print the sorted probabilities for inspection.
    # print(sorted_probas)

    # Return the label with the highest probability and the full probability array.
    return pred_label, probStatic

def validation(df: pd.DataFrame, group_label: str) -> tuple[pd.DataFrame, float, float, float, float]:
    """
    Runs hierarchical validation on the test dataset using a local classifier per node (LCN) approach.
    Predicts Class first, then uses the predicted Class to select and predict Sub-Class.
    Returns a DataFrame of actual vs. predicted values and key metrics.
    Parameters:
        df (pd.DataFrame): DataFrame containing features, labels, and identifiers.
        group_label (str): The Group label to filter data for validation.
    Returns:
        group_results (pd.DataFrame): DataFrame with actual and predicted Class and Sub-Class.
        acc (float): Overall accuracy (%).
        hP (float): Hierarchical precision (%).
        hR (float): Hierarchical recall (%).
        hF (float): Hierarchical F-score.
    """

    # Filter DataFrame to only include rows for the specified group label, remove missing values, and reset index.
    df1 = df.loc[df['primary'] == group_label].copy().dropna().reset_index()

    # Get unique Class (secondary) labels, sorted by frequency.
    arr_class = np.asanyarray(
        df1.groupby(['secondary']).nunique().sort_values(by=['index'], ascending=False).index
    )

    # Proceed only if there is data to validate.
    if df1.shape[0] != 0:
        # Select feature columns (assumes first five columns are identifiers/labels).
        features = df1.iloc[:, -(df1.shape[1] - 5):]
        label = df1['secondary']
        # Split data to get test features (5% of data).
        _, x_test, _, _ = train_test_split(features, label, test_size=0.05, random_state=42)

    # Initialize lists to store results for each test sample.
    sigid = []
    group = []
    actual_class = []
    actual_subclass = []
    pred_class = []
    pred_subclass = []

    # Iterate through each test sample.
    for idx in x_test.index:
        print("Testing for Sig ID {}".format(df1.iloc[idx]["sigid"]))
        test_features = x_test[x_test.index == idx]  # Extract features for current sample.

        # Store identifiers and actual labels.
        sigid.append(df1.iloc[idx]['sigid'])
        group.append(group_label)
        actual_class.append(df1.iloc[idx]['secondary'])
        actual_subclass.append(df1.iloc[idx]['tertiary'])

        # Predict Class using binary classifiers.
        pred_class_1, static = successive_binary_classifier(test_features, arr_class)
        pred_class.append(pred_class_1)

        # Predict Sub-Class using classifiers for the predicted Class.
        pred_subclass_1, static = successive_binary_classifier(test_features, sec_to_tert_dict[pred_class_1])
        pred_subclass.append(pred_subclass_1)
    
    # Create DataFrame to store results for all test samples.
    group_results = pd.DataFrame(columns=[
        'Sig_ID', 'Group', 'Actual_Class', 'Predicted_Class', 'Actual_Subclass', 'Predicted_Subclass'
    ])
    group_results['Sig_ID'] = sigid
    group_results['Group'] = group
    group_results['Actual_Class'] = actual_class
    group_results['Predicted_Class'] = pred_class
    group_results['Actual_Subclass'] = actual_subclass
    group_results['Predicted_Subclass'] = pred_subclass

    # Calculate accuracy as the percentage of samples where Sub-Class prediction matches actual.
    correct_classification = group_results.loc[
        group_results['Actual_Subclass'] == group_results['Predicted_Subclass']
    ].shape[0]
    acc = correct_classification / group_results.shape[0] * 100

    # Initialize counters for hierarchical precision and recall calculations.
    hP_num = 0
    hP_denom = 0
    hR_num = 0
    hR_denom = 0
    beta = 1  # Beta parameter for F-score.

    # Calculate hierarchical metrics for each sample.
    for row in range(group_results.shape[0]):
        C_i = []          # Actual label set for sample.
        C_i_hat = []      # Actual hierarchical label set (Class + Sub-Class).
        C_i_prime = []    # Predicted label set for sample.
        C_i_prime_hat = []# Predicted hierarchical label set (Class + Sub-Class).

        # Actual Sub-Class.
        C_i.append(group_results['Actual_Subclass'][row])
        
        # Actual Class and Sub-Class.
        C_i_hat.append(group_results['Actual_Class'][row])
        C_i_hat.append(group_results['Actual_Subclass'][row])

        # Predicted Sub-Class.
        C_i_prime.append(group_results['Predicted_Subclass'][row])

        # Predicted Class and Sub-Class.
        C_i_prime_hat.append(group_results['Predicted_Class'][row])
        C_i_prime_hat.append(group_results['Predicted_Subclass'][row])

        # Update hierarchical precision and recall numerators and denominators.
        hP_num += len(intersection(C_i_hat, C_i_prime_hat))
        hP_denom += len(C_i_prime_hat)
        hR_num += len(intersection(C_i_hat, C_i_prime_hat))
        hR_denom += len(C_i_hat)

    # Calculate hierarchical precision, recall, and F-score.
    hP = hP_num / hP_denom * 100
    hR = hR_num / hR_denom * 100
    hF = ((beta ** 2 + 1) * hP * hR) / (beta**2 * hP + hR)

    # Return results DataFrame and metrics.
    return group_results, acc, hP, hR, hF

def main():
    """
    Main function that runs all training and testing automatically. 
    """
    # Define the group labels for classification tasks.
    group_labels = ['Events', 'State', 'Equipment', 'Conditions', 'Phase']
    
    # If training flag is set, retrain classifiers for each group label.
    if flag_train_function:
        # Remove prefix "P001." from DataFrame column names for consistency.
        rename_headers = [s.replace("P001.", "") for s in df.columns.tolist()]
        df.columns = rename_headers
        
        # Train classifiers for each group label.
        for group_label in group_labels:
            train_for_group(df, group_label)

    # Initialize lists to store metrics for each group label.
    acc = []  # Accuracy
    hP = []   # Hierarchical Precision
    hR = []   # Hierarchical Recall
    hF = []   # Hierarchical F1 Score

    # Run validation/testing for the first group label.
    print("Testing for {} Group label...\n".format(group_labels[0]))
    group_results, acc_group, hP_group, hR_group, hF_group = validation(df, group_labels[0])
    acc.append(acc_group)
    hP.append(hP_group)
    hR.append(hR_group)
    hF.append(hF_group)

    # Run validation/testing for the remaining group labels.
    for group_label in group_labels:
        print("Testing for {} Group label...\n".format(group_label))
        if group_label != group_labels[0]:  # Skip first label, already processed above.
            group_results_add, acc_group, hP_group, hR_group, hF_group = validation(df, group_label)
            acc.append(acc_group)
            hP.append(hP_group)
            hR.append(hR_group)
            hF.append(hF_group)
            
            # Concatenate results for all group labels.
            group_results = pd.concat([group_results, group_results_add], ignore_index=True)

    # Save detailed results for all group labels to CSV.
    print("Saving per Group label results...")
    group_results.to_csv('Outputs/lcn_rf_test_results.csv', index=False)

    # Create a DataFrame to store overall metrics for each group label.
    metrics_results = pd.DataFrame(columns=['Group', 'Accuracy', 'hP', 'hR', 'hF'])
    metrics_results['Group'] = group_labels
    metrics_results['Accuracy'] = acc
    metrics_results['hP'] = hP
    metrics_results['hR'] = hR
    metrics_results['hF'] = hF

    # Save overall metrics to CSV.
    print("Saving overall results...")
    metrics_results.to_csv('Outputs/lcn_rf_metrics_results.csv', index=False)

if __name__ == "__main__":
    # Re-training classifiers to match with latest version of sklearn started at 5/30 10:46am.
    # Load the input DataFrame with features for Providers 9 and 10, separated by sensor.
    df = pd.read_csv('Inputs/pmu_transmission_features_header_rename.csv')

    # Load mapping dictionary for hierarchical classification (secondary to tertiary labels).
    sec_to_tert_dict = json.load(open('Inputs/sec_to_tert_dict_transmission.json'))
    
    # Flag to indicate whether training should be performed.
    flag_train_function = 1
    
    # Run the main function to start training and testing pipeline.
    main()

# Copyright (c) 2024, Lawrence Livermore National Security, LLC. All rights reserved. LLNL-CODE-868565

# EXPORT CONTROLLED INFORMATION

# This work was produced at the Lawrence Livermore National Laboratory (LLNL) under contract no. DE-AC52-07NA27344 (Contract 44) between the U.S. Department of Energy (DOE) and Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.  See license for disclaimers, notice of U.S. Government Rights and license terms and conditions.