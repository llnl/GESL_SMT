# Copyright (c) 2025, Lawrence Livermore National Security, LLC. All rights reserved. LLNL-CODE-2003701
# EXPORT CONTROLLED INFORMATION
# This work was produced at the Lawrence Livermore National Laboratory (LLNL) under contract no. DE-AC52-07NA27344 (Contract 44) between the U.S. Department of Energy (DOE) and Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.  See license for disclaimers, notice of U.S. Government Rights and license terms and conditions.

import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import dash
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output
import webbrowser

# Classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Pickling
import pickle

# Data formats
import json

# User defined functions
import Functions as fn

# Disable Pandas Warnings for Future
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)  

def successive_binary_classifier(test_features: pd.DataFrame, label_list: list, path_to_read: str = 'Classifiers') -> str:
    """
    Function that inputs the features of a single test sample to the binary classifiers associated with each label in the input label_list.
    :param test_features: The features for a single test sample.
    :param label_list: Each label in the label_list should have an associated binary classifier and test_features will be inputted to each of these binary classifiers. 
    :param path_to_read: Specifies the directory to find the pickled classifier. Right now, this assumes that the path_to_read is in the directory `Classifiers/Transmission_PMU` by default and that there is a file or files named `RF_{evttag}` where evttag is the event tag label. 
    :return: The predicted class after getting the maximum positive probabilities outputted by each binary classifier representing each label in label_list. 
    """
    # Initialize DataFrames to store predictions and feature importances
    df_prediction = pd.DataFrame(columns=["label", "raw_prob"])
    df_importance = pd.DataFrame(columns=["label", "feature", "importance"])
    
    # Iterate through each label and apply its associated classifier
    for evttag in label_list:
        # Load the pickled classifier for the current label
        with open('{}/RF_{}.pickle'.format(path_to_read, evttag), 'rb') as handle:
            rf = pickle.load(handle)
            # Get probability predictions for the test sample
            proba = rf.predict_proba(test_features)
            # Store the positive class probability
            df_prediction.loc[len(df_prediction)] = [evttag, proba[0][1]]
            
            # Retrieve and store feature importances
            importances = rf.feature_importances_
            feature_names = test_features.columns.tolist()
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False).reset_index(drop=True)
            importance_df['label'] = evttag
            df_importance = pd.concat([df_importance, importance_df], ignore_index=True)
    
    # Normalize probabilities to sum to 1 (adjusted probabilities)
    df_prediction['adj_prob'] = df_prediction['raw_prob'] / df_prediction['raw_prob'].sum()
    # Filter out labels with very low probability (< 0.05)
    df_prediction = df_prediction[df_prediction['adj_prob'] >= 0.05]
    # Renormalize after filtering
    df_prediction['adj_prob'] = df_prediction['adj_prob'] / df_prediction['adj_prob'].sum()
    # Limit to top 3 labels if more than 3 remain
    if df_prediction.shape[0] > 3:
        df_prediction = df_prediction.nlargest(3, 'adj_prob').reset_index(drop=True)
    # Filter feature importances to only include top predicted labels
    df_importance = df_importance[df_importance['label'].isin(df_prediction['label'].tolist())]
    
    # Return predictions and feature importances
    return df_prediction, df_importance

def validation(df: pd.DataFrame, group_label: str) -> tuple[pd.DataFrame, float, float, float, float]:
    """
    Function that runs all the signatures in the testing dataset in a way that is inherently hierarchical. In particular, this function will follow the local classifier per node (LCN) approach, assuming that the Class label that is first predicted is correct, i.e. the predicted Class label will be used to predict the Sub-Class label regardless of correctness of Class label.
    :param df: The DataFrame that contains features, labels, and other identifying columns such as Signature ID.
    :param group_label: A string that defines which Group label to focus on.
    :return: A DataFrame that contains all the actual versus predicted Class and Sub-Class.
    """
    # Dictionary to collect feature importances for each label and subclass
    importance_indices = {}
    
    # Prepare test data: drop missing values and reset index
    df1 = df.copy().dropna().reset_index()
    
    # Get array of primary class labels for the specified group
    arr_class = np.asanyarray(pri_to_sec_dict[group_label])
    
    # Select feature columns for testing
    x_test = df1.iloc[:, 2:]
    
    # Initialize DataFrame to store subclass prediction results
    df_pred_subclass_result = None
    
    # Loop through each test sample
    for idx in x_test.index:
        print("Testing for Sig ID {}".format(df1.iloc[idx]["sigid"]))
        # Extract features for the current sample
        test_features = x_test[x_test.index == idx]
        
        # Predict class label and get feature importances
        df_pred_class_result, df_pred_class_impt_analysis = successive_binary_classifier(test_features, arr_class)
        
        # Store feature importances for each predicted class label
        for col in df_pred_class_impt_analysis['label'].unique():
            if col not in importance_indices:
                importance_indices[col] = {}
                importance_indices[col]['instance'] = {}
            for ind, row in df_pred_class_impt_analysis[df_pred_class_impt_analysis['label'] == col].iterrows():
                importance_indices[col]['instance'][row['feature']] = row['importance']
        
        # For each predicted class label, predict subclass label(s)
        for label in df_pred_class_result['label']:
            pred_subclass_1, pred_subclass_impt_analysis = successive_binary_classifier(test_features, sec_to_tert_dict[label])
            
            # Store feature importances for each subclass label
            for col in pred_subclass_impt_analysis['label'].unique():
                importance_indices[label][col] = {}
                for ind, row in pred_subclass_impt_analysis[pred_subclass_impt_analysis['label'] == col].iterrows():
                    importance_indices[label][col][row['feature']] = row['importance']
            
            # Format subclass prediction results
            pred_subclass_1.rename(columns={'label': 'tert_label'}, inplace=True)
            pred_subclass_1['sec_label'] = label
            pred_subclass_1.insert(0, 'sec_label', pred_subclass_1.pop('sec_label'))
            
            # Concatenate results for all samples
            if df_pred_subclass_result is None:
                df_pred_subclass_result = pred_subclass_1
            else:
                df_pred_subclass_result = pd.concat([df_pred_subclass_result, pred_subclass_1], axis=0)
    
    # Return class predictions, subclass predictions, and feature importances
    return df_pred_class_result, df_pred_subclass_result, importance_indices

def vis_result(pri_label, sec_res, tert_res):
    """
    Generates a nested pie chart to visualize the likelihood distribution of primary, secondary, and tertiary labels.
    :pri_label: The primary label whose likelihood distribution is being visualized.
    :sec_res: DataFrame containing secondary labels and their adjusted probabilities (adj_prob).
    :tert_res: DataFrame containing tertiary labels, their associated secondary labels, and adjusted probabilities (adj_prob).
    """
    # Reset index for secondary results DataFrame
    sec_res = sec_res.reset_index(drop=True)
    
    # Initialize lists for pie chart data and legend formatting
    valsArr = []
    inner_sizes = []
    inner_labels = []
    inner_labels_on_fig = []
    inner_labels_each_tert_group = []
    inner_color_arr = []
    legend_labels_reodered = []
    legend_index_reodered = []
    
    # Normalize secondary probabilities
    sec_prob = [x / sum(sec_res['adj_prob'].tolist()) for x in sec_res['adj_prob'].tolist()]
    outer_sizes = [round(x, 3) for x in sec_prob]
    outer_labels = sec_res['label'].tolist()
    
    # Format outer labels with percentage
    for ind, label in enumerate(outer_labels):
        outer_labels[ind] = label + ': ' + str(round(sec_prob[ind]*100,1)) + '%'
    outer_labels_on_fig = outer_labels
    
    # Sort secondary labels by probability for display order
    sorted_prob_sec_label_index = [i for i, _ in sorted(enumerate(sec_prob), key=lambda x: x[1], reverse=True)]
    outer_labels = [outer_labels[i] for i in sorted_prob_sec_label_index]
    
    # Loop through each secondary label to prepare tertiary data
    for indSecLayer, rowSec in sec_res.iterrows():
        df_slc_sec_label = tert_res[tert_res['sec_label'] == rowSec['label']]
        tert_prob = [round(x, 4) for x in df_slc_sec_label['adj_prob'].tolist()]
        prob_for_label = [x * 100 for x in tert_prob]
        tert_labels = []
        for ind, label in enumerate(df_slc_sec_label['tert_label']):
            inner_labels.append(label + ': ' + str(round(prob_for_label[ind],1)) + '%')
            tert_labels.append(label + ': ' + str(round(prob_for_label[ind],1)) + '%')
            inner_labels_on_fig.append(str(round(prob_for_label[ind],1)) + '%')
        # Sort tertiary labels by probability
        sorted_prob_tert_label_index = [i for i, _ in sorted(enumerate(prob_for_label), key=lambda x: x[1], reverse=True)]
        ordered_tert_labels = [tert_labels[i] for i in sorted_prob_tert_label_index]
        inner_labels_each_tert_group.append(ordered_tert_labels)
        # Calculate inner pie sizes proportional to secondary probability
        inner_sizes.extend([x * outer_sizes[indSecLayer] / sum(tert_prob) for x in tert_prob])
        inner_color_arr.extend(list(range(indSecLayer*4 + 1, indSecLayer*4 + 1 + len(tert_prob))))
    
    # Reorder inner labels to match sorted secondary labels
    inner_labels_each_tert_group = [inner_labels_each_tert_group[i] for i in sorted_prob_sec_label_index]
    for idx, sec_label in enumerate(outer_labels):
        legend_labels_reodered.append(sec_label)
        legend_labels_reodered.extend(inner_labels_each_tert_group[idx])
    
    legend_labels = outer_labels + inner_labels
    for label in legend_labels_reodered:
        legend_index_reodered.append(legend_labels.index(label))
    
    # Create figure and axes for plotting
    fig, ax = plt.subplots(figsize=(16,9))
    cmap = plt.colormaps["tab20c"]
    outer_colors = cmap(np.arange(len(outer_sizes))*4)
    inner_colors = cmap(inner_color_arr)
    
    # Handle specific data type formatting
    if data_type == 'PMU' and pri_label == 'Phase':
        legend_labels = outer_labels
        legend_index_reodered = [x for x in legend_index_reodered if x < len(legend_labels)]
        outer_pi_width = 0.8
    else:
        # Plot inner pie chart for tertiary labels
        inner_pie = ax.pie(inner_sizes, 
                        radius = 0.8,
                        labels = inner_labels_on_fig,
                        labeldistance = 0.5, 
                        colors = inner_colors, 
                        wedgeprops=dict(width = 0.5, edgecolor='w'),
                        textprops = {'fontsize': 8})
        legend_labels = outer_labels + ["  -- " + item for item in inner_labels]
        outer_pi_width = 0.2

    # Plot outer pie chart for secondary labels
    outer_pie = ax.pie(outer_sizes, 
                    radius = 1, 
                    labels = outer_labels_on_fig, 
                    labeldistance = 1.1, 
                    colors = outer_colors, 
                    wedgeprops = dict(width=outer_pi_width, edgecolor='w'), 
                    textprops = {'fontsize': 14, 'weight': 'bold'})
        
    # Prepare legend handles and reorder them
    handles, labels = ax.get_legend_handles_labels() 
    ordered_handles = [handles[i] for i in legend_index_reodered]
    ordered_labels = [legend_labels[i] for i in legend_index_reodered]

    # Map colors for legend items
    value_to_index = {value: idx for idx, value in enumerate(outer_labels_on_fig)}
    mapping = [value_to_index[val] for val in outer_labels]
    outer_color_index = (np.arange(len(outer_sizes))*4).tolist()
    ordered_outer_color_index = [outer_color_index[i] for i in mapping]
    color_code = []
    ct_sec_label = -1
    ct_tert_label = 1
    for tag in ordered_labels:
        if tag in outer_labels:
            ct_sec_label = ct_sec_label + 1
            color_code.append(ordered_outer_color_index[ct_sec_label])
            ct_tert_label = 1
        else:
            color_code.append(ordered_outer_color_index[ct_sec_label] + ct_tert_label)
            ct_tert_label = ct_tert_label + 1
    ordered_colors = cmap(color_code)

    # Create legend markers for each label
    test_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=ordered_colors[i],
                   markersize=10)
        for i in range(len(ordered_labels))
    ]

    # Add legend to plot
    ax.legend(test_handles, 
              ordered_labels, 
              loc="upper left", 
              bbox_to_anchor = (1, 0, 0.5, 1))   
    ax.set(aspect="equal")
    plt.title('Likelihood for the Primary Label: ' + pri_label, 
              fontweight='bold', 
              fontsize=20) 
    return ordered_labels

def vis_table(df_labels):
    primary_labels = ['Events', 'State', 'Equipment', 'Conditions', 'Phase']
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.Label("Select a primary label:"),
        dcc.RadioItems(
            id="primary-radio",
            options=[{"label": label, "value": label} for label in primary_labels],
            value=primary_labels[0],
            inline=True  # display horizontally
        ),
        dash_table.DataTable(
            id="table",
            columns=[{"name": i, "id": i} for i in df_labels.columns],
            data=df_labels.to_dict("records"),
            style_cell={'textAlign': 'left'}
        )
    ])

    # highlight the selected column
    @app.callback(
        Output("table", "style_data_conditional"),
        Input("primary-radio", "value")
    )
    def highlight_column(selected):
        return [
            {
                'if': {'column_id': selected},
                'backgroundColor': "#8CD812"
            }
        ]
    # Open browser automatically
    webbrowser.open("http://127.0.0.1:8050/")
    app.run(debug=False)

def format_cell(val):
    if isinstance(val, str) and not val.strip().startswith("--"):
        return f"<b>{val.strip()}</b>"
    return val.strip()

def vis_importance(data, pri_label):
    """
    Visualizes feature importance scores for secondary labels using a grouped bar chart.
    :data: Dictionary of feature importances for each secondary label.
    :pri_label: Primary label for context in the plot title.
    """
    scond_labels = list(data.keys())
    df_scond_scores = pd.DataFrame()
    # Build DataFrame from feature importance data
    for ind, scond_label in enumerate(scond_labels):
        df_scond_scores[scond_label] = data[scond_label]['instance']
    df_scond_scores = df_scond_scores.T.reset_index()
    first_col = df_scond_scores.columns[0]
    other_cols_sorted = sorted(df_scond_scores.columns[1:])
    df_scond_scores = df_scond_scores[[first_col] + other_cols_sorted]

    labels = df_scond_scores['index']
    x = np.arange(len(labels))  # Position for each secondary label
    groups = df_scond_scores.columns[1:]   # Feature names
    width = 0.8 / len(groups)   # Adjust bar width based on number of features

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('tab20', len(groups))
    # Plot grouped bars for each feature across secondary labels
    for i, group in enumerate(groups):
        offset = (i - len(groups)/2) * width + width/2
        color = cmap(i)
        ax.bar(x + offset, df_scond_scores[group], width, label=group, color=color)

    # Set axis labels and legend
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Values")
    ax.set_title("Importance Scores for Secondary Labels under: " + pri_label.upper())
    ax.legend(ncol=2, loc='upper right')
    plt.tight_layout()


def create_input_dataframe(raw_data):
    """
    Processes input time-series PMU data into a statistical feature DataFrame for ML models.
    Steps:
    1. Normalize the data (excluding 'Time' column).
    2. Calculate mean, std, skew, and kurtosis for each measurement.
    3. Return a formatted DataFrame suitable for signature matching.
    """
    sta_data = []         # List to store computed statistics
    sta_headers = []      # List to store feature names
    scaler = StandardScaler()  # Initialize standard scaler for normalization

    # Drop 'Time' column and standardize the data
    data_values = raw_data.drop(columns=['Time']).values
    data_values_standardized = scaler.fit_transform(data_values)
    df_standardized = pd.DataFrame(data_values_standardized, columns=raw_data.drop(columns=['Time']).columns)

    # Compute statistics for each column (measurement)
    for col in df_standardized.columns:
        if col != 'Time':
            sta_data.extend([
                df_standardized[col].mean(),
                df_standardized[col].std(),
                df_standardized[col].skew(),
                df_standardized[col].kurtosis()
            ])
            sta_headers.extend([
                col + '_mean', 
                col + '_std',
                col + '_skew',
                col + '_kurtosis'
            ])

    # Create single-row DataFrame with statistics
    df = pd.DataFrame([pd.Series(sta_data).values], columns=sta_headers)
    df.insert(0, 'sigid', ['1'])  # Insert signature ID as first column

    # Prepare feature names for columns
    headers_feature_names = [s.split('.')[1] for s in df.columns.tolist()[1:]][0:24]
    headers_feature_names = ['sigid'] + headers_feature_names

    # Reshape DataFrame if multiple PMUs are present
    if int(nPMU) > 1:
        headers_pmu_names = [s.split('.')[0] for s in df.columns.tolist()[1:]]
        pmu_names = sorted(list(set(headers_pmu_names)))
        df_vals = df.iloc[0, 1:].values
        df_reshape = np.reshape(df_vals, (-1, int(len(sta_headers) / nPMU)))
        rows = []
        for i, row in enumerate(df_reshape):
            new_id = pmu_names[i]  # Assign PMU name as row identifier
            rows.append([new_id] + list(row))
        df = pd.DataFrame(rows, columns=headers_feature_names)
    else:
        df.columns = headers_feature_names  # Assign feature names for single PMU

    return df  # Return processed DataFrame



def main():
    """
    Main function that runs the signature-matching process and renders results.

    Workflow:
      1) Build a per-PMU dataframe from the raw input (df_input).
      2) For each PMU row and each primary label (group), run validation to get:
           - predicted class
           - predicted subclass
           - (optionally) feature importance scores
      3) Visualize the predicted results (and optionally importance).
      4) Show all plots at the end.
    """
    # Primary label groups used for hierarchical classification
    group_labels = ['Events', 'State', 'Equipment', 'Conditions', 'Phase']

    # Transform the raw input table into a dataframe where each row corresponds to one PMU instance
    df_all_pmu = create_input_dataframe(df_input)
    vis_table_labels = []
    for ind, row in df_all_pmu.iterrows():
        # Convert the Series 'row' back to a 1-row DataFrame for downstream functions that expect DataFrame input
        df = pd.DataFrame(row).T

        # For each primary group, run the validator and visualize the prediction
        for pri_label in group_labels:
            # Run model/validator: returns predicted class, predicted subclass, and optional importance scores
            pred_class, pred_subclass, importance_scores = validation(df, pri_label)
            chart_labels = vis_result(pri_label,pred_class,pred_subclass)
            vis_table_labels.append(chart_labels)
            # vis_importance(importance_scores,pri_label)
    df_vis_table = pd.DataFrame.from_records(vis_table_labels).T
    df_vis_table.columns = [f"Column_{i+1}" for i in range(len(df_vis_table.columns))]
    df_vis_table.columns = group_labels
    vis_table(df_vis_table)
    plt.show()


if __name__ == "__main__":

    # Provide PMU waveform data (in .csv format) from GESL (Provider 9-10), e.g., sigID-978.csv
    df_input = pd.read_csv('Inputs/sigId-978.csv')
    data_type = 'PMU'
    nFeatures = 6
    nPMU = ((df_input.shape[1]) - 1) / nFeatures
    if nPMU % 1 != 0:
        print('ERROR: Please check the number of features in the PMU data. '
              'Each PMU should have exactly SIX features in the columns.')
        exit(1)
    else:
        print('The input data contains a total of {} PMU entries.'.format(int(nPMU)))
    
    # Input a dictionary to map:
    #   (1) primary labels to secondary labels and 
    #   (2) secondary labels to tertiary labels, 
    # to be used during hierarchical classification.
    pri_to_sec_dict = json.load(open('Inputs/pri_to_sec_dict_transmission.json'))
    sec_to_tert_dict = json.load(open('Inputs/sec_to_tert_dict_transmission.json'))

    main()
