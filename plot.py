#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_experiments_from_multiple_json(json_paths, data_labels=None, output_fig=None):
    """
    Reads multiple JSON files (each with keys "augplan", "accuracy", "time_taken"),
    collects their accuracy sequences into a list, and then plots the curves.
    
    Parameters:
        json_paths (list): List of paths to JSON files.
        data_labels (list, optional): List of labels for the experiments. If not provided
                                      or if the number of labels does not match the number
                                      of experiments, default labels ("Exp 1", "Exp 2", …)
                                      will be used.
        output_fig (str, optional): Output figure filename. If not provided and only one JSON
                                    file is passed, the output figure will have the same base name
                                    as the JSON file with a .png extension. For multiple JSON files,
                                    the filename is created by concatenating their base names.
    
    Returns:
        DataFrame: A pandas DataFrame containing the accuracy data per iteration.
    """
    experiments = []
    
    # Load each JSON file and add its data (a dict or list) to the experiments list.
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                experiments.extend(data)
            else:
                experiments.append(data)
    
    n_experiments = len(experiments)
    
    # Assign default labels if none provided or if the count doesn't match.
    if (data_labels is None) or (len(data_labels) != n_experiments):
        data_labels = [f"Exp {i+1}" for i in range(n_experiments)]
    
    # Extract each experiment's "accuracy" list.
    accuracy_lists = [exp.get("accuracy", []) for exp in experiments]
    
    # Determine the maximum length among all accuracy sequences.
    max_len = max(len(acc) for acc in accuracy_lists)
    
    # Pad each accuracy list with np.nan so that they all have the same length.
    padded_accuracies = [acc + [np.nan] * (max_len - len(acc)) for acc in accuracy_lists]
    
    # Create a DataFrame with iterations and each experiment's accuracy.
    iterations = list(range(1, max_len + 1))
    df = pd.DataFrame({"Iteration": iterations})
    for label, acc in zip(data_labels, padded_accuracies):
        df[label] = acc
    
    # Determine the output figure filename.
    if output_fig is None:
        if n_experiments == 1:
            base, _ = os.path.splitext(os.path.basename(json_paths[0]))
            output_fig = "results/" + base + ".png"
        else:
            # Concatenate the base names of all JSON files (without extensions) with underscores.
            bases = [os.path.splitext(os.path.basename(path))[0] for path in json_paths]
            output_fig = "results/" + "_".join(bases) + ".png"
    
    # Plot the accuracy curves.
    plt.figure(figsize=(8, 6))
    for label in data_labels:
        plt.plot(df["Iteration"], df[label], marker="o", label=label)
    
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Experiment Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_fig)
    plt.show()
    
    return df

if __name__ == "__main__":
    # Example list of JSON file paths.
    json_files = [
        os.path.join("results", "run_country_alite_exp.json"),
        os.path.join("results", "run_country_exp.json")
    ]
    
    # Provide data labels if desired; otherwise, default labels will be used.
    data_labels = ["alite", "kitana"]
    
    # Call the function to plot the experiments.
    df_results = plot_experiments_from_multiple_json(json_files, data_labels=data_labels)
    
    # Optionally, print the DataFrame to inspect the data.
    print(df_results)
