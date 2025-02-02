import pandas as pd
import numpy as np
import os
from ..entity_linking.el_test import process_csv

def link_dbpedia(percentage=100, target_column=["country"], csv_dir="data/country/buyer/buyer_gini.csv", output_dir="el_data/country/buyer/100/buyer_gini.json", meta_data=None):
    if meta_data is None:
        meta_data = {
            'Document_ID': 'DOC123',
            'Document Title': 'Country Buyer GINI Analysis',
            'Document Description': 'Analyzing GINI index by country for buyers.',
            'Document Caption': 'GINI Index Data'
        }

    df = pd.read_csv(csv_dir)

    for col in target_column:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in the data.")

    entity_link_positions = []
    for col in target_column:
        non_null_indices = df[col].dropna().index
        num_to_select = int(len(non_null_indices) * (percentage / 100))
        selected_indices = np.random.choice(non_null_indices, num_to_select, replace=False)
        col_index = df.columns.get_loc(col)
        entity_link_positions.extend([[index, col_index] for index in selected_indices])
    print("entity_link_positions", entity_link_positions)
    process_csv(csv_dir, meta_data, output_dir, auto_el=False, entity_link_positions=entity_link_positions)
    print(f"Data linked and saved to {output_dir}")
