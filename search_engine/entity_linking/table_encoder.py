import pandas as pd
from .dbpedia_lookup_api_call import dbpedia_lookup

def format_data_structure(df, doc_id, title, description, caption, entity_link_positions):
    data_structure = {
        'doc_id': doc_id,
        'title': title,
        'description': description,
        'caption': caption,
        'column_headers': df.columns.tolist(),
        'records': [],
        'lookup_results': [],
        'menu': [],
        'dbpedia_results_index': []
    }
    lookup_index = 0
    for index, row in df.iterrows():
        record = []
        for col_index, entry in enumerate(row):
            if [index, col_index] in entity_link_positions:
                data_structure['menu'].append(lookup_index)
                data_structure['dbpedia_results_index'].append([lookup_index])
                position_encoding = [index, col_index]
                lookup_results = dbpedia_lookup(entry)
                for result in lookup_results:
                    data_structure['lookup_results'].append(result)
                    data_structure['dbpedia_results_index'][-1].append(lookup_index)
                    lookup_index += 1
                record = [position_encoding, entry]
                print(entry)
                data_structure['records'].append(record)
    return_list = [data_structure['doc_id'],
                   data_structure['title'],
                   data_structure['description'],
                   data_structure['caption'],
                   data_structure['column_headers'],
                   data_structure['records'],
                   data_structure['lookup_results'],
                   data_structure['menu'],
                   data_structure['dbpedia_results_index']]
    return return_list


