import pandas as pd
import numpy as np
import os
import json
import subprocess
from .dbpedia_lookup_api_call import dbpedia_lookup
import chardet


class DBpediaLinker:
    def __init__(self, output_dir_base="el_data", auto_load_meta_data=False):
        """
        Initialize the DBpediaLinker class.

        Parameters:
            output_dir_base (str): Base directory for output files.
            auto_load_meta_data (bool): Whether to automatically load metadata.
        """
        self.output_dir_base = output_dir_base
        self.auto_load_meta_data = auto_load_meta_data
        self.structured_data = []
        self.selected_input = {}

    def batch_link(self, input_data, percentage_list, meta_data_dict=None):
        """
        Batch process multiple CSV files and perform entity linking.

        Parameters:
            input_data (dict): dictionary where the key is the CSV path and the value is a list of columns to process.
            percentage_list (list): list specifying the percentage of entity linking for each CSV file.
            meta_data_dict (dict): dictionary containing metadata for each CSV file.

        Returns:
            None
        """
        if len(input_data) != len(percentage_list):
            raise ValueError("The length of input_data and percentage_list must match.")

        if not self.auto_load_meta_data and (meta_data_dict is None or len(input_data) != len(meta_data_dict)):
            raise ValueError("The length of input_data and meta_data_dict must match when auto_load_meta_data is False.")

        for i, (csv_dir, target_columns) in enumerate(input_data.items()):
            try:
                percentage = percentage_list[i]

                if self.auto_load_meta_data:
                    meta_data = self.meta_data_extractor(csv_dir)
                else:
                    if csv_dir not in meta_data_dict:
                        raise ValueError(f"Meta data not found for file: {csv_dir}")
                    meta_data = meta_data_dict[csv_dir]

                output_dir = self.output_dir_base

                with open(csv_dir, 'rb') as file:
                    result = chardet.detect(file.read(10000))

                df = pd.read_csv(csv_dir, encoding=result['encoding'])

                for col in target_columns:
                    if col not in df.columns:
                        raise ValueError(f"Column {col} not found in the data for file: {csv_dir}")

                entity_link_positions = []
                for col in target_columns:
                    # Drop NA
                    non_null_series = df[col].dropna()
                    unique_entities = non_null_series.unique()
                    col_index = df.columns.get_loc(col)

                    # Calculate the numebr of entities to choose
                    num_entities_to_select = int(len(unique_entities) * (percentage / 100.0))
                    if num_entities_to_select <= 0:
                        # If percentage is too little
                        print(f"Warning: For column '{col}', the computed number of entities to link is 0. Skipping.")
                        continue

                    # Random pick
                    selected_entities = np.random.choice(unique_entities, size=num_entities_to_select, replace=False)

                    # Pick the selected entities in the original df
                    for ent in selected_entities:
                        print("ent picked: ", ent)
                        row_indices = df.index[df[col] == ent].tolist()
                        if len(row_indices) > 0:
                            # pick the first is enough
                            r_idx = row_indices[0]  
                            # r_idx = np.random.choice(row_indices, 1)[0] # or we can randomly pick one
                            entity_link_positions.append([r_idx, col_index])

                print(f"Processing file: {csv_dir}")
                print("Entity link positions:", entity_link_positions)

                structured_data_single = self.process_csv(csv_dir=csv_dir, meta_data=meta_data, output_dir=output_dir, auto_el=False, to_disk=False, entity_link_positions=entity_link_positions)
                print(f"Data linked and saved to {output_dir}")
                self.structured_data.append(structured_data_single)
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # store the single structured data to the output_dir with the name of the csv file + structured_data_single.json
                output_file_path = os.path.join(output_dir, f"{os.path.basename(csv_dir).split('.')[0]}_structured_data_single.json")
                with open(output_file_path, 'w') as outfile:
                    json.dump([structured_data_single], outfile, indent=4)
                self.selected_input[csv_dir] = target_columns
            except Exception as e:
                print(f"Error processing file: {csv_dir}. Skipping. Details: {e}")
                continue
                
        

        json_path = os.path.join(output_dir, 'structured_data.json')
        with open(json_path, 'w') as json_file:
            json.dump(self.structured_data, json_file, indent=4)

        print(f"Data saved to {json_path}")
        output_file_path = os.path.join(self.output_dir_base, "selected_input.json")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(self.selected_input, outfile, indent=4)
        print(f"Selected input data saved to {output_file_path}")

    def meta_data_extractor(self, csv_dir):
        """
        Extract metadata from the meta_data.json file located in the same directory as the specified CSV file.

        Parameters:
            csv_dir (str): Path to the CSV file.

        Returns:
            dict: Metadata dictionary for the specified CSV file.
        """
        folder_path = os.path.dirname(csv_dir)
        csv_name = os.path.basename(csv_dir)

        meta_data_path = os.path.join(folder_path, "meta_data.json")

        if not os.path.exists(meta_data_path):
            raise FileNotFoundError(f"meta_data.json not found in the folder: {folder_path}")

        with open(meta_data_path, "r") as meta_file:
            meta_data = json.load(meta_file)

        if csv_name not in meta_data:
            raise ValueError(f"Metadata for {csv_name} not found in meta_data.json.")

        return meta_data[csv_name]

    def process_csv(self, csv_dir, meta_data, output_dir, auto_el=True, to_disk=False, entity_link_positions=None):
        """
        Process a single CSV file and perform entity linking.

        Parameters:
            csv_dir (str): Path to the CSV file.
            meta_data (dict): Metadata for the current file.
            output_dir (str): Directory to save the output.
            auto_el (bool): Whether to automatically generate entity link positions.
            entity_link_positions (list): Specified entity link positions.

        Returns:
            None
        """
        df = pd.read_csv(csv_dir)

        if auto_el:
            entity_link_positions = []
            for col in df.columns:
                if df[col].dtype == object and df[col].notna().any():
                    entity_link_positions.extend([[index, df.columns.get_loc(col)] for index in df.index if pd.notna(df.at[index, col])])

        if not auto_el and entity_link_positions is None:
            raise ValueError("Entity link positions must be provided if auto_el is set to False.")
        doc_id = meta_data['Document_ID']
        title = meta_data['Document_Title']
        description = meta_data['Document_Description']
        caption = meta_data['Document_Caption']

        structured_data = self.format_data_structure(df, doc_id, title, description, caption, entity_link_positions)

        if to_disk:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            json_path = os.path.join(output_dir, 'structured_data.json')
            with open(json_path, 'w') as json_file:
                json.dump(structured_data, json_file, indent=4)

            print(f"Data saved to {json_path}")

        return structured_data

    @staticmethod
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
                    data_structure['menu']]
                    # data_structure['dbpedia_results_index']]
        return return_list

    def load_structured_data(self, file_path):
        """
        Load structured data from a JSON file.
        """
        with open(file_path, 'r') as file:
            self.structured_data = json.load(file)
        print(f"Structured data loaded from {file_path}")

class TURLExecuter:
    def __init__(
        self, 
        tmp_dir="/home/ec2-user/Kitana_e2e/Kitana-e2e/tmp",
        input_data="/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country/buyer/2/structured_data.json", 
        host_input_dir="/TURL/data/kitana/country/buyer/2",
        container_working_dir="/TURL/", 
        output_path="/home/ec2-user/TURL/predictions.txt", 
        container_name="c3138415edf0"
    ):
        self.tmp_dir = tmp_dir
        self.input_data = input_data
        self.host_input_dir = host_input_dir
        self.container_working_dir = container_working_dir
        self.output_path = output_path
        self.container_name = container_name
        
    def prepare_input_file(self):
        """
        Prepare the input file for the container.

        This method directly copies the JSON content from self.input_data to the appropriate location.
        """

        input_file_path = os.path.join(self.tmp_dir, "dev.table_entity_linking.json")

        # Copy content from input_data to temporary path
        with open(self.input_data, "r") as src_file:
            with open(input_file_path, "w") as dest_file:
                dest_file.write(src_file.read())

        print(f"Input file temporarily saved at {input_file_path}")

        # Ensure the directory exists inside the container
        container_dir = self.host_input_dir
        cmd_create_dir = [
            "docker", "exec", self.container_name,
            "bash", "-c", f"mkdir -p {container_dir}"
        ]
        print(f"Ensuring container directory exists: {' '.join(cmd_create_dir)}")
        result = subprocess.run(cmd_create_dir, check=True, capture_output=True, text=True)
        print("Output from creating directory:", result.stdout)
        print("Error (if any):", result.stderr)

        # Copy the file into the container directory
        cmd_copy = [
            "docker", "cp", input_file_path,
            f"{self.container_name}:{os.path.join(self.host_input_dir, 'dev.table_entity_linking.json')}"
        ]
        print(f"Copying file into container with command: {' '.join(cmd_copy)}")
        result = subprocess.run(cmd_copy, check=True, capture_output=True, text=True)
        print("Output from copying file:", result.stdout)
        print("Error (if any):", result.stderr)

        # Check if file exists in the container
        cmd_check_file = [
            "docker", "exec", self.container_name,
            "bash", "-c", f"test -f {os.path.join(self.host_input_dir, 'dev.table_entity_linking.json')} && echo 'File exists' || echo 'File does not exist'"
        ]
        result = subprocess.run(cmd_check_file, capture_output=True, text=True)
        if result.stdout.strip() == 'File exists':
            print("File successfully copied into container and verified.")
        else:
            print("Failed to verify file presence in the container:", result.stdout)
            print("Error (if any):", result.stderr)

    def prepare_input_file_single(self, input_data):
        """
        Prepare the input file for the container.

        This method directly copies the JSON content from self.input_data to the appropriate location.
        """

        input_file_path = os.path.join(self.tmp_dir, "dev.table_entity_linking.json")

        # Copy content from input_data to temporary path
        with open(input_data, "r") as src_file:
            with open(input_file_path, "w") as dest_file:
                dest_file.write(src_file.read())

        print(f"Input file temporarily saved at {input_file_path}")

        # Ensure the directory exists inside the container
        container_dir = self.host_input_dir
        cmd_create_dir = [
            "docker", "exec", self.container_name,
            "bash", "-c", f"mkdir -p {container_dir}"
        ]
        print(f"Ensuring container directory exists: {' '.join(cmd_create_dir)}")
        result = subprocess.run(cmd_create_dir, check=True, capture_output=True, text=True)
        print("Output from creating directory:", result.stdout)
        print("Error (if any):", result.stderr)

        # Copy the file into the container directory
        cmd_copy = [
            "docker", "cp", input_file_path,
            f"{self.container_name}:{os.path.join(self.host_input_dir, 'dev.table_entity_linking.json')}"
        ]
        print(f"Copying file into container with command: {' '.join(cmd_copy)}")
        result = subprocess.run(cmd_copy, check=True, capture_output=True, text=True)
        print("Output from copying file:", result.stdout)
        print("Error (if any):", result.stderr)

        # Check if file exists in the container
        cmd_check_file = [
            "docker", "exec", self.container_name,
            "bash", "-c", f"test -f {os.path.join(self.host_input_dir, 'dev.table_entity_linking.json')} && echo 'File exists' || echo 'File does not exist'"
        ]
        result = subprocess.run(cmd_check_file, capture_output=True, text=True)
        if result.stdout.strip() == 'File exists':
            print("File successfully copied into container and verified.")
        else:
            print("Failed to verify file presence in the container:", result.stdout)
            print("Error (if any):", result.stderr)

    def run_container_evaluation_single(self):
        # get the json file ending with structured_data_single.json
        input_data = [f for f in os.listdir(os.path.dirname(self.input_data)) if f.endswith("structured_data_single.json")]
        predictions = []
        prediciton_list_all = []
        # Put the *_structured_data_single.json into the container and run the evaluation one at a time
        for data in input_data:
            prepared_input_data = os.path.join(os.path.dirname(self.input_data), data)
            self.prepare_input_file_single(prepared_input_data)
            self.run_container_evaluation()
            predictions.append(self.read_predictions())
            # get predictions list
            with open(os.path.join(os.path.dirname(self.output_path), "Predictions_list.txt"), 'r', encoding='utf-8') as f:
                prediction_list = json.load(f)
            prediciton_list_all.append(prediction_list)
        
        with open(os.path.join(os.path.dirname(self.input_data), "Predictions_list.txt"), 'w', encoding='utf-8') as f:
            json.dump(prediciton_list_all, f, ensure_ascii=False, indent=4)
        with open(os.path.join(os.path.dirname(self.input_data), "Predictions.txt"), 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)
        return predictions, prediciton_list_all


    def run_container_evaluation(self):
        """
        Run the container's evaluation script and monitor progress in real time, ensuring the Anaconda environment is activated.
        """
        cmd = [
            "docker", "exec", self.container_name,
            "bash", "-c",
            f"source /root/miniconda3/etc/profile.d/conda.sh && conda activate TURL_origin && cd /TURL/ && "
            f"CUDA_VISIBLE_DEVICES=0 python run_table_EL_finetuning.py "
            f"--data_dir={self.host_input_dir} "
            f"--output_dir=output/EL/v2/0/model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam "
            f"--model_name_or_path=output/EL/v2/0/model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam "
            f"--model_type=EL "
            f"--do_eval "
            f"--per_gpu_eval_batch_size=10 "
            f"--overwrite_output_dir "
            f"--config_name=configs/table-base-config_v2.json"
        ]


        print(f"Running command: {' '.join(cmd)}")
        
        # Stream the output in real time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            print(line, end="")  # Output logs in real time
        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            print(f"Error: Command exited with status {return_code}")
            for line in process.stderr:
                print(line, end="")
            raise subprocess.CalledProcessError(return_code, cmd)

        print("Container evaluation completed.")


    def read_predictions(self):
        """
        Read predictions from the container's output file.

        Parameters:
            output_dir (str): The directory containing the output file.

        Returns:
            list: A list of predictions read from the output file.
        """
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"Prediction file not found at {self.output_path}")
        
        predictions = []
        with open(self.output_path, "r") as f:
            for line in f:
                if line.startswith("Predicted:"):
                    parts = line.strip().split(", ")
                    predicted = int(parts[0].split(": ")[1])
                    true = int(parts[1].split(": ")[1])
                    predictions.append({"predicted": predicted, "true": true})
        
        print(f"Predictions read from {self.output_path}")
        return predictions

class PredictionEntityConverter():
    def __init__(self, predictions: list, linker: DBpediaLinker, data_file_path: str):
        self.predictions = predictions
        self.linker = linker
        # Load structured data from the provided JSON file path
        self.linker.load_structured_data(data_file_path)
        self.entity_mappings = []

    def get_converted(self, dbpedia_only = False):
        """
        Convert prediction indices to entity names using the linked data and map original entities to their linked names.
        """
        batch_size = len(self.predictions)//len(self.linker.structured_data)
        current_batch = 0
        entity_mapping = []
        index = 0
        prediction = {}
        while current_batch < len(self.predictions)//batch_size:
            for index in range(min(batch_size, len(self.predictions)-current_batch*batch_size)):
                prediction = self.predictions[index+current_batch*batch_size]
                if prediction['true'] == -1:
                    # Move to the next batch if we hit the separator
                    self.entity_mappings.append(entity_mapping)
                    entity_mapping = []
                    break

                # Calculate the current index within the structured data's records and lookup results
                record_index = index  # Index within the current batch
                structured_data_index = prediction['predicted']
                if dbpedia_only:
                    structured_data_index = prediction['true']

                try:
                    # Access the specific record from structured data to get the original entity
                    original_entity = self.linker.structured_data[current_batch][5][record_index][1]

                    # Access the specific lookup result from structured data
                    entity_name = self.linker.structured_data[current_batch][6][structured_data_index][0]
                    # Create a mapping dictionary
                    mapping = {
                        "original_entity": original_entity,
                        "linked_entity": entity_name
                    }
                    entity_mapping.append(mapping)

                except IndexError:
                    # Handle cases where the index is out of the range of available data
                    print(f"No data available for index {structured_data_index} in batch {current_batch} or record index {record_index}")
                    continue
            current_batch += 1
        self.entity_mappings.append(entity_mapping)
        return self.entity_mappings
        
    def apply_convertion(self, input_data, output_dir):
        """
        Apply entity mappings to the specified columns of input data tables and save the results to output_dir.

        Parameters:
            input_data (dict): dictionary where keys are file paths to CSVs and values are lists of columns to modify.
            output_dir (str): Directory to save the updated CSV files.

        Returns:
            None: Saves the modified CSV files to output_dir.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("in apply_convertion, the entity_mapping: ", self.entity_mappings)
        for i, (csv_path, target_columns) in enumerate(input_data.items()):
            # Load the CSV file into a DataFrame
            df = pd.read_csv(csv_path)

            # Get the corresponding entity mappings for this table
            entity_mapping = self.entity_mappings[i]

            # Convert the entity mapping list to a dictionary for faster lookup
            mapping_dict = {item['original_entity']: item['linked_entity'] for item in entity_mapping}

            # Apply the conversion to the target columns
            for column in target_columns:
                if column in df.columns:
                    # Replace the entities in the column using the mapping dictionary
                    df[column] = df[column].replace(mapping_dict)
                else:
                    print(f"Column '{column}' not found in {csv_path}. Skipping.")

            # Construct the output file path
            csv_name = os.path.basename(csv_path)
            output_file_path = os.path.join(output_dir, csv_name)

            # Save the modified DataFrame to the output directory
            df.to_csv(output_file_path, index=False)
            print(f"Updated entities saved to {output_file_path}")


class TopkPredictionEntityConverter:
    def __init__(self, prediction_list_path: str = "", linker: DBpediaLinker = None, data_file_path: str = "", process_single: bool = False, dbpedia_only: bool = False):
        self.linker = linker
        self.linker.load_structured_data(data_file_path)
        self.data_file_path = data_file_path
        self.entity_mappings: list[list[dict[str, any]]] = []
        self.process_single = process_single
        self.dbpedia_only = dbpedia_only
        if not dbpedia_only:
            with open(prediction_list_path, 'r', encoding='utf-8') as f:
                self.predictions = json.load(f)

    def get_converted_topk_single(self, k: int = 3, force_update = False) -> list[list[dict[str, any]]]:
        self.entity_mappings = []
        entity_mapping = []
        current_batch = 0

        if os.path.exists(f"{self.data_file_path}/entity_mappings.json") and not force_update:
            with open(f"{self.data_file_path}/entity_mappings.json", 'r', encoding='utf-8') as f:
                self.entity_mappings = json.load(f)
            return self.entity_mappings

        if self.dbpedia_only and os.path.exists(f"{self.data_file_path}/entity_mappings_dbpedia_only.json") and not force_update:
            with open(f"{self.data_file_path}/entity_mappings_dbpedia_only.json", 'r', encoding='utf-8') as f:
                self.entity_mappings = json.load(f)
            return self.entity_mappings
        
        if self.dbpedia_only:
            for structured_data in self.linker.structured_data:
                entity_mapping = []
                for idx, record in enumerate(structured_data[5]):
                    original_entity = record
                    dbpedia_label_idx = structured_data[7][idx]
                    # Get the top k starting from the dbpedia label index
                    linked_entities = []
                    for i in range(dbpedia_label_idx, dbpedia_label_idx + k):
                        try:
                            entity_name = structured_data[6][i][0]
                            linked_entities.append([entity_name, 1]) # 1 is just to take the place of the probability so that the format is consistent
                        except IndexError:
                            print(f"Entity index {i} out of range for batch {current_batch}. Skipping.")
                            continue
                    # Create a mapping dictionary
                    mapping = {
                        "original_entity": original_entity,
                        "linked_entities": linked_entities
                    }
                    entity_mapping.append(mapping)
                self.entity_mappings.append(entity_mapping)
            # get the direction name of data file path
            data_file_dir = os.path.dirname(self.data_file_path)
            with open(f"{data_file_dir}/entity_mappings_dbpedia_only.json", 'w', encoding='utf-8') as f:
                json.dump(self.entity_mappings, f, ensure_ascii=False, indent=4)
            return self.entity_mappings



        for idx_table, prediction_by_table in enumerate(self.predictions):
            entity_mapping = []
            for idx_entity, prediction_by_entity in enumerate(prediction_by_table[0]):
                topk_indices_probs = self._get_topk_indices(prediction_by_entity, k)

                try:
                    original_entity = self.linker.structured_data[idx_table][5][idx_entity]
                except IndexError:
                    print(f"Original entity not found for batch {idx_table}, record {topk_indices_probs}. Skipping.")
                    continue
                # Get the top k linked entities
                linked_entities = []
                for idx, prob in topk_indices_probs:
                    try:
                        entity_name = self.linker.structured_data[idx_table][6][idx][0]
                        linked_entities.append((entity_name, prob))
                    except IndexError:
                        print(f"Entity index {idx} out of range for batch {idx_table}. Skipping.")
                        continue
                # Create a mapping dictionary
                mapping = {
                    "original_entity": original_entity,
                    "linked_entities": linked_entities
                }
                entity_mapping.append(mapping)

            self.entity_mappings.append(entity_mapping)
        # get the direction name of data file path
        data_file_dir = os.path.dirname(self.data_file_path)
        with open(f"{data_file_dir}/entity_mappings.json", 'w', encoding='utf-8') as f:
            json.dump(self.entity_mappings, f, ensure_ascii=False, indent=4)
        return self.entity_mappings

    def get_converted_topk(self, k: int = 3, force_update = False) -> list[list[dict[str, any]]]:
        batch_size = len(self.predictions[0])//len(self.linker.structured_data)
        self.entity_mappings = []
        entity_mapping = []
        index = 0
        prediction = {}
        current_batch = 0

        if os.path.exists(f"{self.data_file_path}/entity_mappings.json") and not force_update:
            with open(f"{self.data_file_path}/entity_mappings.json", 'r', encoding='utf-8') as f:
                self.entity_mappings = json.load(f)
            return self.entity_mappings

        print(f"The length of predictions: {len(self.predictions)}")
        print(f"The length of the first item in predictions: {len(self.predictions[0])}")
        while current_batch < len(self.predictions[0])//batch_size:
            for index in range(min(batch_size, len(self.predictions[0])-current_batch*batch_size)):
                prediction = self.predictions[0][index+current_batch*batch_size]
                if index > len(self.linker.structured_data[current_batch][5]):
                    # Move to the next batch if we hit the separator
                    self.entity_mappings.append(entity_mapping)
                    entity_mapping = []
                    break

                # Calculate the current index within the structured data's records and lookup results
                record_index = index
                # Get the top k highest probability entity index
                topk_indices_probs = self._get_topk_indices(prediction, k)
                # Get the original entity from the structured data
                try:
                    original_entity = self.linker.structured_data[current_batch][5][record_index][1]
                except IndexError:
                    print(f"Original entity not found for batch {current_batch}, record {record_index}. Skipping.")
                    continue
                # Get the top k linked entities
                linked_entities = []
                for idx, prob in topk_indices_probs:
                    try:
                        entity_name = self.linker.structured_data[current_batch][6][idx][0]
                        linked_entities.append((entity_name, prob))
                    except IndexError:
                        print(f"Entity index {idx} out of range for batch {current_batch}. Skipping.")
                        continue
                # Create a mapping dictionary
                mapping = {
                    "original_entity": original_entity,
                    "linked_entities": linked_entities
                }
                entity_mapping.append(mapping)
            current_batch += 1

            self.entity_mappings.append(entity_mapping)
        # get the direction name of data file path
        data_file_dir = os.path.dirname(self.data_file_path)
        with open(f"{data_file_dir}/entity_mappings.json", 'w', encoding='utf-8') as f:
            json.dump(self.entity_mappings, f, ensure_ascii=False, indent=4)
        return self.entity_mappings

    def _get_topk_indices(self, probabilities: list[float], k: int) -> list[tuple[int, float]]:

        indexed_probs = list(enumerate(probabilities))

        sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)

        topk = sorted_probs[:k]
        return topk

    def apply_conversion_topk(self, input_data: dict[str, list[str]], output_dir: str, k: int = 3):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Applying top{k} entity mappings...")

        for i, (csv_path, target_columns) in enumerate(input_data.items()):
            if i >= len(self.entity_mappings):
                print(f"No available entity mappings for {csv_path}, skipped.")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}, skipped.")
                continue

            entity_mapping = self.entity_mappings[i]

            # original_entity -> linked_entities list
            mapping_dict = {item['original_entity']: item['linked_entities'] for item in entity_mapping}

            expanded_rows = []

            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                rows_to_add = [row_dict]
                for column in target_columns:
                    if column in df.columns:
                        original_entity = row_dict[column]
                        if original_entity in mapping_dict:
                            linked_entities = mapping_dict[original_entity]
                            new_rows = []
                            for linked_entity, prob in linked_entities:
                                new_row = row_dict.copy()
                                new_row[column] = linked_entity
                                new_column_name = f"{column}$turl$"
                                new_row[new_column_name] = original_entity
                                new_rows.append(new_row)
                            rows_to_add = new_rows
                        else:
                            # If no mapping available, put the original entity in $turl$ column
                            for r in rows_to_add:
                                r[f"{column}$turl$"] = original_entity
                            pass
                    else:
                        print(f"Column '{column}' is not found in {csv_path}, skipped.")
                expanded_rows.extend(rows_to_add)

            expanded_df = pd.DataFrame(expanded_rows)

            csv_name = os.path.basename(csv_path)
            output_file_path = os.path.join(output_dir, csv_name)

            try:
                expanded_df.to_csv(output_file_path, index=False)
                print(f"Saved to {output_file_path}")
            except Exception as e:
                print(f"Error in saving {output_file_path}: {e}")

    def save_entity_mappings(self, output_path: str):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.entity_mappings, f, ensure_ascii=False, indent=4)
            print(f"Entity mappings saved to {output_path}")
        except Exception as e:
            print(f"Error saving entity mappings to {output_path}: {e}")

class CandidateJoiner():
    def __init__(self, buyer_candidates: list[list[dict[str, any]]]=None, seller_candidates: list[list[dict[str, any]]]=None, from_disk=False, file_path=None):
        self.buyer_candidates = buyer_candidates
        self.seller_candidates = seller_candidates
        if from_disk:
            self.load_candidates(file_path)

    def load_candidates(self, file_path: str):
        # should be json
        with open(file_path, 'r', encoding='utf-8') as f:
            self.seller_candidates = json.load(f)
        
    def _candidate_join(self) -> list[list[dict]]:
        """
        Buyer candidate list example: [[{"original_entity": "United States", "linked_entities": [("United States of America", 0.9), ("United States dollar", 0.8)]}, ...], ...]
        """
        buyer_candidates = self.buyer_candidates
        seller_candidates = self.seller_candidates
        for buyer in buyer_candidates:
            for entity in buyer:
                # Build a set of linked entities of buyer
                buyer_linked_entities = set([linked_entity[0] for linked_entity in entity["linked_entities"]])
                for seller in seller_candidates:
                    for s_entity in seller:
                        # Build a set of linked entities of seller
                        seller_linked_entities = set([linked_entity[0] for linked_entity in s_entity["linked_entities"]])
                        # Find the intersection of linked entities
                        intersection = buyer_linked_entities.intersection(seller_linked_entities)
                        # If there is an intersection, add "linked_entity" key with the buyer origin_entity as the value to the seller entity
                        if intersection:
                            s_entity["linked_entity"] = entity["original_entity"]
                        else:
                            s_entity["linked_entity"] = s_entity["original_entity"]
        return seller_candidates

    def apply_conversion_with_candidate_join(self, input_data: dict, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        seller_candidates = self._candidate_join()
        for i, (csv_path, target_columns) in enumerate(input_data.items()):
            # Load the CSV file into a DataFrame
            df = pd.read_csv(csv_path)

            # Convert the entity mapping list to a dictionary for faster lookup
            mapping_dict = {item['original_entity'][1]: item['linked_entity'][1] for item in seller_candidates[i]}
            print("mapping_dict: ", mapping_dict)
            # Apply the conversion to the target columns
            for column in target_columns:
                if column in df.columns:
                    # Replace the entities in the column using the mapping dictionary
                    df[column] = df[column].replace(mapping_dict)
                else:
                    print(f"Column '{column}' not found in {csv_path}. Skipping.")

            # Construct the output file path
            csv_name = os.path.basename(csv_path)
            output_file_path = os.path.join(output_dir, csv_name)

            # Save the modified DataFrame to the output directory
            df.to_csv(output_file_path, index=False)
            print(f"Updated entities saved to {output_file_path}")