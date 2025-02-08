import json

def load_prediction_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def link_predictions_with_entities(predictions, entities):
    linked_data = []
    for index, prediction_set in enumerate(predictions):
        for score, entity in zip(prediction_set, entities[index::3]):  # Adjust indexing to match entities correctly
            linked_data.append((entity[0], score))  # Convert scores to list if needed
    return linked_data

# Save the linked data to a new file
def save_linked_data(linked_data, output_file_path):
    with open(output_file_path, 'w') as file:
        json.dump(linked_data, file, indent=4)

def load_entity_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

from search_engine.experiment import ScaledExperiment
from search_engine.config import get_config, Config, DataConfig, SearchConfig, ExperimentConfig, LoggingConfig
from search_engine.entity_linking.el_test import DBpediaLinker, TURLExecuter, PredictionEntityConverter, TopkPredictionEntityConverter, CandidateJoiner
import os
import json

def build_input_data(folder_path: str, columns: list) -> dict:
    input_data = {}
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            csv_path = os.path.join(folder_path, file)
            input_data[csv_path] = columns
    return input_data
def print_json_length(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if isinstance(data, list):
            print("Length of the list:", len(data))

    except FileNotFoundError:
        print("File not found:", json_file_path)
    except json.JSONDecodeError:
        print("Failed to decode JSON from the file.")
    except Exception as e:
        print("An error occurred:", str(e))

def get_selected(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print("An error occurred:", str(e))
        
def main():
    file_label = "10_top10"
    config = get_config()
    # Instantiate the DBpediaLinker class
    linker = DBpediaLinker(output_dir_base=f"el_data/country_extend/buyer/true/{file_label}/", auto_load_meta_data=True)
    input_data = {
        "data/country_extend/buyer/master.csv": ["Country"]
    }
    percentage_list = [10]
    if not os.path.exists(linker.output_dir_base):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)


    executer = TURLExecuter(
        input_data = "/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/buyer/true/10/structured_data.json",
        host_input_dir="/TURL/data/kitana/country_extend/buyer/10",
        output_path="/home/ec2-user/TURL/data/kitana/country_extend/buyer/10/predictions.txt"
    )
    executer.prepare_input_file()
    executer.run_container_evaluation()
    buyer_predictions = executer.read_predictions()
    
    converter = PredictionEntityConverter(buyer_predictions, linker, "/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/buyer/true/10/structured_data.json")
    print("converted seller: ", converter.get_converted())
    print("selected: ",get_selected("el_data/country_extend/buyer/true/10/selected_input.json"))
    converter.apply_convertion(input_data=get_selected("el_data/country_extend/buyer/true/10/selected_input.json"), output_dir="/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/buyer/10")

    input_data = build_input_data("data/country_extend/seller", ["Country"])
    percentage_list = [10] * len(input_data)
    linker.output_dir_base = "el_data/country_extend/seller/10/"
    if not os.path.exists(linker.output_dir_base):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)

    executer = TURLExecuter(
        input_data="/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/10/structured_data.json",
        host_input_dir="/TURL/data/kitana/country_extend/seller/10",
        output_path="/home/ec2-user/TURL/data/kitana/country_extend/seller/10/predictions.txt"
    )
    executer.prepare_input_file()
    executer.run_container_evaluation()
    seller_predictions = executer.read_predictions()

    converter = PredictionEntityConverter(seller_predictions, linker, "/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/10/structured_data.json")
    print("converted seller: ", converter.get_converted())
    print("selected: ",get_selected("el_data/country_extend/seller/10/selected_input.json"))
    converter.apply_convertion(input_data=get_selected("el_data/country_extend/seller/10/selected_input.json"), output_dir="/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/10/")

def topk_connection_single_joinable_search():

    file_label = "100_top10_single"
    linker = DBpediaLinker(output_dir_base=f"el_data/country_extend_table_search/buyer/{file_label}/", auto_load_meta_data=True)
    input_data = {
        "data/country_extend_table_search/buyer/master.csv": ["Country"]
    }
    percentage_list = [100]
    if not os.path.exists(os.path.join(linker.output_dir_base, "structured_data.json")):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)

    if not os.path.exists(f"el_data/country_extend_table_search/buyer/{file_label}/Predictions_list.json"):
        executer = TURLExecuter(
            input_data=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend_table_search/buyer/{file_label}/structured_data.json",
            host_input_dir=f"/TURL/data/kitana/country_extend_table_search/buyer/{file_label}",
            output_path=f"/home/ec2-user/TURL/data/kitana/country_extend_table_search/buyer/{file_label}/predictions.txt"
        )
        executer.run_container_evaluation_single()


    buyer_converter = TopkPredictionEntityConverter(
        prediction_list_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend_table_search/seller/{file_label}/Predictions_list.txt",
        linker=linker,
        data_file_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend_table_search/buyer/{file_label}/structured_data.json"
    )
    buyer_converter.get_converted_topk_single(10)
    # Directly copy the buyer csv to the output directory
    os.system(f"cp data/country_extend_table_search/buyer/master.csv el_data/country_extend_table_search/buyer/{file_label}/master.csv")
    

    linker = DBpediaLinker(output_dir_base=f"el_data/country_extend_table_search/seller/{file_label}/", auto_load_meta_data=True)
    input_data = build_input_data("data/country_extend_table_search/seller", ["Country"])
    percentage_list = [100] * len(input_data)
    if not os.path.exists(os.path.join(linker.output_dir_base, "structured_data.json")):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)

    if not os.path.exists(f"el_data/country_extend_table_search/seller/{file_label}/Predictions_list.txt"):
        executer = TURLExecuter(
            input_data=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend_table_search/seller/{file_label}/structured_data.json",
            host_input_dir=f"/TURL/data/kitana/country_extend_table_search/seller/{file_label}",
            output_path=f"/home/ec2-user/TURL/data/kitana/country_extend_table_search/seller/{file_label}/predictions.txt"
        )
        executer.run_container_evaluation_single()
    
    seller_converter = TopkPredictionEntityConverter(
        prediction_list_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend_table_search/seller/{file_label}/Predictions_list.txt",
        linker=linker,
        data_file_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend_table_search/seller/{file_label}/structured_data.json"
    )
    seller_converter.get_converted_topk_single(10)

    joiner = CandidateJoiner(buyer_converter.entity_mappings, seller_converter.entity_mappings)
    joiner.apply_conversion_with_candidate_join(
        input_data=get_selected(f"el_data/country_extend_table_search/seller/{file_label}/selected_input.json"),
        output_dir=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend_table_search/seller/{file_label}/"
    )

def topk_connection_single_dbpedia_joinable_search():
    file_label = "100_top10_single_dbpedia"
    linker = DBpediaLinker(output_dir_base=f"el_data/country_extend_table_search/buyer/{file_label}/", auto_load_meta_data=True)
    input_data = {
        "data/country_extend_table_search/buyer/master.csv": ["Country"]
    }
    percentage_list = [100]
    if not os.path.exists(os.path.join(linker.output_dir_base, "structured_data.json")):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)
        
    buyer_converter = TopkPredictionEntityConverter(
        linker=linker,
        data_file_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend_table_search/buyer/{file_label}/structured_data.json",
        dbpedia_only=True
    )

    print("converted buyer: ", buyer_converter.get_converted_topk_single(10))
    # Directly copy the buyer csv to the output directory
    os.system(f"cp data/country_extend_table_search/buyer/master.csv el_data/country_extend_table_search/buyer/{file_label}/master.csv")


    linker = DBpediaLinker(output_dir_base=f"el_data/country_extend_table_search/seller/{file_label}/", auto_load_meta_data=True)
    input_data = build_input_data("data/country_extend_table_search/seller", ["Country"])
    percentage_list = [100] * len(input_data)
    if not os.path.exists(os.path.join(linker.output_dir_base, "structured_data.json")):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)


    seller_converter = TopkPredictionEntityConverter(
        linker=linker,
        data_file_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend_table_search/seller/{file_label}/structured_data.json",
        dbpedia_only=True
    )
    print("converted seller: ", seller_converter.get_converted_topk_single(10))

    joiner = CandidateJoiner(buyer_converter.entity_mappings, seller_converter.entity_mappings)
    joiner.apply_conversion_with_candidate_join(
        input_data=get_selected(f"el_data/country_extend_table_search/seller/{file_label}/selected_input.json"),
        output_dir=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend_table_search/seller/{file_label}/"
    )


def topk_connection_single_dbpedia():
    file_label = "100_top10_single_dbpedia"
    linker = DBpediaLinker(output_dir_base=f"el_data/country_extend/buyer/{file_label}/", auto_load_meta_data=True)
    input_data = {
        "data/country_extend/buyer/master.csv": ["Country"]
    }
    percentage_list = [100]
    if not os.path.exists(os.path.join(linker.output_dir_base, "structured_data.json")):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)
        
    buyer_converter = TopkPredictionEntityConverter(
        linker=linker,
        data_file_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/buyer/{file_label}/structured_data.json",
        dbpedia_only=True
    )
    print("converted buyer: ", buyer_converter.get_converted_topk_single(10))
    # Directly copy the buyer csv to the output directory
    os.system(f"cp data/country_extend/buyer/master.csv el_data/country_extend/buyer/{file_label}/master.csv")

    linker = DBpediaLinker(output_dir_base=f"el_data/country_extend/seller/{file_label}/", auto_load_meta_data=True)
    input_data = build_input_data("data/country_extend/seller", ["Country"])
    percentage_list = [100] * len(input_data)
    if not os.path.exists(os.path.join(linker.output_dir_base, "structured_data.json")):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)

    seller_converter = TopkPredictionEntityConverter(
        linker=linker,
        data_file_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/{file_label}/structured_data.json",
        dbpedia_only=True
    )
    print("converted seller: ", seller_converter.get_converted_topk_single(10))

    joiner = CandidateJoiner(buyer_converter.entity_mappings, seller_converter.entity_mappings)
    joiner.apply_conversion_with_candidate_join(
        input_data=get_selected(f"el_data/country_extend/seller/{file_label}/selected_input.json"),
        output_dir=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/{file_label}/"
    )


def topk_connection_single():
    file_label = "100_top10_single"
    linker = DBpediaLinker(output_dir_base=f"el_data/country_extend/buyer/{file_label}/", auto_load_meta_data=True)
    input_data = {
        "data/country_extend/buyer/master.csv": ["Country"]
    }
    percentage_list = [100]
    if not os.path.exists(os.path.join(linker.output_dir_base, "structured_data.json")):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)

    if not os.path.exists(f"el_data/country_extend/buyer/{file_label}/entity_mappings.json"):
        executer = TURLExecuter(
            input_data=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/buyer/{file_label}/structured_data.json",
            host_input_dir=f"/TURL/data/kitana/country_extend/buyer/{file_label}",
            output_path=f"/home/ec2-user/TURL/data/kitana/country_extend/buyer/{file_label}/predictions.txt"
        )
        executer.run_container_evaluation_single()


    buyer_converter = TopkPredictionEntityConverter(
        prediction_list_path=f"/home/ec2-user/TURL/data/kitana/country_extend/buyer/{file_label}/Predictions_list.txt",
        linker=linker,
        data_file_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/buyer/{file_label}/structured_data.json"
    )
    print("converted buyer: ", buyer_converter.get_converted_topk(10))
    # Directly copy the buyer csv to the output directory
    os.system(f"cp data/country_extend/buyer/master.csv el_data/country_extend/buyer/{file_label}/master.csv")
    

    linker = DBpediaLinker(output_dir_base=f"el_data/country_extend/seller/{file_label}/", auto_load_meta_data=True)
    input_data = build_input_data("data/country_extend/seller", ["Country"])
    percentage_list = [100] * len(input_data)
    if not os.path.exists(os.path.join(linker.output_dir_base, "structured_data.json")):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)

    if not os.path.exists(f"el_data/country_extend/seller/{file_label}/Predictions_list.txt"):
        executer = TURLExecuter(
            input_data=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/{file_label}/structured_data.json",
            host_input_dir=f"/TURL/data/kitana/country_extend/seller/{file_label}",
            output_path=f"/home/ec2-user/TURL/data/kitana/country_extend/seller/{file_label}/predictions.txt"
        )
        executer.run_container_evaluation_single()
    
    seller_converter = TopkPredictionEntityConverter(
        prediction_list_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/{file_label}/Predictions_list.txt",
        linker=linker,
        data_file_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/{file_label}/structured_data.json"
    )
    print("converted seller: ", seller_converter.get_converted_topk_single(10))

    joiner = CandidateJoiner(buyer_converter.entity_mappings, seller_converter.entity_mappings)
    joiner.apply_conversion_with_candidate_join(
        input_data=get_selected(f"el_data/country_extend/seller/{file_label}/selected_input.json"),
        output_dir=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/{file_label}/"
    )



def topk_connection():
    file_label = "100_top10"
    linker = DBpediaLinker(output_dir_base=f"el_data/country_extend/buyer/{file_label}/", auto_load_meta_data=True)
    input_data = {
        "data/country_extend/buyer/master.csv": ["Country"]
    }
    percentage_list = [100]
    if not os.path.exists(os.path.join(linker.output_dir_base, "structured_data.json")):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)

    if not os.path.exists(f"el_data/country_extend/buyer/{file_label}/entity_mappings.json"):
        executer = TURLExecuter(
            input_data=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/buyer/{file_label}/structured_data.json",
            host_input_dir=f"/TURL/data/kitana/country_extend/buyer/{file_label}",
            output_path=f"/home/ec2-user/TURL/data/kitana/country_extend/buyer/{file_label}/predictions.txt"
        )
        executer.prepare_input_file()
        executer.run_container_evaluation()

    buyer_converter = TopkPredictionEntityConverter(
        prediction_list_path=f"/home/ec2-user/TURL/data/kitana/country_extend/buyer/{file_label}/Predictions_list.txt",
        linker=linker,
        data_file_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/buyer/{file_label}/structured_data.json"
    )
    print("converted buyer: ", buyer_converter.get_converted_topk(10))
    # Directly copy the buyer csv to the output directory
    os.system(f"cp data/country_extend/buyer/master.csv el_data/country_extend/buyer/{file_label}/master.csv")
    # print("selected: ", get_selected(f"el_data/country_extend/buyer/{file_label}/selected_input.json"))
    # buyer_converter.apply_conversion_topk(
    #     input_data=get_selected(f"el_data/country_extend/buyer/{file_label}/selected_input.json"),
    #     output_dir=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/buyer/{file_label}/"
    # )
    
    input_data = build_input_data("data/country_extend/seller", ["Country"])
    percentage_list = [100] * len(input_data)
    linker.output_dir_base = f"el_data/country_extend/seller/{file_label}/"
    # if the output directory does not have structured_data.json, then run the batch_link
    if not os.path.exists(os.path.join(linker.output_dir_base, "structured_data.json")):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)

    if not os.path.exists(f"el_data/country_extend/seller/{file_label}/entity_mappings.json"):

        executer = TURLExecuter(
            input_data=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/{file_label}/structured_data.json",
            host_input_dir=f"/TURL/data/kitana/country_extend/seller/{file_label}",
            output_path=f"/home/ec2-user/TURL/data/kitana/country_extend/seller/{file_label}/predictions.txt"
        )
        executer.prepare_input_file()
        executer.run_container_evaluation()


    seller_converter = TopkPredictionEntityConverter(
        prediction_list_path=f"/home/ec2-user/TURL/data/kitana/country_extend/seller/{file_label}/Predictions_list.txt",
        linker=linker,
        data_file_path=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/{file_label}/structured_data.json"
    )
    print("converted seller: ", seller_converter.get_converted_topk(10))
    # print("selected: ", get_selected(f"el_data/country_extend/seller/{file_label}/selected_input.json"))
    # seller_converter.apply_conversion_topk(
    #     input_data=get_selected(f"el_data/country_extend/seller/{file_label}/selected_input.json"),
    #     output_dir=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/{file_label}/"
    # )
    joiner = CandidateJoiner(buyer_converter.entity_mappings, seller_converter.entity_mappings)
    joiner.apply_conversion_with_candidate_join(
        input_data=get_selected(f"el_data/country_extend/seller/{file_label}/selected_input.json"),
        output_dir=f"/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country_extend/seller/{file_label}/"
    )

    
    
if __name__ == "__main__":
    topk_connection_single_joinable_search()