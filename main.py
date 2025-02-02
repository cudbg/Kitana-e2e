from search_engine.experiment import ScaledExperiment
from search_engine.config import get_config
from search_engine.entity_linking.el_test import DBpediaLinker, TURLExecuter, PredictionEntityConverter
import os


def main():
    config = get_config()
    # Instantiate the DBpediaLinker class
    linker = DBpediaLinker(output_dir_base="el_data/country/buyer/2/", auto_load_meta_data=True)
    input_data = {
        "data/country/buyer/buyer_gini.csv": ["country"]
    }
    percentage_list = [2]
    if not os.path.exists(linker.output_dir_base):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)


    executer = TURLExecuter()
    executer.prepare_input_file()
    executer.run_container_evaluation()
    buyer_predictions = executer.read_predictions()
    
    converter = PredictionEntityConverter(buyer_predictions, linker, "/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country/buyer/2/structured_data.json")
    print("converted: ", converter.get_converted())
    converter.apply_convertion(input_data=input_data, output_dir="/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country/buyer/2")

    input_data = {
        #"data/country/seller/seller_fifa.csv": ["country"],
        "data/country/seller/seller_happiness.csv": ["country"],
        "data/country/seller/seller_life.csv": ["country"],
        #"data/country/seller/seller_pollution.csv": ["country"],
        "data/country/seller/seller_suicide.csv": ["country"]
    }
    percentage_list = [2,2,2]
    linker.output_dir_base = "el_data/country/seller/2/"
    if not os.path.exists(linker.output_dir_base):
        linker.batch_link(input_data=input_data, percentage_list=percentage_list)

    executer = TURLExecuter(
        input_data="/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country/seller/2/structured_data.json",
        host_input_dir="/TURL/data/kitana/country/seller/2"
    )
    executer.prepare_input_file()
    executer.run_container_evaluation()
    seller_predictions = executer.read_predictions()

    converter = PredictionEntityConverter(seller_predictions, linker, "/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country/seller/2/structured_data.json")
    print("converted seller: ", converter.get_converted())
    converter.apply_convertion(input_data=input_data, output_dir="/home/ec2-user/Kitana_e2e/Kitana-e2e/el_data/country/seller/2/")
    
    experiment = ScaledExperiment(config)
    results = experiment.run()
    
    print("\nExperiment Results:")
    print(f"Final Accuracy: {results['accuracy']}")
    print(f"Time Taken: {results['time_taken']:.2f} seconds")
    print(f"Number of Features Found: {len(results['augplan'])}")

if __name__ == "__main__":
    main()