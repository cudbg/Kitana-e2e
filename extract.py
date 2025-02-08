import os
import shutil
import pandas as pd
import csv
import json
from tqdm import tqdm
import random
import argparse

def collect_csv_files():
    # get teh arguments for source_dir and destination_dir
    parser = argparse.ArgumentParser(description="Collect csv files from source_dir to destination_dir")
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Source directory containing csv files"
    )
    parser.add_argument(
        "--destination_dir",
        type=str,
        required=True,
        help="Destination directory to copy csv files"
    )
    args = parser.parse_args()
    source_dir = args.source_dir
    destination_dir = args.destination_dir

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(destination_dir, file)
                count = 1
                base, ext = os.path.splitext(file)
                while os.path.exists(dst_file):
                    dst_file = os.path.join(destination_dir, f"{base}_{count}{ext}")
                    count += 1
                
                try:
                    shutil.copy2(src_file, dst_file)
                    print(f"copy: {src_file} -> {dst_file}")
                except Exception as e:
                    print(f"Error in copying {src_file} {e}")



if __name__ == "__main__":
    collect_csv_files()
    # input_file = "companiesmarketcap.com - Companies ranked by earnings - CompaniesMarketCap.com.csv"
    # output_file = "companiesmarketcap.com - Companies ranked by earnings - CompaniesMarketCap.com_clean.csv"
    # remove_quotes_from_csv(input_file, output_file)

    # see_column_content(dir_path=source_dir,  csv_header_dict=csv_header)
    # num_rows_per_table = 250
    # output_dir = "/home/ec2-user/Kitana_e2e/Kitana-e2e/data/company"
    # split_and_shuffle_large_csv(csv_header_dict=csv_header, intermediate_chunk_size=1000000, final_chunk_size=100000, output_dir="/home/ec2-user/Kitana_e2e/Kitana-e2e/data/company")

