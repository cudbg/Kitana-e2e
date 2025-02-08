#!/usr/bin/env python3
import os
import argparse
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    parser = argparse.ArgumentParser(description="Download CSV datasets from Kaggle using a search query (with pagination).")
    parser.add_argument(
        "--search", 
        type=str, 
        required=True, 
        help="Search keyword for Kaggle datasets."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=10, 
        help="Maximum number of datasets to download."
    )
    parser.add_argument(
        "--outdir", 
        type=str, 
        default="./data", 
        help="Directory to store downloaded datasets."
    )
    
    args = parser.parse_args()
    search_key = args.search
    limit = args.limit
    output_dir = args.outdir
    os.makedirs(output_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    all_datasets = []
    page = 1

    while True:
        if len(all_datasets) >= limit:
            break

        datasets_page = api.dataset_list(
            search=search_key,
            file_type='csv',
            page=page,
            max_size=250000000
        )

        if not datasets_page:
            break

        all_datasets.extend(datasets_page)
        page += 1

    all_datasets = all_datasets[:limit]
    print("Retrieved datasets: ", all_datasets)

    for ds in all_datasets:
        ds_ref = ds.ref
        ds_name = ds_ref.replace("/", "_")
        ds_path = os.path.join(output_dir, ds_name)

        os.makedirs(ds_path, exist_ok=True)

        print(f"Downloading dataset: {ds_ref} ...")
        api.dataset_download_files(dataset=ds_ref, path=ds_path, quiet=False)

        for filename in os.listdir(ds_path):
            if filename.endswith(".zip"):
                zip_file_path = os.path.join(ds_path, filename)
                print(f"Unzipping {zip_file_path}...")
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(ds_path)
                os.remove(zip_file_path)

    print("All datasets downloaded and unzipped successfully.")

if __name__ == "__main__":
    main()
