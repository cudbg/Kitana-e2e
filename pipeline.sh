#!/bin/bash
# pipeline.sh
# This script loops through a list of Kaggle search keywords,
# downloads 100 datasets per keyword using kaggle_download.py,
# and collects all CSV files using extract.py.
#
# For each keyword, if either the data directory or output directory already exists,
# the script will skip processing that keyword.

# Array of search keywords
keywords=(
    "Country",
    "United Nation"
)

# Loop through each keyword
for keyword in "${keywords[@]}"; do
    # Convert the keyword to a safe directory name (replace spaces with underscores)
    keyword_dir=$(echo "$keyword" | tr ' ' '_')

    # Define the data directory for downloaded and unzipped datasets
    data_dir="${keyword_dir}_dataset"
    
    # Define the output directory for collected CSV files
    output_dir="data/${keyword_dir}_datasets"

    echo "=================================="
    echo "Processing keyword: $keyword"
    echo "Data directory: $data_dir"
    echo "Output directory: $output_dir"
    echo "=================================="

    # Check if either data_dir or output_dir already exists.
    if [ -d "$data_dir" ] || [ -d "$output_dir" ]; then
        echo "Directory '$data_dir' or '$output_dir' already exists. Skipping keyword: $keyword."
        continue
    fi

    # Create directories for this keyword
    mkdir -p "$data_dir"
    mkdir -p "$output_dir"

    # Step 1: Download datasets using kaggle_download.py
    echo "Downloading datasets for keyword: $keyword..."
    python3 kaggle_download.py --search "$keyword" --limit 100 --outdir "$data_dir"
    if [ $? -ne 0 ]; then
        echo "Error: kaggle_download.py failed for keyword: $keyword"
        exit 1
    fi
    echo "Datasets downloaded and unzipped for keyword: $keyword."

    # Step 2: Collect all CSV files using extract.py
    echo "Collecting CSV files for keyword: $keyword..."
    python3 extract.py --source_dir "$data_dir" --destination_dir "$output_dir"
    if [ $? -ne 0 ]; then
        echo "Error: extract.py failed for keyword: $keyword"
        exit 1
    fi
    echo "CSV files collected successfully for keyword: $keyword."
done

echo "All keywords processed successfully."
