#!/bin/bash

# dataset_retrieve.sh
# This script is used to search for folders containing specified keywords in the S3 bucket and optionally save results to a file.

# Default S3 base path variables 
# Please modify S3_BUCKET and S3_BASE_PREFIX according to actual needs
DEFAULT_S3_BUCKET="kitana-data"
DEFAULT_S3_BASE_PREFIX="TURL"  # Can be a subfolder path, like "data/experiments"

# Ensure AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install AWS CLI and configure credentials."
    exit 1
fi

# Function: Display script usage
usage() {
    echo "Usage: $0 -k <search_key> [-b <bucket_name>] [-p <base_prefix>] [-s <store_path>] [-h]"
    echo "  -k, --key        Search keyword (required)"
    echo "  -b, --bucket     S3 bucket name (optional, default: $DEFAULT_S3_BUCKET)"
    echo "  -p, --prefix     S3 base prefix (optional, default: $DEFAULT_S3_BASE_PREFIX)"
    echo "  -s, --store      Path to save results (optional)"  
    echo "  -h, --help       Display this help message"
    exit 1
}

# Initialize variables
SEARCH_KEY=""
S3_BUCKET="$DEFAULT_S3_BUCKET"
S3_BASE_PREFIX="$DEFAULT_S3_BASE_PREFIX"
STORE_PATH=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -k|--key)
            SEARCH_KEY="$2"
            shift 2
            ;;
        -b|--bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        -p|--prefix)
            S3_BASE_PREFIX="$2"
            shift 2
            ;;
        -s|--store)
            STORE_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
done

# Check if search keyword is provided
if [ -z "$SEARCH_KEY" ]; then
    echo "Error: No search keyword provided."
    usage
fi

# Build complete S3 path prefix
S3_PATH="s3://$S3_BUCKET/$S3_BASE_PREFIX"

echo "Searching for folders containing keyword '$SEARCH_KEY' in S3 path: $S3_PATH..."

# Use AWS CLI to list all folders containing the search keyword
# Note: S3 doesn't have real folder structure, we simulate folders by listing object prefixes
MATCHING_FOLDERS=$(aws s3 ls "$S3_PATH" --recursive | \
    awk '{print $4}' | \
    grep "/$SEARCH_KEY/" | \
    awk -F/ '{for(i=1;i<=NF-1;i++) printf $i"/"; print ""}' | \
    sort | uniq)

# Check if matching folders are found
if [ -z "$MATCHING_FOLDERS" ]; then
    echo "No folders found containing keyword '$SEARCH_KEY'."
    exit 0
fi

# Display list of matching folders
echo "Found following folders containing keyword '$SEARCH_KEY':"
echo "-------------------------------------------"
echo "$MATCHING_FOLDERS"
echo "-------------------------------------------"

# If storage path is specified, save results to file
if [ -n "$STORE_PATH" ]; then
    # Check if target directory exists, create if not
    STORE_DIR=$(dirname "$STORE_PATH")
    if [ ! -d "$STORE_DIR" ]; then
        echo "Target directory '$STORE_DIR' does not exist, creating..."
        mkdir -p "$STORE_DIR"
        if [ $? -ne 0 ]; then
            echo "Cannot create directory '$STORE_DIR'. Please check permissions or path."
            exit 1
        fi
    fi

    # Save results to specified file
    echo "$MATCHING_FOLDERS" > "$STORE_PATH"
    if [ $? -eq 0 ]; then
        echo "Results successfully saved to '$STORE_PATH'."
    else
        echo "Error saving results to '$STORE_PATH'."
        exit 1
    fi
fi

exit 0
