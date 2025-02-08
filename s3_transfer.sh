#!/bin/bash

# Define S3 URL variables here
S3_UPLOAD_URL="s3://kitana-data/kitana-e2e"
S3_DOWNLOAD_URL="s3://kitana-data/kitana-e2e"

# Ensure AWS CLI is installed (should already be available in SageMaker environments)
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please ensure the SageMaker environment has AWS CLI available."
    exit 1
fi

# Extract bucket name and folder path from S3 URL
BUCKET_NAME=$(echo $S3_UPLOAD_URL | awk -F/ '{print $3}')
UPLOAD_FOLDER=$(echo $S3_UPLOAD_URL | awk -F/ '{print substr($0, index($0,$4))}')

check_and_create_s3_folder() {
    echo "Checking if $S3_UPLOAD_URL exists..."
    if aws s3 ls "s3://$BUCKET_NAME/$UPLOAD_FOLDER" &> /dev/null; then
        echo "Upload folder exists."
    else
        echo "Upload folder does not exist."
        read -p "Do you want to create the folder in S3? (yes/no): " response
        if [[ "$response" == "yes" ]]; then
            echo "Creating folder $S3_UPLOAD_URL..."
            aws s3api put-object --bucket "$BUCKET_NAME" --key "$UPLOAD_FOLDER/"
            if [ $? -eq 0 ]; then
                echo "Folder $S3_UPLOAD_URL created successfully!"
            else
                echo "Failed to create folder. Please check your AWS permissions."
                exit 1
            fi
        else
            echo "Upload folder creation aborted. Exiting."
            exit 1
        fi
    fi
}

delete_files_except_script() {
    echo "Do you want to delete all files in the current directory except for s3_transfer.sh? (yes/no): "
    read -p "Your choice: " delete_choice
    if [[ "$delete_choice" == "yes" ]]; then
        echo "Deleting all files in the current directory except for s3_transfer.sh..."
        find . -type f ! -name "s3_transfer.sh" -exec rm -f {} +
        echo "All files except s3_transfer.sh have been deleted."
    else
        echo "Skipping file deletion step."
    fi
}

echo "Choose an option:"
echo "1. Upload all files in the current directory to a predefined S3 folder"
echo "2. Download all files from a predefined S3 folder to the current directory"
read -p "Enter your choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    # Check if bucket exists
    if aws s3 ls "s3://$BUCKET_NAME" &> /dev/null; then
        echo "Bucket $BUCKET_NAME exists."
        # Check and create folder if necessary
        check_and_create_s3_folder
        echo "Uploading all files in the current directory to $S3_UPLOAD_URL..."
        aws s3 cp . "$S3_UPLOAD_URL" --recursive --exclude "s3_transfer.sh"
        if [ $? -eq 0 ]; then
            echo "Upload completed successfully!"
            delete_files_except_script
        else
            echo "Upload failed. Please check your AWS CLI configuration."
        fi
    else
        echo "Bucket $BUCKET_NAME does not exist. Exiting."
        exit 1
    fi

elif [ "$choice" == "2" ]; then
    # Download files from predefined S3 folder
    echo "Downloading all files from $S3_DOWNLOAD_URL to the current directory..."
    aws s3 cp "$S3_DOWNLOAD_URL" . --recursive
    if [ $? -eq 0 ]; then
        echo "Download completed successfully!"
    else
        echo "Download failed. Please check your S3 URL and AWS CLI configuration."
    fi

else
    echo "Invalid choice. Please enter 1 or 2."
fi
