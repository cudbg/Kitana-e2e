

# Kitana e2e 


## Data Augmentation for Kitana
This repository contains the scalable e2e implementation for data augmentation for Kitana. The code is written in Python and contains sample data, sample execution code, and the data augmentation code.

Please follow the instructions below to run the code.

### Instructions
1. Clone the repository
2. Make sure you are in the correct directory:
```bash
cd kitana-e2e
```
3. Run the following command to install the required libraries:
```bash
# If you are using python venv.
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
# If you are using conda, there is a environment.yml file in the repository.
conda env create -f environment.yml
```
3. Run the following command to execute the code:
```bash
python sample_execution.py
``` 
## Project Structure
- **`api/`** - Contains the interfaces for external modules to interact with the core functionality of the search engine.
- **`config/`** - Configuration settings for the project, including default paths, device settings, etc.
- **`data_provider/`** - Core modules for data management, handling buyer and seller data.
- **`market/`** - It loads the buyer and seller data.
- **`models/`** - Defines all data models used throughout the project, including loaders and specific models for buyers and sellers.
- **`preprocessing/`** - Data preprocessing utilities, ensuring data is clean and formatted correctly before entering the workflow.
- **`resources/`** - Manages and optimizes computing resources, ensuring efficient use of available hardware.
- **`search/`** - Core search engine functionality, implementing the algorithms that enhance buyer dataset with seller features.
- **`sketches/`** - Contains the sketches for the data augmentation process. It is indexed by the `join_keys`.
- **`statistics/`** - Statistical tools and functions. It contains a linear regression model to determine the augmentation effect.
- **`utils/`** - General utilities used across the project for a variety of support tasks.
- **`main.py`** - The entry point of the project, initializing and starting the search engine.
