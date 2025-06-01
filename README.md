

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

## A typical flow
<img width="1073" alt="image" src="https://github.com/user-attachments/assets/97c2e083-d15a-4156-b2d4-b7f98c835c93" />


This part outlines the typical execution flow within the codebase, starting from input data specifications to the generation of experiment results, specifically focusing on how a single experiment run (like the one configured for `data/country_extend_1/seller`) proceeds.

### 1. Entry Point: `main.py`

- The process begins in `main.py`. This script is responsible for defining and triggering various experiment configurations.
- An experiment is typically defined by instantiating a `Config` object (from `search_engine.config`). This object aggregates several configuration dataclasses:
    - `DataConfig`: Specifies the data inputs. Key parameters include:
        - `directory_path`: Path to the directory containing seller CSV files (e.g., `data/country_extend_1/seller`).
        - `buyer_csv`: Path to the buyer's CSV file.
        - `join_keys`: A list of lists specifying potential join columns (e.g., `[['country']]`).
        - `target_feature`: The column name in the buyer's data to be predicted.
        - `need_to_clean_data`: Boolean, if true, data cleaning is performed.
        - `one_target_feature`: Boolean, if true, buyer data is initially reduced to join keys and target.
    - `SearchConfig`: Specifies search algorithm parameters. Key parameters:
        - `iterations`: Number of features to select.
        - `fit_by_residual`: Boolean, if true, the search aims to explain the residuals of a model built on the buyer's initial features.
        - `device`: 'cpu' or 'cuda'.
        - `batch_size`: For sketch processing, can be 'auto'.
    - `ExperimentConfig`: Specifies experiment logging and output.
    - `LoggingConfig`: Configures logging.
- `main.py` then instantiates `ScaledExperiment` (from `search_engine.experiment.experiment`) with this `Config`.
- Finally, it calls the `run()` method on the `ScaledExperiment` instance and typically saves the returned results (often an 'augplan' and 'accuracy' list) to a JSON file.

### 2. Experiment Orchestration: `ScaledExperiment`

The `ScaledExperiment` class (`search_engine/experiment/experiment.py`) orchestrates the entire experimental procedure. Its `run()` method executes the following steps:

#### 2.1. Load Buyer Data (`load_buyer`)
- An instance of `PrepareBuyer` (from `search_engine.data_provider.buyer_data`) is created.
- `PrepareBuyer` inherits from `PrepareData` (`search_engine/data_provider/base_data.py`):
    - Loads the buyer CSV specified in `DataConfig.buyer_csv`.
    - **Join Key Processing (`_check_join_keys`, `_construct_join_keys` in `PrepareData`):**
        - Validates that the provided `join_keys` (e.g., `[['country']]`) exist in the buyer's DataFrame columns.
        - If a join key is multi-column (e.g., `['colA', 'colB']`), a new concatenated column (e.g., `'colA_colB'`) is created in the DataFrame to serve as a single string join key. A list of these processed string join keys is stored (e.g., `self.join_keys_in_string`).
    - **Data Cleaning (`_data_cleaning` in `PrepareData`, if `need_to_clean_data` is true):**
        - Converts columns to numeric where possible (coercing errors).
        - Removes columns with > 40% missing values.
        - Fills remaining NaNs (e.g., with 0 or mean, depending on context).
    - **Buyer-Specific Column Selection (`PrepareBuyer.__init__`):**
        - The buyer DataFrame is trimmed to include only:
            - The processed `join_keys_in_string`.
            - The `target_feature`.
            - Numerical columns identified during cleaning (or a predefined list if cleaning is off).
        - If `DataConfig.one_target_feature` is true, it's further trimmed to just join keys and the target feature.
    - Calculates `buyer_key_domain`: a dictionary mapping each processed join key string to a set of its unique values in the buyer data.
- The `PrepareBuyer` instance is stored in `PrepareBuyerSellers` (a wrapper class).

#### 2.2. Load Seller Data (`load_sellers`)
- Iterates through all CSV files in the `DataConfig.directory_path`.
- For each seller CSV:
    - An instance of `PrepareSeller` (from `search_engine.data_provider.seller_data`) is created.
    - `PrepareSeller` also inherits from `PrepareData` and performs similar loading, join key processing, and data cleaning as `PrepareBuyer`. Seller columns are trimmed to join keys and numerical/specified features.
    - The `PrepareSeller` instance is added via `PrepareBuyerSellers.add_seller()`, which in turn calls `PrepareSellers.add_sellers()` (`search_engine/data_provider/sellers.py`).
    - **Seller Filtering (`PrepareSellers.add_sellers`):**
        - For each of the seller's processed join keys, it checks if the seller's unique values for that key have any intersection with the `buyer_key_domain` for the same key.
        - If a join key in the seller has *no overlap* with the buyer's values for that key, that join key column is *dropped* from the seller's DataFrame, and the key is removed from the seller's list of active join keys.
        - If a seller ends up with no valid (overlapping) join keys, it is effectively discarded.
    - **Global Domain Update (`PrepareSellers.update_domain`):**
        - If the seller is kept, its unique values for its valid join keys are used to update a global `join_key_domains` dictionary within `PrepareSellers`. This dictionary tracks all unique values seen for each join key across all *valid and filtered* sellers.

#### 2.3. Setup Data Market (`setup_market`)
- An instance of `DataMarket` (from `search_engine.market.data_market`) is created.
- **Register Buyer (`DataMarket.register_buyer`):**
    - The (cleaned and processed) buyer DataFrame and its `join_keys` (filtered by `PrepareBuyerSellers.filter_join_keys` against the global seller domain) are passed. The `target_feature` and global `join_key_domains` (from `PrepareSellers`) are also passed.
    - **Initial Model & Residuals:**
        - A linear regression model is fit on the buyer's initial non-join-key features against the `target_feature`. The R² of this baseline model is calculated and stored (this is the first value in the `accuracy` list of the experiment results).
        - If `SearchConfig.fit_by_residual` is true, the buyer's DataFrame for sketching purposes is replaced by a DataFrame containing only its join keys and the *residuals* from this initial model. The "target feature" for sketching becomes these residuals.
    - **Buyer Sketch Creation:** For each active join key of the buyer:
        - A `SketchBase` object (from `search_engine.sketches.base`) is obtained/created. This `SketchBase` is specific to the join key string and is shared by all sellers (and the buyer) using this key. It's initialized with the global `join_key_domains` for this key.
        - A `BuyerSketch` object (from `search_engine.sketches.sketch_buyer`) is created.
        - `BuyerSketch.register_this_buyer()` is called:
            - This calls `SketchBase._calibrate()`:
                - Calculates sums (X), sums of squares (X²), and counts (1) of the buyer's features (or residuals if `fit_by_residual`) grouped by the join key values.
                - If not `fit_by_residual`, it also calculates sums of cross-products of buyer features with the target (XY).
                - These statistics are normalized (to means) and aligned/reindexed to the full `join_key_domain`, creating PyTorch tensors. These tensors are the "buyer sketch" components.
            - Then `SketchBase._register_df()` is called to pass these tensors to a `SketchLoader` instance associated with the `SketchBase`.
            - `SketchLoader` (`search_engine/sketches/loader.py`) stores these buyer sketch tensors (typically in memory, in "batch 0" as buyers are assumed small).
- **Register Sellers (`DataMarket.register_seller`):** For each valid (filtered) seller from `PrepareSellers`:
    - The seller's feature column names are prefixed with `seller_name_` to avoid collisions.
    - For each active join key of the seller:
        - A `SellerSketch` object (from `search_engine.sketches.sketch_seller`) is created, using the shared `SketchBase` for that join key.
        - `SellerSketch.register_this_seller()` is called:
            - If the seller has many features (more than `SketchBase.ram_batch_size`), its features are partitioned.
            - For each partition (or all features if few):
                - `SketchBase._calibrate()` is called to create sketch tensors (X, X², 1) for this set of seller features, grouped by join key, normalized, and aligned to the domain.
                - `SketchBase._register_df()` passes these tensors to the `SketchLoader`.
            - `SketchLoader` appends these seller feature sketch tensors to appropriate batches (batching along the feature dimension). If a batch becomes full (reaches `batch_size` features), it can be offloaded to disk. `SketchLoader` also maintains a `feature_index_map` within the `SketchBase` to trace features in sketch batches back to their original seller and column name.

#### 2.4. Run Search (`run_search`)
- An instance of `SearchEngine` (from `search_engine.search.search_engine`) is created with the populated `DataMarket`.
- `SearchEngine.start(iterations)` is called:
    - This initiates an iterative feature selection process for the number of `iterations` specified in `SearchConfig`.
    - **In each iteration (`SearchEngine.search_one_iteration`):**
        - It aims to find the single best seller feature to add to the current buyer model.
        - It iterates over each `join_key` common to the buyer and sellers.
        - It retrieves the buyer's current sketch tensors (Y, Y², 1, XY or Residuals, Residuals², 1) for that join key.
        - It iterates through all batches of seller sketch features (X, X², 1) for that join key from the `SketchLoader`.
        - **R² Calculation using Sketches:** For each candidate seller feature in a batch:
            - It performs "sketch joins" by element-wise multiplication and summation of the buyer's and seller's sketch tensors. This allows it to calculate the sufficient statistics (covariances, etc.) needed for a linear regression model *as if* the buyer data were joined with that seller feature.
            - If `fit_by_residual` is true: It calculates the R² of regressing the buyer's current residuals onto the candidate seller feature.
            - If `fit_by_residual` is false: It calculates the R² of a model including all current buyer features *plus* the candidate seller feature, predicting the original buyer target. This involves more complex matrix operations (XTX, XTY, inversion) using the sketch components.
            - Features causing singularities or already selected are skipped.
        - The seller feature (from any join key, any batch) that yields the highest R² improvement is chosen as `best_feature` for this iteration. Its original `seller_id` and `best_feature` name (prefixed) are retrieved using `SketchBase.get_df_by_feature_index`.
    - **Augmentation Plan Update:** The chosen `(seller_id, iteration_number, seller_name, best_feature_name)` is added to `self.augplan`.
    - **Update Buyer State (`SearchEngine._update_residual`):**
        - The original buyer DataFrame (from `DataMarket.buyer_dataset_for_residual` if fitting by residual, or `DataMarket.buyer_id_to_df_and_name[0]["dataframe"]` which should reflect augmentations from previous iterations) is retrieved.
        - The original DataFrame for the chosen `seller_id` is retrieved. The `best_feature` (with its `seller_name_` prefix) is selected from it, along with the `join_key`.
        - The buyer DataFrame is **joined** with this seller feature on the `join_key`. The seller feature is typically aggregated (e.g., mean) by `join_key` before joining.
        - This augmented buyer DataFrame becomes the new basis for the buyer.
        - **Crucially, the buyer is effectively re-registered in the `DataMarket` with this augmented data:**
            - `DataMarket.buyer_sketches` and related buyer info are reset.
            - `DataMarket.register_buyer()` is called again with the augmented buyer DataFrame. This recalculates the R² of the new model (now including the `best_feature`), and this R² is appended to `DataMarket.augplan_acc`. If `fit_by_residual`, new residuals are computed. New buyer sketches are generated and loaded.
- The loop continues for the specified number of `iterations`.

#### 2.5. Plot Results and Return
- After the search loop, `plot_results()` may be called to generate plots (e.g., accuracy vs. iteration).
- The `run()` method returns a dictionary containing:
    - `'augplan'`: The list of selected features.
    - `'accuracy'`: The list of R² values, starting with the buyer's initial R² and then the R² after adding each feature from the `augplan`.
    - `'time_taken'`: Execution time.

This flow details how the system processes input data, transforms it into sketches, uses these sketches to efficiently find relevant features from sellers, and iteratively augments the buyer's dataset, tracking predictive accuracy along the way. 
