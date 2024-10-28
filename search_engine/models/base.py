import torch
import psutil
from functools import reduce
from .loader import SketchLoader

"""
This class is the base class for the sketch base. 
It contains the basic functions that are used by 
both the buyer and seller classes.
"""

class SketchBase:
    def __init__(self, join_key_domain, device='cpu', is_buyer=False) -> None:
        self.feature_index_mapping = {}
        self.dfid_feature_mapping = {}
        self.device = device
        self.join_key_domain = join_key_domain
        self.current_df_id = 0
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.init()
            gpu_total_mem = torch.cuda.get_device_properties(0).total_memory
            self.gpu_free_mem = gpu_total_mem - torch.cuda.memory_allocated(0)
        else:
            self.gpu_free_mem = None
        self.gpu_batch_size, self.ram_batch_size = self.estimate_batch_size()        
        # TODO: join key domain requried in estimate_batch_size
        # sketch loader only needs to fully utilize gpu memory
        self.sketch_loader = SketchLoader(self.gpu_batch_size, device=device, is_buyer=is_buyer)


    """
    This function is used to estimate the batch size based on the available memory.
    It will return the batch size for both GPU and RAM.

    @param join_key_domain: a dictionary containing the domain of each join key. 
            We need this because the size(0) would be the product of the domain of all join keys.
    
    @return: a tuple containing the batch size for GPU and RAM

    """
    def estimate_batch_size(self):
        # Similar logic as search_gpu.py, just copy and paste
        bytes_per_element = 4
        tensor_width = reduce(lambda x, y: x * len(y), 
                              self.join_key_domain.values(), 1)
        memory = psutil.virtual_memory()
        # TODO: 2 is a workaround
        available_memory = memory.available // 2
        ram_batch_size = available_memory // (bytes_per_element * 3 * tensor_width)
        if not self.gpu_free_mem or not torch.cuda.is_available():
            gpu_batch_size = ram_batch_size
        else:
            gpu_batch_size = self.gpu_free_mem // (bytes_per_element * 3 * tensor_width)
        return gpu_batch_size, ram_batch_size

    """
    This function is used to register a dataframe to the sketch base.
    As a base function, it only takes in 1, x, x_x, feature_num, df_id, and offset as input params. 
    It will check if the current tensors satisfies the ram requirements and load it to sketches with load_sketches. 
    After loading the sketches, it returns the updated offset corresponding to the id of the dataframe.

    @param df_id: the unique identifier of the dataframe. Later on, when we want to fetch the 
              corresponding df_id, we could use this identifier. It stores in a priority queue in feature_index_mapping.
    @param feature_num: the number of features in the dataframe
    @param seller_1: the 1 matrix of the dataframe
    @param seller_x: the X matrix of the dataframe
    @param seller_x_x: the X_X matrix of the dataframe
    @param to_disk: whether to save the sketches to disk

    @return: a dictionary containing the batch_id, df_id, and offset
    """
    def _register_df(self, df_id, feature_num, seller_1, seller_x, seller_x_x, seller_x_y=None, to_disk=False):
        # Before loading the sketches, check if the current tensors satisfy the ram requirements
        if seller_x.size(1) > min(self.gpu_batch_size, self.ram_batch_size):
            raise ValueError("The number of features in the dataframe is too large.")
        # Load the sketches
        self.sketch_loader.load_sketches(
            seller_1 = seller_1,
            seller_x = seller_x,
            seller_x_x = seller_x_x,
            seller_x_y = seller_x_y,
            feature_index_map = self.feature_index_mapping,
            seller_id = df_id,
            to_disk = to_disk
        )
        # Return the updated offset corresponding to the df_id. This is not efficient. Only for unit test usage.
        def find_by_seller_id(feature_index_map, seller_id):
            for batch_id, entries in feature_index_map.items():
                for end_pos, id, offset in entries:
                    if id == seller_id:
                        return batch_id, offset
            return None, None  # If the seller_id is not found

        batch_id, offset = find_by_seller_id(self.feature_index_mapping, df_id)
        return {"batch_id": batch_id, "df_id": df_id, "offset": offset}



    """
    This function is used to get the 1, X, X_X, and X_Y matrices of a dataframe.
    takes in a df( with join key as a col)  and the key_domainsreturn and returns the calibrated 1, x, x_x tensors.
    @param df_id: the unique identifier of the dataframe.
    @param df: the dataframe to be calibrated. This df should be with join keys.
    @param num_features: the number of features in the dataframe.
    @param key_domains: a dictionary containing the domain of each join key.
    @param join_keys: a list containing the names of the join keys.
    @param fit_by_residual: a boolean indicating whether to fit by residual.
    @param is_buyer: a boolean indicating whether the df is a buyer or seller.

    @return: a tuple containing the calibrated 1, x, x_x tensors
    """

    def _calibrate(self, df_id, df, num_features, key_domains, join_keys, normalized=True, fit_by_residual=False, is_buyer=False):
        # Get a squared df but not include join keys
        non_join_key_columns = df.columns.difference(join_keys)
        df_squared = df[non_join_key_columns] ** 2
        df_squared[join_keys] = df[join_keys]

        seller_sum = df.groupby(join_keys).sum()
        ordered_columns = list(seller_sum.columns)

        if df_id not in self.dfid_feature_mapping:
            self.dfid_feature_mapping[df_id] = ordered_columns
        else:
            self.dfid_feature_mapping[df_id] += ordered_columns
        
        seller_sum_squares = df_squared.groupby(join_keys).sum()[ordered_columns]
        seller_count = df.groupby(join_keys).size().to_frame('count')

        
        if not fit_by_residual and is_buyer:
            df_cross, ordered_cross_cols = {}, []
            for col1, col2 in combinations(ordered_columns, 2):
                df_cross[f"{col1}_{col2}"] = df[col1] * df[col2]
                ordered_cross_cols.append(f"{col1}_{col2}")
            df_cross = pd.DataFrame(df_cross)
            df_cross[join_keys] = df[join_keys]
            seller_sum_cross = df_cross.groupby(join_keys).sum()[ordered_cross_cols]
            if normalized:
                seller_sum_cross = seller_sum_cross.div(seller_count['count'], axis=0)
        
                
        # Normalize by seller_count if normalization is enabled
        if normalized:
            seller_sum = seller_sum.div(seller_count['count'], axis=0)
            seller_sum_squares = seller_sum_squares.div(seller_count['count'], axis=0)

            # Set seller_count to 1 for each group
            seller_count = seller_count.assign(count=1)

        if not isinstance(seller_sum.index, pd.MultiIndex):
            seller_sum.index = pd.MultiIndex.from_arrays(
                [seller_sum.index], names=join_keys)
            seller_sum_squares.index = pd.MultiIndex.from_arrays(
                [seller_sum_squares.index], names=join_keys)
            seller_count.index = pd.MultiIndex.from_arrays(
                [seller_count.index], names=join_keys)
            
            if not fit_by_residual and is_buyer:
                seller_sum_cross.index = pd.MultiIndex.from_arrays(
                    [seller_sum_cross.index], names=join_keys)
            
        # Create the correct multi_index for cartesian product
        index_ranges = [key_domains[col] for col in join_keys]
        multi_index = pd.MultiIndex.from_product(index_ranges, names=join_keys)
        
        # Temporary DataFrame to facilitate 'inner' join
        temp_df = pd.DataFrame(index=multi_index)

        # Reindex and perform inner join
        seller_x = seller_sum.reindex(multi_index, fill_value=0)
        seller_x = seller_x[seller_x.index.isin(temp_df.index)].values

        seller_x_x = seller_sum_squares.reindex(multi_index, fill_value=0)
        seller_x_x = seller_x_x[seller_x_x.index.isin(temp_df.index)].values

        seller_count = seller_count.reindex(multi_index, fill_value=1)
        seller_count = seller_count[seller_count.index.isin(temp_df.index)].values
        
        seller_x_y_tensor = None
        
        if not fit_by_residual and is_buyer:
            seller_x_y = seller_sum_cross.reindex(multi_index, fill_value=0)
            seller_x_y = seller_x_y[seller_x_y.index.isin(temp_df.index)].values
            seller_x_y_tensor = torch.tensor(seller_x_y, dtype=torch.float32)
        
        # Convert to PyTorch tensors
        seller_x_tensor = torch.tensor(seller_x, dtype=torch.float32)
        seller_x_x_tensor = torch.tensor(seller_x_x, dtype=torch.float32)
        seller_count_tensor = torch.tensor(
            seller_count, dtype=torch.int).view(-1, 1)
        seller_1_tensor = seller_count_tensor.expand(-1, num_features)

        return seller_x_tensor, seller_x_x_tensor, seller_1_tensor, seller_x_y_tensor
    
    """
    This function gets the batch_id and the feature_index in this batch. These are all found in the searchEngine class.
    @param batch_id: the batch_id of the feature_index
    @param feature_index: the feature_index of the feature in the batch

    @return: a tuple containing the df_id and a feature name indicated by the dfid_feature_mapping
    """
    def get_df_by_feature_index(self, batch_id, feature_index):
        # Perform a binary search to find the right interval
        # bisect.bisect returns the insertion point which gives us the index where the feature_index would be inserted to maintain order.
        # We subtract one to get the tuple corresponding to the start index of the range that the feature_index falls into.
        def bisect(a, x):
            lo, hi = 0, len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if x < a[mid][0]:  # Compare with the first element of the tuple at mid
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        index = bisect(self.feature_index_mapping[batch_id], feature_index) - 1
        start_index, df_id, offset = self.feature_index_mapping[batch_id][index]
        # Calculate the local feature index within the seller's dataset
        local_feature_index = feature_index - start_index + offset
        return df_id, self.dfid_feature_mapping[df_id][local_feature_index]
    
    def get_sketch_loader(self):
        return self.sketch_loader